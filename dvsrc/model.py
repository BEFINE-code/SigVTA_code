from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import convnext_small, convnext_tiny, resnet18

from .config import ModelConfig


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def bin_pool(x: Tensor, mask: Tensor, bins: int) -> tuple[Tensor, Tensor]:
    """Order-preserving variable-length pooling without interpolation."""
    outputs, masks = [], []
    for token_index in range(bins):
        length = x.shape[1]
        start = math.floor(token_index * length / bins)
        end = max(start + 1, math.floor((token_index + 1) * length / bins))
        end = min(end, length)
        local_mask = mask[:, start:end]
        outputs.append(masked_mean(x[:, start:end], local_mask, dim=1))
        masks.append(local_mask.any(dim=1))
    return torch.stack(outputs, dim=1), torch.stack(masks, dim=1)


def temporal_resample(x: Tensor, mask: Tensor, tokens: int) -> tuple[Tensor, Tensor]:
    """Resample each valid prefix independently without pooling padded values."""
    batch, padded_length, dim = x.shape
    lengths = mask.sum(dim=1).clamp_min(1)
    counts = lengths.clamp_max(tokens)
    token_index = torch.arange(tokens, device=x.device)[None]
    output_mask = token_index < counts[:, None]
    denominator = (counts - 1).clamp_min(1).to(x.dtype)
    position = token_index.to(x.dtype) * (lengths - 1).to(x.dtype)[:, None] / denominator[:, None]
    position = position.masked_fill(~output_mask, 0)
    if padded_length == 1:
        grid_x = torch.full_like(position, -1)
    else:
        grid_x = position / (padded_length - 1) * 2 - 1
    grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1).unsqueeze(1)
    values = F.grid_sample(
        x.transpose(1, 2).unsqueeze(2), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).squeeze(2).transpose(1, 2)
    return values * output_mask.unsqueeze(-1), output_mask


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MaskedBatchNorm1d(nn.Module):
    """Batch normalization over valid temporal tokens only."""

    def __init__(self, channels: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        valid = mask[:, None, :].to(x.dtype)
        if self.training:
            count = valid.sum().clamp_min(1.0)
            mean = (x * valid).sum(dim=(0, 2)) / count
            variance = ((x - mean[None, :, None]).square() * valid).sum(dim=(0, 2)) / count
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                correction = count / (count - 1).clamp_min(1.0)
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_((variance * correction).detach(), self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (x - mean[None, :, None]) * torch.rsqrt(variance[None, :, None] + self.eps)
        output = normalized * self.weight[None, :, None] + self.bias[None, :, None]
        return output * valid


class RelativeSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, max_distance: int = 128):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance + 1, heads)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        positions = torch.arange(length, device=x.device)
        distance = (positions[None, :] - positions[:, None]).clamp(-self.max_distance, self.max_distance)
        bias = self.relative_bias(distance + self.max_distance).permute(2, 0, 1)
        logits = logits + bias.unsqueeze(0)
        logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attention = self.dropout(logits.softmax(dim=-1))
        output = torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, dim)
        return self.out(output) * mask.unsqueeze(-1)


class ConformerConv(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.batch_norm = MaskedBatchNorm1d(dim)
        self.pointwise_out = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        value = self.norm(x).transpose(1, 2)
        value = F.glu(self.pointwise_in(value), dim=1)
        value = F.silu(self.batch_norm(self.depthwise(value), mask))
        value = self.dropout(self.pointwise_out(value)).transpose(1, 2)
        return value * mask.unsqueeze(-1)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, kernel: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(dim, ffn, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = RelativeSelfAttention(dim, heads, dropout)
        self.conv = ConformerConv(dim, kernel, dropout)
        self.ff2 = FeedForward(dim, ffn, dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.conv(x, mask)
        x = self.out_norm(x + 0.5 * self.ff2(x))
        return x * mask.unsqueeze(-1)


class RawTrajectoryStem(nn.Module):
    """Learn local dynamics directly from bounded physical trajectory primitives."""

    def __init__(self, feature_dim: int, dim: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Conv1d(feature_dim, dim, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
                nn.Conv1d(dim, dim, 1),
                nn.GELU(),
            )
            for kernel in (3, 7, 15)
        ])
        self.fuse = nn.Conv1d(dim * 3, dim * 2, 1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.subsample = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=5, stride=2, padding=2),
            nn.Conv1d(dim, dim, kernel_size=5, stride=2, padding=2),
        ])

    def forward(self, sequence: Tensor, mask: Tensor, output_tokens: int) -> tuple[Tensor, Tensor]:
        value = self.input_projection(sequence.transpose(1, 2)) * mask.unsqueeze(1)
        branches = [branch(value) for branch in self.branches]
        candidate, gate = self.fuse(torch.cat(branches, dim=1)).chunk(2, dim=1)
        value = value + self.dropout(candidate * torch.sigmoid(gate))
        value = self.norm(value.transpose(1, 2)).transpose(1, 2) * mask.unsqueeze(1)
        lengths = mask.sum(dim=1)
        for layer in self.subsample:
            value = F.gelu(layer(value))
            lengths = (lengths + 1) // 2
            current_mask = torch.arange(value.shape[-1], device=value.device)[None] < lengths[:, None]
            value = value * current_mask.unsqueeze(1)
        value, current_mask = temporal_resample(value.transpose(1, 2), current_mask, output_tokens)
        return value, current_mask


class SequenceEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_dim
        self.stem_type = config.sequence_stem
        self.model_tokens = config.sequence_model_tokens
        if self.stem_type == "legacy_stride8":
            channels = [config.feature_dim, dim // 2, dim, dim]
            self.subsample = nn.ModuleList([
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=5, stride=2, padding=2)
                for i in range(3)
            ])
            self.raw_stem = None
        elif self.stem_type == "raw_multiscale":
            if config.feature_dim != 5:
                raise ValueError("raw_multiscale requires feature_dim=5")
            self.subsample = nn.ModuleList()
            self.raw_stem = RawTrajectoryStem(config.feature_dim, dim, config.dropout)
        else:
            raise ValueError(f"Unsupported sequence stem: {self.stem_type}")
        self.blocks = nn.ModuleList([
            ConformerBlock(dim, config.conformer_heads, config.conformer_ffn,
                           config.conformer_kernel, config.dropout)
            for _ in range(config.conformer_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output_tokens = config.sequence_tokens

    def _forward(
        self, sequence: Tensor, mask: Tensor,
        adapters: nn.ModuleList | None = None, collect_stages: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], dict[str, Tensor]]:
        if self.raw_stem is not None:
            x, token_mask = self.raw_stem(sequence, mask, self.model_tokens)
        else:
            x = sequence.transpose(1, 2)
            lengths = mask.sum(dim=1)
            for layer in self.subsample:
                x = F.gelu(layer(x))
                lengths = (lengths + 1) // 2
            x = x.transpose(1, 2)
            token_mask = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        stages: dict[str, Tensor] = {}
        if adapters is not None and len(adapters) != len(self.blocks):
            raise ValueError("Sequence adapter count must match Conformer block count")
        for index, block in enumerate(self.blocks):
            x = block(x, token_mask)
            if adapters is not None:
                x = adapters[index](x) * token_mask.unsqueeze(-1)
            if collect_stages:
                stages[f"sequence.block.{index}"] = x
        x = self.norm(x)
        if self.raw_stem is not None:
            local, local_mask = temporal_resample(x, token_mask, self.output_tokens)
        else:
            local, local_mask = bin_pool(x, token_mask, self.output_tokens)
        global_token = masked_mean(local, local_mask, dim=1)
        if collect_stages:
            stages["sequence.local"] = local
            stages["sequence.global"] = global_token
        return (global_token, local, local_mask, token_mask), stages

    def forward(self, sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output, _ = self._forward(sequence, mask)
        return output

    def forward_adapted(
        self, sequence: Tensor, mask: Tensor, adapters: nn.ModuleList | None,
        collect_stages: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], dict[str, Tensor]]:
        return self._forward(sequence, mask, adapters, collect_stages)


class ImageEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.image_backbone == "convnext_tiny":
            backbone = convnext_tiny(weights=None)
            filename = "convnext_tiny-983f1562.pth"
            self.features = backbone.features
            stage3_channels, stage4_channels, self.stage3_index = 384, 768, 5
        elif config.image_backbone == "convnext_small":
            backbone = convnext_small(weights=None)
            filename = "convnext_small-0c510722.pth"
            self.features = backbone.features
            stage3_channels, stage4_channels, self.stage3_index = 384, 768, 5
        elif config.image_backbone == "resnet18":
            backbone = resnet18(weights=None)
            filename = "resnet18-f37072fd.pth"
            stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.features = nn.Sequential(stem, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
            stage3_channels, stage4_channels, self.stage3_index = 256, 512, 3
        else:
            raise ValueError(f"Unsupported image backbone: {config.image_backbone}")
        if config.pretrained:
            candidates = [
                Path(os.environ.get("DVSRC_MODEL_DIR", Path(__file__).resolve().parents[1] / "models")) / filename,
                Path("/root/autodl-tmp/sigproject/models") / filename,
                Path(torch.hub.get_dir()) / "checkpoints" / filename,
            ]
            checkpoint = next((path for path in candidates if path.exists()), None)
            if checkpoint is None:
                raise FileNotFoundError(
                    f"Pretrained image backbone is missing. Place {filename} in the project's models/ directory."
                )
            backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        dim = config.hidden_dim
        self.stage3_projection = nn.Conv2d(stage3_channels, dim, 1)
        self.stage4_projection = nn.Conv2d(stage4_channels, dim, 1)
        self.global_projection = nn.Sequential(nn.LayerNorm(stage4_channels), nn.Linear(stage4_channels, dim))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _forward(
        self, image: Tensor, global_adapter: nn.Module | None = None,
        stage3_adapter: nn.Module | None = None, stage4_adapter: nn.Module | None = None,
        collect_stages: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], dict[str, Tensor]]:
        x = (image - self.image_mean) / self.image_std
        stage3 = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == self.stage3_index:
                stage3 = x
        assert stage3 is not None
        global_token = self.global_projection(x.mean(dim=(-2, -1)))
        projected3 = self.stage3_projection(stage3)
        projected4 = self.stage4_projection(x)
        if global_adapter is not None:
            global_token = global_adapter(global_token)
        if stage3_adapter is not None:
            projected3 = stage3_adapter(projected3)
        if stage4_adapter is not None:
            projected4 = stage4_adapter(projected4)
        stages = {}
        if collect_stages:
            stages = {
                "image.global": global_token,
                "image.stage3": projected3,
                "image.stage4": projected4,
            }
        return (global_token, projected3, projected4), stages

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output, _ = self._forward(image)
        return output

    def forward_adapted(
        self, image: Tensor, global_adapter: nn.Module | None,
        stage3_adapter: nn.Module | None, stage4_adapter: nn.Module | None,
        collect_stages: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], dict[str, Tensor]]:
        return self._forward(
            image, global_adapter, stage3_adapter, stage4_adapter, collect_stages,
        )


class MultiScaleCoordinateFusion(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.offset = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 10), nn.Tanh())
        self.weight = nn.Sequential(nn.Linear(dim * 2 + 6, dim), nn.GELU(), nn.Linear(dim, 5))
        self.image_projection = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2 + 7, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _sample(feature: Tensor, grid: Tensor) -> Tensor:
        # feature [B,D,H,W], grid [B,K,5,2]
        sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled.permute(0, 2, 3, 1)

    def _forward(
        self, temporal: Tensor, temporal_mask: Tensor, anchors: Tensor, anchor_valid: Tensor,
        stage3: Tensor, stage4: Tensor, image_global: Tensor,
        input_adapter: nn.Module | None = None,
        token_adapters: nn.ModuleList | None = None,
        collect_stages: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, dict[str, Tensor]], dict[str, Tensor]]:
        if input_adapter is not None:
            temporal = input_adapter(temporal) * temporal_mask.unsqueeze(-1)
        center = anchors[..., :2]
        extent = anchors[..., 2:4].clamp(0, 0.25) * 2
        base = torch.stack([
            torch.zeros_like(extent),
            torch.stack([extent[..., 0], torch.zeros_like(extent[..., 0])], -1),
            torch.stack([-extent[..., 0], torch.zeros_like(extent[..., 0])], -1),
            torch.stack([torch.zeros_like(extent[..., 1]), extent[..., 1]], -1),
            torch.stack([torch.zeros_like(extent[..., 1]), -extent[..., 1]], -1),
        ], dim=2)
        correction = self.offset(temporal).reshape(*temporal.shape[:2], 5, 2) * 0.08
        grid = (center.unsqueeze(2) + base + correction).clamp(-1, 1)
        local3, local4 = self._sample(stage3, grid), self._sample(stage4, grid)
        local = local3 + local4
        context = local.mean(dim=2)
        logits = self.weight(torch.cat([temporal, context, anchors], dim=-1))
        weights = logits.softmax(dim=-1)
        visual = (local * weights.unsqueeze(-1)).sum(dim=2)
        valid = temporal_mask & anchor_valid
        gate_input = torch.cat([temporal, visual, anchors, valid.unsqueeze(-1).to(temporal.dtype)], dim=-1)
        gate = self.gate(gate_input) * valid.unsqueeze(-1)
        fused = self.norm(temporal + gate * self.image_projection(visual))
        tokens = torch.cat([self.cls.expand(len(fused), -1, -1), fused, image_global.unsqueeze(1)], dim=1)
        mask = torch.cat([
            torch.ones(len(fused), 1, dtype=torch.bool, device=fused.device), temporal_mask,
            torch.ones(len(fused), 1, dtype=torch.bool, device=fused.device),
        ], dim=1)
        stages: dict[str, Tensor] = {}
        if collect_stages:
            stages["fusion.input"] = temporal
        if token_adapters is None and not collect_stages:
            tokens = self.transformer(tokens, src_key_padding_mask=~mask)
        else:
            if token_adapters is not None and len(token_adapters) != len(self.transformer.layers):
                raise ValueError("Fusion adapter count must match Transformer layer count")
            for index, layer in enumerate(self.transformer.layers):
                tokens = layer(tokens, src_key_padding_mask=~mask)
                if token_adapters is not None:
                    tokens = token_adapters[index](tokens) * mask.unsqueeze(-1)
                if collect_stages:
                    stages[f"fusion.layer.{index}"] = tokens
            if self.transformer.norm is not None:
                tokens = self.transformer.norm(tokens)
        diagnostics = {"fusion_gate": gate.squeeze(-1), "aligned_visual": visual, "anchor_valid": valid}
        if collect_stages:
            stages["fusion.global"] = tokens[:, 0]
            stages["fusion.local"] = tokens[:, 1:-1]
        return (tokens[:, 0], tokens[:, 1:-1], temporal_mask, diagnostics), stages

    def forward(self, temporal: Tensor, temporal_mask: Tensor, anchors: Tensor, anchor_valid: Tensor,
                stage3: Tensor, stage4: Tensor, image_global: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        output, _ = self._forward(
            temporal, temporal_mask, anchors, anchor_valid, stage3, stage4, image_global,
        )
        return output

    def forward_adapted(
        self, temporal: Tensor, temporal_mask: Tensor, anchors: Tensor, anchor_valid: Tensor,
        stage3: Tensor, stage4: Tensor, image_global: Tensor,
        input_adapter: nn.Module | None, token_adapters: nn.ModuleList | None,
        collect_stages: bool = False,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, dict[str, Tensor]], dict[str, Tensor]]:
        return self._forward(
            temporal, temporal_mask, anchors, anchor_valid, stage3, stage4, image_global,
            input_adapter, token_adapters, collect_stages,
        )


@dataclass
class SignatureEncoding:
    global_shared: Tensor
    local_shared: Tensor
    valid_mask: Tensor
    spatial: Tensor
    global_sequence: Tensor
    global_image: Tensor
    diagnostics: dict[str, Tensor]

    def select(self, index: Tensor) -> "SignatureEncoding":
        return SignatureEncoding(
            global_shared=self.global_shared[index], local_shared=self.local_shared[index],
            valid_mask=self.valid_mask[index], spatial=self.spatial[index],
            global_sequence=self.global_sequence[index], global_image=self.global_image[index],
            diagnostics={key: value[index] for key, value in self.diagnostics.items()},
        )

    def detach(self) -> "SignatureEncoding":
        return SignatureEncoding(
            global_shared=self.global_shared.detach(), local_shared=self.local_shared.detach(),
            valid_mask=self.valid_mask, spatial=self.spatial.detach(),
            global_sequence=self.global_sequence.detach(), global_image=self.global_image.detach(),
            diagnostics={key: value.detach() for key, value in self.diagnostics.items()},
        )

    def as_single_set(self) -> "SignatureEncoding":
        """Treat a flat list of encoded materials as one candidate set."""
        if self.global_shared.ndim != 2:
            raise ValueError("as_single_set expects flat [K,...] signature encodings")
        return SignatureEncoding(
            global_shared=self.global_shared.unsqueeze(0),
            local_shared=self.local_shared.unsqueeze(0),
            valid_mask=self.valid_mask.unsqueeze(0),
            spatial=self.spatial.unsqueeze(0),
            global_sequence=self.global_sequence.unsqueeze(0),
            global_image=self.global_image.unsqueeze(0),
            diagnostics={key: value.unsqueeze(0) for key, value in self.diagnostics.items()},
        )


@dataclass
class BayesianEvidenceState:
    """Encoded query plus the external candidate materials observed so far."""

    query: SignatureEncoding
    candidates: SignatureEncoding | None
    set_mask: Tensor | None
    posterior: Tensor
    posterior_history: tuple[Tensor, ...]


@dataclass
class DualSignatureEncoding:
    """Independent material encodings for the V10 rank and open-set paths."""

    rank: SignatureEncoding
    open: SignatureEncoding


@dataclass
class DualBayesianEvidenceState:
    query: DualSignatureEncoding
    candidates: DualSignatureEncoding | None
    set_mask: Tensor | None
    posterior: Tensor
    posterior_history: tuple[Tensor, ...]


class SignatureEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.sequence = SequenceEncoder(config)
        self.image = ImageEncoder(config)
        self.fusion = MultiScaleCoordinateFusion(config.hidden_dim, config.conformer_heads, config.dropout)
        self.late = nn.Sequential(nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim))
        if config.t1_variant == "simple_v2":
            self.sequence_output_norm = nn.LayerNorm(config.hidden_dim)
            self.fusion_output_norm = nn.LayerNorm(config.hidden_dim)
            self.residual_output_norm = nn.LayerNorm(config.hidden_dim)
            self.fusion_mix_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def _forward(
        self, sequence: Tensor, sequence_mask: Tensor, image: Tensor,
        anchors: Tensor, anchor_mask: Tensor,
        task_adapter: "T2InternalAdapter | None" = None,
        collect_stages: bool = False,
    ) -> tuple[SignatureEncoding, dict[str, Tensor]]:
        need_sequence = self.config.fusion != "image_only"
        need_image = self.config.fusion != "sequence_only"
        stages: dict[str, Tensor] = {}
        if need_sequence:
            sequence_output, sequence_stages = self.sequence.forward_adapted(
                sequence, sequence_mask,
                task_adapter.sequence if task_adapter is not None else None,
                collect_stages,
            )
            z_seq, h_seq, h_mask, _ = sequence_output
            stages.update(sequence_stages)
        else:
            z_seq = h_seq = h_mask = None
        if self.config.sequence_stem == "raw_multiscale":
            pooled_anchor, pooled_anchor_mask = temporal_resample(
                anchors, anchor_mask, self.config.sequence_tokens,
            )
        else:
            pooled_anchor, pooled_anchor_mask = bin_pool(anchors, anchor_mask, self.config.sequence_tokens)
        if need_image:
            image_outputs = []
            image_stage_outputs: list[dict[str, Tensor]] = []
            for start in range(0, len(image), self.config.image_chunk_size):
                output, chunk_stages = self.image.forward_adapted(
                    image[start:start + self.config.image_chunk_size],
                    task_adapter.image_global if task_adapter is not None else None,
                    task_adapter.image_stage3 if task_adapter is not None else None,
                    task_adapter.image_stage4 if task_adapter is not None else None,
                    collect_stages,
                )
                image_outputs.append(output)
                image_stage_outputs.append(chunk_stages)
            z_img = torch.cat([output[0] for output in image_outputs])
            stage3 = torch.cat([output[1] for output in image_outputs])
            stage4 = torch.cat([output[2] for output in image_outputs])
            if collect_stages:
                for key in image_stage_outputs[0]:
                    stages[key] = torch.cat([chunk[key] for chunk in image_stage_outputs])
        else:
            z_img = stage3 = stage4 = None
        if self.config.fusion == "ms_caf":
            assert h_seq is not None and h_mask is not None
            assert stage3 is not None and stage4 is not None and z_img is not None
            fusion_output, fusion_stages = self.fusion.forward_adapted(
                h_seq, h_mask, pooled_anchor, pooled_anchor_mask, stage3, stage4, z_img,
                task_adapter.fusion_input if task_adapter is not None else None,
                task_adapter.fusion_tokens if task_adapter is not None else None,
                collect_stages,
            )
            z_shared, h_shared, valid, diagnostics = fusion_output
            stages.update(fusion_stages)
        elif self.config.fusion == "sequence_only":
            assert z_seq is not None and h_seq is not None and h_mask is not None
            z_shared, h_shared, valid, diagnostics = z_seq, h_seq, h_mask, {}
            z_img = torch.zeros_like(z_seq)
        elif self.config.fusion == "image_only":
            assert z_img is not None and stage3 is not None
            z_shared = z_img
            h_shared = F.adaptive_avg_pool2d(stage3, (4, 4)).flatten(2).transpose(1, 2)
            valid = torch.ones(h_shared.shape[:2], dtype=torch.bool, device=h_shared.device)
            pooled_anchor = torch.zeros(*h_shared.shape[:2], 6, device=h_shared.device)
            diagnostics = {}
            z_seq = torch.zeros_like(z_img)
        else:
            assert z_seq is not None and h_seq is not None and h_mask is not None and z_img is not None
            z_shared, h_shared, valid = self.late(torch.cat([z_seq, z_img], -1)), h_seq, h_mask
            diagnostics = {}
        assert z_seq is not None and z_img is not None
        if self.config.t1_variant == "simple_v2" and self.config.t1_sequence_residual:
            if self.config.fusion in {"ms_caf", "late"}:
                alpha = self.fusion_mix_logit.sigmoid()
                z_shared = self.residual_output_norm(
                    self.sequence_output_norm(z_seq) + alpha * self.fusion_output_norm(z_shared)
                )
                diagnostics["fusion_mix"] = alpha.expand(len(z_shared))
            elif self.config.fusion == "sequence_only":
                z_shared = self.residual_output_norm(self.sequence_output_norm(z_seq))
            else:
                z_shared = self.residual_output_norm(self.fusion_output_norm(z_img))
        encoding = SignatureEncoding(
            z_shared, h_shared, valid, pooled_anchor[..., :2], z_seq, z_img, diagnostics,
        )
        if collect_stages:
            stages.update({
                "encoding.global_shared": z_shared,
                "encoding.local_shared": h_shared,
                "encoding.global_sequence": z_seq,
                "encoding.global_image": z_img,
            })
        return encoding, stages

    def forward(self, sequence: Tensor, sequence_mask: Tensor, image: Tensor,
                anchors: Tensor, anchor_mask: Tensor,
                task_adapter: "T2InternalAdapter | None" = None) -> SignatureEncoding:
        encoding, _ = self._forward(
            sequence, sequence_mask, image, anchors, anchor_mask, task_adapter,
        )
        return encoding

    def forward_with_intermediates(
        self, sequence: Tensor, sequence_mask: Tensor, image: Tensor,
        anchors: Tensor, anchor_mask: Tensor,
        task_adapter: "T2InternalAdapter | None" = None,
    ) -> tuple[SignatureEncoding, dict[str, Tensor]]:
        return self._forward(
            sequence, sequence_mask, image, anchors, anchor_mask,
            task_adapter, collect_stages=True,
        )


class TaskAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))


class SpatialTaskAdapter(nn.Module):
    """Zero-initialized channel adapter for projected image feature maps."""

    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.down = nn.Conv2d(dim, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))


class T2InternalAdapter(nn.Module):
    """One T2 task-delta package distributed through the shared encoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_dim
        bottleneck = config.t2_adapter_bottleneck
        if bottleneck < 1:
            raise ValueError("T2 adapter bottleneck must be positive")
        self.sequence = nn.ModuleList([
            TaskAdapter(dim, bottleneck) for _ in range(config.conformer_layers)
        ])
        self.image_global = TaskAdapter(dim, bottleneck)
        self.image_stage3 = SpatialTaskAdapter(dim, bottleneck)
        self.image_stage4 = SpatialTaskAdapter(dim, bottleneck)
        self.fusion_input = TaskAdapter(dim, bottleneck)
        self.fusion_tokens = nn.ModuleList([
            TaskAdapter(dim, bottleneck) for _ in range(2)
        ])


class T2ResidualAdapter(nn.Module):
    """Small T2-only correction on top of the frozen shared signature encoder."""

    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        if bottleneck < 1:
            raise ValueError("T2 adapter bottleneck must be positive")
        self.global_shared = TaskAdapter(dim, bottleneck)
        self.global_sequence = TaskAdapter(dim, bottleneck)
        self.global_image = TaskAdapter(dim, bottleneck)
        self.local_shared = TaskAdapter(dim, bottleneck)

    def forward(self, encoding: SignatureEncoding) -> SignatureEncoding:
        local = self.local_shared(encoding.local_shared)
        local = torch.where(encoding.valid_mask.unsqueeze(-1), local, encoding.local_shared)
        return SignatureEncoding(
            global_shared=self.global_shared(encoding.global_shared),
            local_shared=local,
            valid_mask=encoding.valid_mask,
            spatial=encoding.spatial,
            global_sequence=self.global_sequence(encoding.global_sequence),
            global_image=self.global_image(encoding.global_image),
            diagnostics=encoding.diagnostics,
        )


class SourceLocalAdapter(nn.Module):
    """T2-private token adaptation without changing the shared representation."""

    def __init__(self, dim: int, bottleneck: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        layer = nn.TransformerEncoderLayer(
            bottleneck, heads, bottleneck * 2, dropout,
            batch_first=True, norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        shape = x.shape
        tokens = F.gelu(self.down(self.norm(x))).reshape(-1, shape[-2], self.down.out_features)
        mask = valid_mask.reshape(-1, shape[-2])
        tokens = self.mixer(tokens, src_key_padding_mask=~mask)
        residual = self.up(tokens).reshape(shape)
        return x + residual * valid_mask.unsqueeze(-1)


class PairRelation(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 6, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, reference_z: Tensor, query_z: Tensor, reference_h: Tensor, query_h: Tensor,
                reference_mask: Tensor, query_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        reference_h = F.normalize(reference_h, dim=-1)
        query_h = F.normalize(query_h, dim=-1)
        similarity = torch.einsum("...id,...jd->...ij", reference_h, query_h)
        pair_mask = reference_mask.unsqueeze(-1) & query_mask.unsqueeze(-2)
        masked = similarity.masked_fill(~pair_mask, -1e4)
        r_best = masked.max(dim=-1).values.masked_fill(~reference_mask, 0)
        q_best = masked.max(dim=-2).values.masked_fill(~query_mask, 0)
        r_mean = (r_best * reference_mask).sum(-1) / reference_mask.sum(-1).clamp_min(1)
        q_mean = (q_best * query_mask).sum(-1) / query_mask.sum(-1).clamp_min(1)
        coverage_r = ((r_best > 0.25) & reference_mask).sum(-1) / reference_mask.sum(-1).clamp_min(1)
        coverage_q = ((q_best > 0.25) & query_mask).sum(-1) / query_mask.sum(-1).clamp_min(1)
        valid_values = similarity.masked_fill(~pair_mask, 0)
        mean = valid_values.sum((-2, -1)) / pair_mask.sum((-2, -1)).clamp_min(1)
        variance = ((valid_values - mean[..., None, None]) ** 2 * pair_mask).sum((-2, -1)) / pair_mask.sum((-2, -1)).clamp_min(1)
        entropy_proxy = variance.clamp_min(1e-6).sqrt()
        stats = torch.stack([(r_mean + q_mean) / 2, torch.minimum(r_mean, q_mean),
                             (coverage_r + coverage_q) / 2, torch.minimum(coverage_r, coverage_q),
                             entropy_proxy, (coverage_r - coverage_q).abs()], -1)
        relation = torch.cat([(reference_z - query_z).abs(), reference_z * query_z, stats], -1)
        embedding = self.mlp(relation)
        return embedding, self.head(embedding).squeeze(-1), stats


class GlobalPairRelation(nn.Module):
    """Symmetric pair scoring using only global difference and product."""

    def __init__(self, dim: int, dropout: float, features: str = "diff_product"):
        super().__init__()
        if features not in {"diff_product", "diff_only", "product_only"}:
            raise ValueError(f"Unsupported global pair features: {features}")
        self.features = features
        input_dim = dim * 2 if features == "diff_product" else dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, reference_z: Tensor, query_z: Tensor) -> tuple[Tensor, Tensor]:
        if self.features == "diff_only":
            features = (reference_z - query_z).abs()
        elif self.features == "product_only":
            features = reference_z * query_z
        else:
            features = torch.cat([(reference_z - query_z).abs(), reference_z * query_z], -1)
        embedding = self.mlp(features)
        return embedding, self.head(embedding).squeeze(-1)


class StabilizedGlobalPairRelation(GlobalPairRelation):
    """Same 2D projection in every arm; feature ablations zero one half."""

    def __init__(self, dim: int, dropout: float, features: str, stabilize: bool):
        super().__init__(dim, dropout, "diff_product")
        if features not in {"diff_product", "diff_only", "product_only"}:
            raise ValueError(f"Unsupported global pair features: {features}")
        self.features = features
        self.stabilize = stabilize
        self.input_norm = nn.LayerNorm(dim)
        self.product_mix_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def forward(self, reference_z: Tensor, query_z: Tensor) -> tuple[Tensor, Tensor]:
        if self.stabilize:
            reference_z, query_z = self.input_norm(reference_z), self.input_norm(query_z)
        difference = (reference_z - query_z).abs()
        product = reference_z * query_z
        if self.stabilize:
            product = self.product_mix_logit.sigmoid() * product
        if self.features == "diff_only":
            product = torch.zeros_like(product)
        elif self.features == "product_only":
            difference = torch.zeros_like(difference)
        embedding = self.mlp(torch.cat([difference, product], dim=-1))
        return embedding, self.head(embedding).squeeze(-1)


class SortedLogitSetHead(nn.Module):
    """A permutation-invariant 113-parameter classifier for five pair logits."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, pair_logits: Tensor, set_mask: Tensor) -> Tensor:
        if pair_logits.ndim != 2 or pair_logits.shape[1] != 5:
            raise ValueError("The simple 5v1 head requires exactly five pair logits per episode")
        if set_mask.shape != pair_logits.shape or set_mask.dtype != torch.bool:
            raise ValueError("set_mask must be boolean with the same shape as pair_logits")
        if not bool(set_mask.all()):
            raise ValueError("The simple 5v1 head requires five valid references per episode")
        sorted_logits = pair_logits.sort(dim=1).values
        return self.mlp(sorted_logits).squeeze(-1)


class SharedPairEvidenceMatcher(nn.Module):
    """One global/local pair matcher shared by T1 verification and T2 retrieval."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(dim * 4 + 6),
            nn.Linear(dim * 4 + 6, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, 1),
        )

    @staticmethod
    def _relation_statistics(
        reference_h: Tensor, query_h: Tensor, reference_mask: Tensor, query_mask: Tensor,
    ) -> Tensor:
        similarity = torch.einsum(
            "...id,...jd->...ij", F.normalize(reference_h, dim=-1), F.normalize(query_h, dim=-1),
        )
        pair_mask = reference_mask.unsqueeze(-1) & query_mask.unsqueeze(-2)
        masked = similarity.masked_fill(~pair_mask, -1e4)
        reference_best = masked.max(dim=-1).values.masked_fill(~reference_mask, 0)
        query_best = masked.max(dim=-2).values.masked_fill(~query_mask, 0)
        reference_mean = (
            (reference_best * reference_mask).sum(-1) / reference_mask.sum(-1).clamp_min(1)
        )
        query_mean = (query_best * query_mask).sum(-1) / query_mask.sum(-1).clamp_min(1)
        reference_coverage = (
            ((reference_best > 0.25) & reference_mask).sum(-1)
            / reference_mask.sum(-1).clamp_min(1)
        )
        query_coverage = (
            ((query_best > 0.25) & query_mask).sum(-1)
            / query_mask.sum(-1).clamp_min(1)
        )
        valid_values = similarity.masked_fill(~pair_mask, 0)
        mean = valid_values.sum((-2, -1)) / pair_mask.sum((-2, -1)).clamp_min(1)
        variance = (
            (valid_values - mean[..., None, None]).square() * pair_mask
        ).sum((-2, -1)) / pair_mask.sum((-2, -1)).clamp_min(1)
        return torch.stack([
            0.5 * (reference_mean + query_mean),
            torch.minimum(reference_mean, query_mean),
            0.5 * (reference_coverage + query_coverage),
            torch.minimum(reference_coverage, query_coverage),
            variance.clamp_min(1e-6).sqrt(),
            (reference_coverage - query_coverage).abs(),
        ], dim=-1)

    def forward(
        self, reference: SignatureEncoding, query: SignatureEncoding,
    ) -> dict[str, Tensor]:
        reference_z = reference.global_shared
        query_z = query.global_shared[:, None].expand_as(reference_z)
        reference_h = reference.local_shared
        query_h = query.local_shared[:, None].expand(
            -1, reference_h.shape[1], -1, -1,
        )
        query_mask = query.valid_mask[:, None].expand(
            -1, reference_h.shape[1], -1,
        )
        shape = reference_h.shape
        flat_reference = reference_h.reshape(-1, shape[-2], shape[-1])
        flat_query = query_h.reshape(-1, shape[-2], shape[-1])
        flat_reference_mask = reference.valid_mask.reshape(-1, shape[-2])
        flat_query_mask = query_mask.reshape(-1, shape[-2])
        query_to_reference, _ = self.cross_attention(
            flat_query, flat_reference, flat_reference,
            key_padding_mask=~flat_reference_mask, need_weights=False,
        )
        reference_to_query, _ = self.cross_attention(
            flat_reference, flat_query, flat_query,
            key_padding_mask=~flat_query_mask, need_weights=False,
        )
        query_context = masked_mean(query_to_reference, flat_query_mask, dim=1).reshape(*shape[:2], -1)
        reference_context = masked_mean(
            reference_to_query, flat_reference_mask, dim=1,
        ).reshape(*shape[:2], -1)
        statistics = self._relation_statistics(
            reference_h, query_h, reference.valid_mask, query_mask,
        )
        features = torch.cat([
            (reference_z - query_z).abs(), reference_z * query_z,
            reference_context, query_context, statistics,
        ], dim=-1)
        embedding = self.projection(features)
        return {
            "pair_embedding": embedding,
            "pair_logits": self.score(embedding).squeeze(-1),
            "relation_stats": statistics,
        }


class T2SourceEvidenceAdapter(nn.Module):
    """Compact T2-only residual that specializes shared pair evidence for source ranking."""

    def __init__(self, dim: int, bottleneck: int, heads: int, dropout: float):
        super().__init__()
        self.residual = TaskAdapter(dim, bottleneck)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 2, dropout, batch_first=True, norm_first=True,
        )
        self.set_mixer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, pair: Tensor, set_mask: Tensor) -> Tensor:
        adapted = self.residual(pair)
        adapted = self.set_mixer(adapted, src_key_padding_mask=~set_mask)
        return self.norm(adapted) * set_mask.unsqueeze(-1)


class QRSA(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.self_quality = nn.Sequential(nn.Linear(dim + 2, 128), nn.GELU(), nn.Linear(128, 1))
        self.match_quality = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 1))
        self.case_query = nn.Linear(dim, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.set_transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(dim * 3 + 2, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, 1))

    def quality_logits(self, reference_z: Tensor, reference_h: Tensor, reference_mask: Tensor) -> Tensor:
        token_mean = masked_mean(reference_h, reference_mask, dim=-2).unsqueeze(-2)
        dispersion = masked_mean((reference_h - token_mean).square(), reference_mask, dim=-2).mean(-1)
        valid_ratio = reference_mask.float().mean(-1)
        return self.self_quality(
            torch.cat([reference_z, dispersion.unsqueeze(-1), valid_ratio.unsqueeze(-1)], -1)
        ).squeeze(-1)

    def forward(self, pair: Tensor, pair_logits: Tensor, reference_z: Tensor, reference_h: Tensor,
                reference_mask: Tensor, relation_stats: Tensor, query_z: Tensor, set_mask: Tensor) -> dict[str, Tensor]:
        quality_logit = self.quality_logits(reference_z, reference_h, reference_mask)
        # Coverage and entropy only; no same/different directional similarity.
        match_inputs = torch.stack([relation_stats[..., 2], relation_stats[..., 3],
                                    1 - relation_stats[..., 4].clamp(0, 1),
                                    1 - relation_stats[..., 5].clamp(0, 1)], -1)
        match_logit = self.match_quality(match_inputs).squeeze(-1)
        reliability = torch.sigmoid(quality_logit) * torch.sigmoid(match_logit) * set_mask
        gated = pair * reliability.unsqueeze(-1)
        case = self.case_query(query_z).unsqueeze(1)
        tokens = torch.cat([case, gated], 1)
        token_mask = torch.cat([torch.ones(len(pair), 1, dtype=torch.bool, device=pair.device), set_mask], 1)
        case_token = self.set_transformer(tokens, src_key_padding_mask=~token_mask)[:, 0]
        denom = reliability.sum(1, keepdim=True).clamp_min(1e-6)
        weighted_mean = (pair * reliability.unsqueeze(-1)).sum(1) / denom
        variance = ((pair - weighted_mean.unsqueeze(1)).square() * reliability.unsqueeze(-1)).sum(1) / denom
        masked_logits = pair_logits.masked_fill(~set_mask, 0)
        score_range = masked_logits.masked_fill(~set_mask, -1e4).max(1).values - masked_logits.masked_fill(~set_mask, 1e4).min(1).values
        case_features = torch.cat([case_token, weighted_mean, variance, score_range.unsqueeze(-1),
                                   relation_stats[..., 4].mean(1, keepdim=True)], -1)
        return {
            "case_logit": self.head(case_features).squeeze(-1), "reliability": reliability,
            "quality_logit": quality_logit, "match_quality_logit": match_logit,
            "pair_variance": variance.mean(-1), "score_range": score_range,
        }


class OneToOneVerificationHead(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim + 6), nn.Linear(dim + 6, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, 1),
        )

    def forward(self, pair: Tensor, relation_stats: Tensor) -> Tensor:
        return self.classifier(torch.cat([pair[:, 0], relation_stats[:, 0]], dim=-1)).squeeze(-1)


class ResidualOneToOneVerificationHead(nn.Module):
    """Preserve the trained V1 pair score and learn only a protocol correction."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.correction = nn.Sequential(
            nn.LayerNorm(dim + 6), nn.Linear(dim + 6, dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim // 2, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, pair: Tensor, relation_stats: Tensor, base_logit: Tensor) -> Tensor:
        correction = self.correction(torch.cat([pair[:, 0], relation_stats[:, 0]], dim=-1)).squeeze(-1)
        return base_logit + correction


class EvidenceAnchoredSetHead(nn.Module):
    """Anchor set decisions to robust pair evidence and learn only a correction."""

    def __init__(self, dropout: float):
        super().__init__()
        self.correction = nn.Sequential(
            nn.LayerNorm(7), nn.Linear(7, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, set_logit: Tensor, pair_logits: Tensor, reliability: Tensor,
                set_mask: Tensor) -> tuple[Tensor, Tensor]:
        valid = set_mask.to(pair_logits.dtype)
        count = valid.sum(1).clamp_min(1)
        mean = (pair_logits * valid).sum(1) / count
        sorted_logits = pair_logits.masked_fill(~set_mask, torch.inf).sort(dim=1).values
        median_index = ((count.long() - 1) // 2).unsqueeze(1)
        median = sorted_logits.gather(1, median_index).squeeze(1)
        minimum = pair_logits.masked_fill(~set_mask, torch.inf).min(1).values
        maximum = pair_logits.masked_fill(~set_mask, -torch.inf).max(1).values
        variance = ((pair_logits - mean[:, None]).square() * valid).sum(1) / count
        reliability_mean = (reliability * valid).sum(1) / count
        anchor = 0.5 * (mean + median)
        features = torch.stack([
            set_logit, mean, median, minimum, maximum,
            variance.clamp_min(1e-6).sqrt(), reliability_mean,
        ], dim=-1)
        return anchor + self.correction(features).squeeze(-1), anchor


class T1Head(nn.Module):
    def __init__(self, config: ModelConfig, variant: str | None = None):
        super().__init__()
        dim = config.hidden_dim
        variant = variant or config.t1_variant or config.variant
        self.global_adapter = TaskAdapter(dim) if config.use_tsa else nn.Identity()
        self.variant = variant
        if variant in {"simple_v1", "simple_v2"}:
            if config.t1_set_aggregation not in {"sorted_mlp", "mean"}:
                raise ValueError(f"Unsupported simple T1 aggregation: {config.t1_set_aggregation}")
            self.relation = (
                StabilizedGlobalPairRelation(dim, config.dropout, config.t1_pair_features, config.t1_pair_stabilize)
                if variant == "simple_v2" else GlobalPairRelation(dim, config.dropout, config.t1_pair_features)
            )
            self.five_to_one = (
                SortedLogitSetHead() if config.t1_set_aggregation == "sorted_mlp" else None
            )
            return
        self.local_adapter = TaskAdapter(dim) if config.use_tsa else nn.Identity()
        self.relation = PairRelation(dim, config.dropout)
        self.one_to_one = OneToOneVerificationHead(dim, config.dropout)
        self.residual_one_to_one = (
            ResidualOneToOneVerificationHead(dim, config.dropout)
            if variant in {"v4", "v5", "v5r1"}
            and config.use_t1_residual_verification else None
        )
        self.qrsa = QRSA(dim, config.conformer_heads, config.dropout)
        self.evidence_anchored_set = (
            EvidenceAnchoredSetHead(config.dropout)
            if variant == "v5r1" and config.use_t1_evidence_anchor else None
        )
        self.use_qrsa = config.use_qrsa
        self.variant = variant

    def _forward_simple(
        self, reference: SignatureEncoding, query: SignatureEncoding, set_mask: Tensor,
    ) -> dict[str, Tensor]:
        ref_z, query_z = reference.global_shared, query.global_shared
        if ref_z.ndim != 3 or query_z.ndim != 2:
            raise ValueError("Simple T1 expects reference [B,N,d] and query [B,d]")
        if ref_z.shape[0] != query_z.shape[0] or ref_z.shape[2] != query_z.shape[1]:
            raise ValueError("Reference and query batch and feature dimensions must match")
        if ref_z.shape[1] not in (1, 5):
            raise ValueError("Simple T1 supports exactly one or five references")
        if set_mask.shape != ref_z.shape[:2] or set_mask.dtype != torch.bool:
            raise ValueError("set_mask must be boolean with shape [B,N]")
        if not bool(set_mask.all()):
            raise ValueError("Simple T1 requires every reference to be valid")
        ref_z = self.global_adapter(ref_z)
        query_z = self.global_adapter(query_z)
        pair, pair_logits = self.relation(ref_z, query_z[:, None].expand_as(ref_z))
        if ref_z.shape[1] == 1:
            case_logit = pair_logits[:, 0]
        elif self.five_to_one is None:
            case_logit = pair_logits.mean(dim=1)
        else:
            case_logit = self.five_to_one(pair_logits, set_mask)
        return {
            "pair_embedding": pair, "pair_logits": pair_logits,
            "case_logit": case_logit, "identity_embedding": query_z,
        }

    def forward(
        self, reference: SignatureEncoding, query: SignatureEncoding, set_mask: Tensor,
        shared_relation: SharedPairEvidenceMatcher | None = None,
    ) -> dict[str, Tensor]:
        # reference fields [B,N,...], query fields [B,...]
        if self.variant in {"simple_v1", "simple_v2"}:
            if shared_relation is not None:
                raise ValueError("Simple T1 requires its own global-only PairRelation")
            return self._forward_simple(reference, query, set_mask)
        if shared_relation is None:
            ref_z = self.global_adapter(reference.global_shared)
            query_z = self.global_adapter(query.global_shared)
            ref_h = self.local_adapter(reference.local_shared)
            query_h = self.local_adapter(query.local_shared)
            expanded_query_z = query_z.unsqueeze(1).expand_as(ref_z)
            expanded_query_h = query_h.unsqueeze(1).expand(-1, ref_h.shape[1], -1, -1)
            expanded_query_mask = query.valid_mask.unsqueeze(1).expand(-1, ref_h.shape[1], -1)
            pair, pair_logits, stats = self.relation(
                ref_z, expanded_query_z, ref_h, expanded_query_h,
                reference.valid_mask, expanded_query_mask,
            )
        else:
            ref_z, query_z = reference.global_shared, query.global_shared
            ref_h, query_h = reference.local_shared, query.local_shared
            relation_output = shared_relation(reference, query)
            pair = relation_output["pair_embedding"]
            pair_logits = relation_output["pair_logits"]
            stats = relation_output["relation_stats"]
        output = {"pair_embedding": pair, "pair_logits": pair_logits, "relation_stats": stats,
                  "identity_embedding": query_z}
        if ref_z.shape[1] == 1:
            if self.variant == "v3":
                case_logit = self.one_to_one(pair, stats)
            elif self.residual_one_to_one is not None:
                case_logit = self.residual_one_to_one(pair, stats, pair_logits[:, 0])
            else:
                case_logit = pair_logits[:, 0]
            output.update({"case_logit": case_logit, "reliability": set_mask.float(),
                           "quality_logit": torch.zeros_like(pair_logits),
                           "pair_variance": torch.zeros(len(pair), device=pair.device),
                           "score_range": torch.zeros(len(pair), device=pair.device)})
        elif self.use_qrsa:
            output.update(self.qrsa(pair, pair_logits, ref_z, ref_h, reference.valid_mask, stats, query_z, set_mask))
            if self.evidence_anchored_set is not None:
                set_aux_logit = output["case_logit"]
                case_logit, evidence_anchor_logit = self.evidence_anchored_set(
                    set_aux_logit, pair_logits, output["reliability"], set_mask,
                )
                output.update({
                    "case_logit": case_logit,
                    "set_aux_logit": set_aux_logit,
                    "evidence_anchor_logit": evidence_anchor_logit,
                })
            if self.training:
                full_probability = torch.sigmoid(output["case_logit"]).detach()
                subset_losses = []
                for subset_size in (3, 4):
                    if ref_z.shape[1] <= subset_size:
                        continue
                    random_order = torch.rand(ref_z.shape[:2], device=ref_z.device).argsort(dim=1)
                    subset_mask = torch.zeros_like(set_mask)
                    subset_mask.scatter_(1, random_order[:, :subset_size], True)
                    subset_mask &= set_mask
                    subset_output = self.qrsa(
                        pair, pair_logits, ref_z, ref_h, reference.valid_mask, stats, query_z, subset_mask,
                    )
                    subset_losses.append((torch.sigmoid(subset_output["case_logit"]) - full_probability).abs().mean())
                output["subset_loss"] = torch.stack(subset_losses).mean() if subset_losses else pair.new_zeros(())
                degraded_mask = reference.valid_mask & (torch.rand_like(reference.valid_mask.float()) > 0.12)
                # Preserve at least one local token for every reference.
                empty = ~degraded_mask.any(-1)
                degraded_mask[..., 0] |= empty
                degraded_h = ref_h + torch.randn_like(ref_h) * 0.03
                degraded_quality = self.qrsa.quality_logits(ref_z, degraded_h, degraded_mask)
                output["quality_rank_loss"] = F.relu(0.1 - output["quality_logit"] + degraded_quality).mean()
        else:
            masked = pair_logits.masked_fill(~set_mask, 0)
            output.update({"case_logit": masked.sum(1) / set_mask.sum(1).clamp_min(1),
                           "reliability": set_mask.float(), "quality_logit": torch.zeros_like(pair_logits),
                           "pair_variance": masked.var(1), "score_range": masked.max(1).values - masked.min(1).values})
        return output


class PartialInheritanceMatcher(nn.Module):
    def __init__(self, dim: int, iterations: int, epsilon: float):
        super().__init__()
        self.candidate = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.space = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1))
        self.beta_space_raw = nn.Parameter(torch.tensor(-2.0))
        self.dustbin = nn.Parameter(torch.tensor(0.0))
        self.iterations, self.epsilon = iterations, epsilon

    def forward(self, candidate: Tensor, query: Tensor, candidate_mask: Tensor, query_mask: Tensor,
                candidate_xy: Tensor, query_xy: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        original_shape = candidate.shape[:2]
        batch, count, tokens, dim = candidate.shape
        cand = candidate.reshape(batch * count, tokens, dim)
        qry = query[:, None].expand(-1, count, -1, -1).reshape(batch * count, query.shape[1], dim)
        cmask = candidate_mask.reshape(batch * count, tokens)
        qmask = query_mask[:, None].expand(-1, count, -1).reshape(batch * count, query.shape[1])
        cxy = candidate_xy.reshape(batch * count, tokens, 2)
        qxy = query_xy[:, None].expand(-1, count, -1, -1).reshape(batch * count, query.shape[1], 2)
        content = torch.einsum("bid,bjd->bij", F.normalize(self.candidate(cand), dim=-1), F.normalize(self.query(qry), dim=-1))
        spatial_delta = (cxy.unsqueeze(2) - qxy.unsqueeze(1)).abs()
        spatial = self.space(spatial_delta).squeeze(-1)
        similarity = content + F.softplus(self.beta_space_raw) * spatial
        pair_mask = cmask.unsqueeze(-1) & qmask.unsqueeze(-2)
        with torch.autocast(device_type=candidate.device.type, enabled=False):
            sim = similarity.float()
            n, m = sim.shape[-2:]
            augmented = torch.ones((len(sim), n + 1, m + 1), device=sim.device) * self.dustbin.float()
            augmented[:, :n, :m] = sim
            valid_aug = torch.cat([cmask, torch.ones(len(sim), 1, dtype=torch.bool, device=sim.device)], 1)
            valid_q_aug = torch.cat([qmask, torch.ones(len(sim), 1, dtype=torch.bool, device=sim.device)], 1)
            valid_matrix = valid_aug.unsqueeze(-1) & valid_q_aug.unsqueeze(-2)
            log_kernel = (augmented / self.epsilon).masked_fill(~valid_matrix, -1e4)
            mu = valid_aug.float() / valid_aug.sum(-1, keepdim=True).clamp_min(1)
            nu = valid_q_aug.float() / valid_q_aug.sum(-1, keepdim=True).clamp_min(1)
            log_mu, log_nu = torch.log(mu.clamp_min(1e-8)), torch.log(nu.clamp_min(1e-8))
            u = torch.zeros_like(log_mu)
            v = torch.zeros_like(log_nu)
            for _ in range(self.iterations):
                u = log_mu - torch.logsumexp(log_kernel + v.unsqueeze(1), dim=2)
                v = log_nu - torch.logsumexp(log_kernel + u.unsqueeze(2), dim=1)
            transport = torch.exp(log_kernel + u.unsqueeze(2) + v.unsqueeze(1)) * valid_matrix
            real = transport[:, :n, :m] * pair_mask
            real_mass = real.sum((1, 2)).clamp_min(1e-8)
            matched = (real * sim).sum((1, 2)) / real_mass
            candidate_coverage = real.sum((1, 2)) / transport[:, :n].sum((1, 2)).clamp_min(1e-8)
            query_coverage = real.sum((1, 2)) / transport[:, :, :m].sum((1, 2)).clamp_min(1e-8)
            best = sim.masked_fill(~pair_mask, -1e4).amax(dim=(-2, -1))
            entropy = -(transport.clamp_min(1e-8) * transport.clamp_min(1e-8).log()).sum((1, 2))
            dust_c = transport[:, :n, -1].sum(1)
            dust_q = transport[:, -1, :m].sum(1)
            residual = (real * spatial_delta.float().norm(dim=-1)).sum((1, 2)) / real_mass
            row_error = (transport.sum(2) - mu).abs().sum(1)
            col_error = (transport.sum(1) - nu).abs().sum(1)
            convergence = row_error + col_error
            stats = torch.stack([matched, candidate_coverage, query_coverage, best, entropy,
                                 dust_c, dust_q, residual], dim=-1).to(candidate.dtype)
        stats = stats.reshape(*original_shape, 8)
        diagnostics = {"sinkhorn_error": convergence.reshape(*original_shape),
                       "transport": transport.reshape(batch, count, n + 1, m + 1)}
        return stats, diagnostics


class OCSR(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.rank = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.exist = nn.Sequential(nn.Linear(dim + 9, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.temperature_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, pair: Tensor, pim: Tensor, set_mask: Tensor) -> dict[str, Tensor]:
        context = self.context(pair, src_key_padding_mask=~set_mask)
        rank_logits = self.rank(torch.cat([pair, context], -1)).squeeze(-1).masked_fill(~set_mask, -1e4)
        top = rank_logits.topk(k=min(2, rank_logits.shape[1]), dim=1).values
        gap = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else top[:, 0])
        valid_count = set_mask.sum(1).clamp_min(1)
        mean = rank_logits.masked_fill(~set_mask, 0).sum(1) / valid_count
        std = (((rank_logits - mean[:, None]) ** 2).masked_fill(~set_mask, 0).sum(1) / valid_count).sqrt()
        probabilities = rank_logits.softmax(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
        best_index = rank_logits.argmax(1)
        best_pim = pim[torch.arange(len(pim), device=pim.device), best_index]
        pooled_context = masked_mean(context, set_mask, dim=1)
        temperature = F.softplus(self.temperature_raw) + 0.05
        energy = -temperature * torch.logsumexp(rank_logits / temperature, dim=1)
        summary = torch.cat([top[:, :1], gap[:, None], mean[:, None], std[:, None], entropy[:, None],
                             best_pim[:, [1, 2, 5]], energy[:, None], pooled_context], -1)
        exist_logit = self.exist(summary).squeeze(-1)
        exist_probability = torch.sigmoid(exist_logit)
        rank_probability = rank_logits.softmax(1)
        joint = torch.cat([exist_probability[:, None] * rank_probability, (1 - exist_probability)[:, None]], 1)
        return {"rank_logits": rank_logits, "rank_probability": rank_probability,
                "exist_logit": exist_logit, "exist_probability": exist_probability,
                "unknown_probability": 1 - exist_probability, "joint_probability": joint,
                "energy": energy, "set_entropy": entropy}


class HierarchicalOCSR(nn.Module):
    """Separate absolute source evidence from conditional candidate ranking."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        evidence_dim = dim * 2 + 8
        self.rank = nn.Sequential(
            nn.Linear(evidence_dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, 1),
        )
        self.match = nn.Sequential(
            nn.Linear(evidence_dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, 1),
        )
        self.exist_residual = nn.Sequential(
            nn.Linear(dim + 14, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, 1),
        )

    def forward(self, pair: Tensor, pim: Tensor, set_mask: Tensor) -> dict[str, Tensor]:
        context = self.context(pair, src_key_padding_mask=~set_mask)
        evidence = torch.cat([pair, context, pim], dim=-1)
        match_logits = self.match(evidence).squeeze(-1).masked_fill(~set_mask, -1e4)
        rank_logits = (self.rank(evidence).squeeze(-1) + match_logits).masked_fill(~set_mask, -1e4)
        valid_count = set_mask.sum(1).clamp_min(1)
        top = match_logits.topk(k=min(2, match_logits.shape[1]), dim=1).values
        gap = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else top[:, 0])
        match_mean = match_logits.masked_fill(~set_mask, 0).sum(1) / valid_count
        match_std = (
            ((match_logits - match_mean[:, None]).square()).masked_fill(~set_mask, 0).sum(1) / valid_count
        ).sqrt()
        rank_probability = rank_logits.softmax(1)
        rank_entropy = -(rank_probability * rank_probability.clamp_min(1e-8).log()).sum(1)
        evidence_lse = torch.logsumexp(match_logits, dim=1) - valid_count.float().log()
        best_index = match_logits.argmax(1)
        best_pim = pim[torch.arange(len(pim), device=pim.device), best_index]
        pooled_context = masked_mean(context, set_mask, dim=1)
        summary = torch.cat([
            top[:, :1], gap[:, None], match_mean[:, None], match_std[:, None],
            rank_entropy[:, None], evidence_lse[:, None], best_pim, pooled_context,
        ], dim=-1)
        exist_logit = evidence_lse + self.exist_residual(summary).squeeze(-1)
        exist_probability = torch.sigmoid(exist_logit)
        joint = torch.cat([
            exist_probability[:, None] * rank_probability,
            (1 - exist_probability)[:, None],
        ], dim=1)
        return {
            "rank_logits": rank_logits, "rank_probability": rank_probability,
            "match_logits": match_logits, "exist_logit": exist_logit,
            "exist_probability": exist_probability, "unknown_probability": 1 - exist_probability,
            "joint_probability": joint, "set_entropy": rank_entropy, "evidence_lse": evidence_lse,
        }


class OpenSetEvidenceOCSR(nn.Module):
    """Let candidates and an explicit Unknown class compete in one open-set space."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        evidence_dim = dim * 2 + 8 + 3
        self.match = nn.Sequential(
            nn.Linear(evidence_dim, dim), nn.GELU(), nn.LayerNorm(dim),
            nn.Dropout(dropout), nn.Linear(dim, 1),
        )
        self.rank_residual = nn.Sequential(
            nn.Linear(evidence_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1),
        )
        self.metric_scale_raw = nn.Parameter(torch.tensor([1.5, 0.75, 0.75]))
        self.unknown = nn.Sequential(
            nn.Linear(dim + 16, dim), nn.GELU(), nn.LayerNorm(dim),
            nn.Dropout(dropout), nn.Linear(dim, 1),
        )

    def forward(self, pair: Tensor, pim: Tensor, set_mask: Tensor,
                absolute_similarity: Tensor) -> dict[str, Tensor]:
        context = self.context(pair, src_key_padding_mask=~set_mask)
        evidence = torch.cat([pair, context, pim, absolute_similarity], dim=-1)
        metric_scale = F.softplus(self.metric_scale_raw)
        metric_evidence = (absolute_similarity * metric_scale).sum(dim=-1)
        match_logits = (self.match(evidence).squeeze(-1) + metric_evidence).masked_fill(~set_mask, -1e4)
        rank_logits = (
            match_logits + self.rank_residual(evidence).squeeze(-1)
        ).masked_fill(~set_mask, -1e4)

        valid_count = set_mask.sum(1).clamp_min(1)
        top = match_logits.topk(k=min(2, match_logits.shape[1]), dim=1).values
        gap = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else top[:, 0])
        match_mean = match_logits.masked_fill(~set_mask, 0).sum(1) / valid_count
        match_std = (
            (match_logits - match_mean[:, None]).square().masked_fill(~set_mask, 0).sum(1) / valid_count
        ).sqrt()
        rank_probability = rank_logits.softmax(1)
        rank_entropy = -(rank_probability * rank_probability.clamp_min(1e-8).log()).sum(1)
        evidence_lme = torch.logsumexp(match_logits, dim=1) - valid_count.float().log()
        absolute_mean = absolute_similarity[..., 0].masked_fill(~set_mask, 0).sum(1) / valid_count
        absolute_std = (
            (absolute_similarity[..., 0] - absolute_mean[:, None]).square().masked_fill(~set_mask, 0).sum(1)
            / valid_count
        ).sqrt()
        best_index = match_logits.argmax(1)
        best_pim = pim[torch.arange(len(pim), device=pim.device), best_index]
        pooled_context = masked_mean(context, set_mask, dim=1)
        summary = torch.cat([
            top[:, :1], gap[:, None], match_mean[:, None], match_std[:, None],
            rank_entropy[:, None], evidence_lme[:, None], absolute_mean[:, None],
            absolute_std[:, None], best_pim, pooled_context,
        ], dim=-1)
        unknown_logit = self.unknown(summary).squeeze(-1)

        # Normalizing candidate logits by set size keeps the open-set decision
        # comparable across L=4/L=8/L=20 candidate protocols.
        open_candidate_logits = rank_logits - valid_count.float().log()[:, None]
        open_set_logits = torch.cat([open_candidate_logits, unknown_logit[:, None]], dim=1)
        joint_probability = open_set_logits.softmax(dim=1)
        exist_probability = joint_probability[:, :-1].sum(dim=1)
        exist_logit = torch.logsumexp(open_candidate_logits, dim=1) - unknown_logit
        return {
            "rank_logits": rank_logits, "rank_probability": rank_probability,
            "match_logits": match_logits, "unknown_logit": unknown_logit,
            "open_set_logits": open_set_logits, "joint_probability": joint_probability,
            "exist_logit": exist_logit, "exist_probability": exist_probability,
            "unknown_probability": joint_probability[:, -1], "set_entropy": rank_entropy,
            "evidence_lse": evidence_lme, "metric_scale": metric_scale,
        }


class SubtypeAwareOpenSetOCSR(nn.Module):
    """Factor source retrieval from two semantically different Unknown causes."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        evidence_dim = dim * 2 + 8 + 3
        self.relation_match = nn.Sequential(
            nn.Linear(evidence_dim, dim), nn.GELU(), nn.LayerNorm(dim),
            nn.Dropout(dropout), nn.Linear(dim, 1),
        )
        self.local_match = nn.Sequential(
            nn.LayerNorm(11), nn.Linear(11, dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim // 2, 1),
        )
        self.rank_residual = nn.Sequential(
            nn.Linear(evidence_dim, dim), nn.GELU(), nn.LayerNorm(dim),
            nn.Dropout(dropout), nn.Linear(dim, 1),
        )
        self.metric_scale_raw = nn.Parameter(torch.tensor([1.5, 0.75, 0.75]))
        self.component_weight_raw = nn.Parameter(torch.tensor([1.0, 0.0, 0.5]))
        self.type_anchor_scale_raw = nn.Parameter(torch.ones(3))
        self.type_residual = nn.Sequential(
            nn.Linear(dim + 16, dim), nn.GELU(), nn.LayerNorm(dim),
            nn.Dropout(dropout), nn.Linear(dim, 3),
        )

    def forward(self, pair: Tensor, pim: Tensor, set_mask: Tensor,
                absolute_similarity: Tensor) -> dict[str, Tensor]:
        context = self.context(pair, src_key_padding_mask=~set_mask)
        evidence = torch.cat([pair, context, pim, absolute_similarity], dim=-1)
        local_evidence = torch.cat([pim, absolute_similarity], dim=-1)
        metric_scale = F.softplus(self.metric_scale_raw)
        metric_match = (absolute_similarity * metric_scale).sum(dim=-1)
        components = torch.stack([
            self.relation_match(evidence).squeeze(-1),
            self.local_match(local_evidence).squeeze(-1),
            metric_match,
        ], dim=-1)
        component_weight = self.component_weight_raw.softmax(dim=0)
        match_logits = (components * component_weight).sum(dim=-1).masked_fill(~set_mask, -1e4)
        rank_logits = (
            match_logits + self.rank_residual(evidence).squeeze(-1)
        ).masked_fill(~set_mask, -1e4)

        valid_count = set_mask.sum(1).clamp_min(1)
        top = match_logits.topk(k=min(2, match_logits.shape[1]), dim=1).values
        gap = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else top[:, 0])
        match_mean = match_logits.masked_fill(~set_mask, 0).sum(1) / valid_count
        match_std = (
            (match_logits - match_mean[:, None]).square().masked_fill(~set_mask, 0).sum(1) / valid_count
        ).sqrt()
        rank_probability = rank_logits.softmax(dim=1)
        rank_entropy = -(rank_probability * rank_probability.clamp_min(1e-8).log()).sum(dim=1)
        evidence_lme = torch.logsumexp(match_logits, dim=1) - valid_count.float().log()
        absolute_mean = absolute_similarity[..., 0].masked_fill(~set_mask, 0).sum(1) / valid_count
        absolute_std = (
            (absolute_similarity[..., 0] - absolute_mean[:, None]).square().masked_fill(~set_mask, 0).sum(1)
            / valid_count
        ).sqrt()
        best_index = match_logits.argmax(dim=1)
        best_pim = pim[torch.arange(len(pim), device=pim.device), best_index]
        pooled_context = masked_mean(context, set_mask, dim=1)
        summary = torch.cat([
            top[:, :1], gap[:, None], match_mean[:, None], match_std[:, None],
            rank_entropy[:, None], evidence_lme[:, None], absolute_mean[:, None],
            absolute_std[:, None], best_pim, pooled_context,
        ], dim=-1)

        # Present is anchored by strong candidate evidence. Source-absent is
        # anchored by weak best-match evidence, while RF has an independent
        # residual path because it is a different Unknown mechanism.
        type_anchor = torch.stack([
            top[:, 0],
            -top[:, 0] + 0.5 * rank_entropy,
            -evidence_lme,
        ], dim=-1)
        type_logits = self.type_residual(summary) + type_anchor * F.softplus(self.type_anchor_scale_raw)
        type_probability = type_logits.softmax(dim=-1)
        unknown_logit = torch.logsumexp(type_logits[:, 1:], dim=-1)
        exist_logit = type_logits[:, 0] - unknown_logit

        # This factorization gives candidate probabilities P(present)*P(index|present)
        # and keeps the two Unknown subtypes identifiable during training.
        candidate_joint_logits = rank_logits.log_softmax(dim=1) + type_logits[:, :1]
        open_set_logits = torch.cat([candidate_joint_logits, unknown_logit[:, None]], dim=1)
        joint_probability = open_set_logits.softmax(dim=1)
        return {
            "rank_logits": rank_logits,
            "rank_probability": rank_probability,
            "match_logits": match_logits,
            "unknown_logit": unknown_logit,
            "open_set_logits": open_set_logits,
            "joint_probability": joint_probability,
            "exist_logit": exist_logit,
            "exist_probability": type_probability[:, 0],
            "unknown_probability": type_probability[:, 1:].sum(dim=1),
            "type_logits": type_logits,
            "type_probability": type_probability,
            "set_entropy": rank_entropy,
            "evidence_lse": evidence_lme,
            "metric_scale": metric_scale,
            "component_weight": component_weight,
            "type_features": summary,
        }


class T2Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.global_adapter = TaskAdapter(dim) if config.use_tsa else nn.Identity()
        self.local_adapter = (
            SourceLocalAdapter(dim, dropout=config.dropout)
            if config.variant in {
                "v6", "v7", "v8", "v9", "v10", "t2_abc_v1",
                "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
            } and config.use_tsa else
            TaskAdapter(dim) if config.use_tsa else nn.Identity()
        )
        self.source_gate = nn.Sequential(nn.Linear(dim * 3, 2), nn.Softmax(-1))
        self.seq_projection = nn.Linear(dim, dim)
        self.image_projection = nn.Linear(dim, dim)
        self.pim = PartialInheritanceMatcher(dim, config.sinkhorn_iters, config.sinkhorn_epsilon)
        self.cross_attention = nn.MultiheadAttention(dim, config.conformer_heads, dropout=config.dropout, batch_first=True)
        self.pair = nn.Sequential(nn.Linear(dim * 6 + 8, dim * 2), nn.GELU(), nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        if config.variant in {
            "v5", "v5r1", "v6", "v7", "v8", "v9", "v10", "t2_abc_v1",
            "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
        }:
            self.ocsr = SubtypeAwareOpenSetOCSR(dim, config.conformer_heads, config.dropout)
        elif config.variant == "v4":
            self.ocsr = OpenSetEvidenceOCSR(dim, config.conformer_heads, config.dropout)
        elif config.variant == "v3":
            self.ocsr = HierarchicalOCSR(dim, config.conformer_heads, config.dropout)
        else:
            self.ocsr = OCSR(dim, config.conformer_heads, config.dropout)
        self.use_pim, self.use_ocsr = config.use_pim, config.use_ocsr
        self.fusion_mode = config.fusion
        self.simple_rank = nn.Linear(dim, 1)
        self.simple_exist = nn.Sequential(nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, 1))
        if config.variant == "v6":
            self.rf_gate = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(dim // 2, 1),
            )
            self.in_set_gate = nn.Sequential(
                nn.LayerNorm(dim + 16), nn.Linear(dim + 16, dim // 2), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(dim // 2, 1),
            )
        self.release_alpha = 1.0
        self.release_beta = 1.0

    def set_release(self, alpha: float, beta: float) -> None:
        self.release_alpha = float(alpha)
        self.release_beta = float(beta)

    def _adapt(self, encoding: SignatureEncoding) -> tuple[Tensor, Tensor]:
        shared = self.global_adapter(encoding.global_shared)
        local = (
            self.local_adapter(encoding.local_shared, encoding.valid_mask)
            if self.config.variant in {
                "v6", "v7", "v8", "v9", "v10", "t2_abc_v1",
                "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
            } else self.local_adapter(encoding.local_shared)
        )
        if self.fusion_mode in {"sequence_only", "image_only"}:
            return shared, local
        weights = self.source_gate(torch.cat([encoding.global_sequence, encoding.global_image, shared], -1))
        global_token = shared + weights[..., 0:1] * self.seq_projection(encoding.global_sequence) + weights[..., 1:2] * self.image_projection(encoding.global_image)
        return global_token, local

    def forward(self, candidates: SignatureEncoding, query: SignatureEncoding, set_mask: Tensor,
                target_index: Tensor | None = None,
                episode_type_index: Tensor | None = None) -> dict[str, Tensor]:
        candidate_z, candidate_h = self._adapt(candidates)
        query_z, query_h = self._adapt(query)
        expanded_query_h = query_h[:, None].expand(-1, candidate_h.shape[1], -1, -1)
        expanded_query_mask = query.valid_mask[:, None].expand(-1, candidate_h.shape[1], -1)
        if self.use_pim:
            pim, diagnostics = self.pim(candidate_h, query_h, candidates.valid_mask, query.valid_mask,
                                        candidates.spatial, query.spatial)
        else:
            similarity = torch.einsum("blkd,bqd->blkq", F.normalize(candidate_h, dim=-1), F.normalize(query_h, dim=-1))
            pim = torch.zeros(*candidate_h.shape[:2], 8, device=candidate_h.device)
            pim[..., 0] = similarity.amax(dim=(-2, -1))
            diagnostics = {}
        flat_query = expanded_query_h.reshape(-1, query_h.shape[1], query_h.shape[2])
        flat_candidate = candidate_h.reshape(-1, candidate_h.shape[2], candidate_h.shape[3])
        flat_mask = candidates.valid_mask.reshape(-1, candidates.valid_mask.shape[-1])
        cross, _ = self.cross_attention(flat_query, flat_candidate, flat_candidate, key_padding_mask=~flat_mask)
        cross = masked_mean(cross, expanded_query_mask.reshape(-1, query_h.shape[1]), dim=1)
        cross = cross.reshape(*candidate_h.shape[:2], -1)
        qz = query_z[:, None].expand_as(candidate_z)
        global_relation = torch.cat([candidate_z, qz, qz - candidate_z,
                                     (qz - candidate_z).abs(), qz * candidate_z], -1)
        pair = self.pair(torch.cat([global_relation, cross, pim], -1))
        if self.use_ocsr:
            if self.config.variant in {
                "v4", "v5", "v5r1", "v6", "v7", "v8", "v9", "v10",
                "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
                "unified_abc_v14",
            }:
                absolute_similarity = torch.stack([
                    F.cosine_similarity(candidate_z, qz, dim=-1),
                    F.cosine_similarity(
                        candidates.global_sequence,
                        query.global_sequence[:, None].expand_as(candidates.global_sequence),
                        dim=-1,
                    ),
                    F.cosine_similarity(
                        candidates.global_image,
                        query.global_image[:, None].expand_as(candidates.global_image),
                        dim=-1,
                    ),
                ], dim=-1)
                output = self.ocsr(pair, pim, set_mask, absolute_similarity)
                output["absolute_similarity"] = absolute_similarity
            else:
                output = self.ocsr(pair, pim, set_mask)
        else:
            rank_logits = self.simple_rank(pair).squeeze(-1).masked_fill(~set_mask, -1e4)
            top = rank_logits.max(1).values
            pooled = masked_mean(pair, set_mask, dim=1)
            exist_logit = self.simple_exist(torch.cat([pooled, top[:, None], rank_logits.std(1, keepdim=True)], -1)).squeeze(-1)
            exist = torch.sigmoid(exist_logit)
            rank_probability = rank_logits.softmax(1)
            output = {"rank_logits": rank_logits, "rank_probability": rank_probability,
                      "exist_logit": exist_logit, "exist_probability": exist,
                      "unknown_probability": 1 - exist,
                      "joint_probability": torch.cat([exist[:, None] * rank_probability, (1 - exist)[:, None]], 1)}
        if self.config.variant == "v6":
            rf_logit = self.rf_gate(query_z).squeeze(-1)
            in_set_logit = self.in_set_gate(output["type_features"]).squeeze(-1)
            rf_probability = torch.sigmoid(rf_logit)
            in_set_probability = torch.sigmoid(in_set_logit)
            sf_probability = 1 - rf_probability
            exist_probability = sf_probability * in_set_probability
            rank_probability = output["rank_logits"].softmax(dim=1)
            predicted_joint = torch.cat([
                exist_probability[:, None] * rank_probability,
                (1 - exist_probability)[:, None],
            ], dim=1)
            type_probability = torch.stack([
                exist_probability,
                sf_probability * (1 - in_set_probability),
                rf_probability,
            ], dim=1)
            training_joint = predicted_joint
            if self.training and target_index is not None and episode_type_index is not None:
                oracle_sf = (episode_type_index != 2).to(rf_probability.dtype)
                oracle_in_set = (target_index >= 0).to(in_set_probability.dtype)
                released_sf = (1 - self.release_alpha) * oracle_sf + self.release_alpha * sf_probability
                released_in_set = (
                    (1 - self.release_beta) * oracle_in_set + self.release_beta * in_set_probability
                )
                released_exist = released_sf * released_in_set
                training_joint = torch.cat([
                    released_exist[:, None] * rank_probability,
                    (1 - released_exist)[:, None],
                ], dim=1)
            output.update({
                "rf_logit": rf_logit,
                "rf_probability": rf_probability,
                "in_set_logit": in_set_logit,
                "in_set_probability": in_set_probability,
                "exist_logit": torch.logit(exist_probability.clamp(1e-6, 1 - 1e-6)),
                "exist_probability": exist_probability,
                "unknown_probability": 1 - exist_probability,
                "joint_probability": predicted_joint,
                "training_joint_probability": training_joint,
                "open_set_logits": predicted_joint.clamp_min(1e-8).log(),
                "type_probability": type_probability,
                "type_logits": type_probability.clamp_min(1e-8).log(),
            })
        output.update({"pim_statistics": pim, "source_pair_embedding": pair, **diagnostics})
        return output


class ProgressiveBayesianT2Head(nn.Module):
    """Isolated ranking plus staged posterior updates for T2 open-set decisions."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.rank_branch = T2Head(config)
        self.open_global_adapter = TaskAdapter(dim) if config.use_tsa else nn.Identity()
        self.open_local_adapter = (
            SourceLocalAdapter(dim, dropout=config.dropout) if config.use_tsa else nn.Identity()
        )
        self.query_evidence = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.global_evidence = nn.Sequential(
            nn.LayerNorm(dim * 3 + 16), nn.Linear(dim * 3 + 16, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.local_evidence = nn.Sequential(
            nn.LayerNorm(dim * 4 + 12), nn.Linear(dim * 4 + 12, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.register_buffer(
            "log_type_prior", torch.tensor([0.5, 0.2, 0.3]).log(), persistent=True,
        )
        self.training_phase = "rank"

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"rank", "open"}:
            raise ValueError(f"Unsupported V7 training phase: {phase}")
        self.training_phase = phase
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("rank_branch.") == (phase == "rank"))
        if phase == "rank":
            self.rank_branch.train()
        else:
            self.rank_branch.eval()

    def enforce_training_mode(self) -> None:
        if self.training_phase == "open":
            self.rank_branch.eval()

    def _open_adapt(self, encoding: SignatureEncoding) -> tuple[Tensor, Tensor]:
        global_token = self.open_global_adapter(encoding.global_shared)
        local_token = (
            self.open_local_adapter(encoding.local_shared, encoding.valid_mask)
            if isinstance(self.open_local_adapter, SourceLocalAdapter)
            else self.open_local_adapter(encoding.local_shared)
        )
        return global_token, local_token

    @staticmethod
    def _local_statistics(candidate_h: Tensor, query_h: Tensor, candidate_mask: Tensor,
                          query_mask: Tensor) -> Tensor:
        similarity = torch.einsum(
            "blkd,bqd->blkq", F.normalize(candidate_h, dim=-1), F.normalize(query_h, dim=-1),
        )
        pair_mask = candidate_mask.unsqueeze(-1) & query_mask[:, None, None, :]
        masked = similarity.masked_fill(~pair_mask, -1e4)
        candidate_best = masked.max(dim=-1).values.masked_fill(~candidate_mask, 0)
        query_best = masked.max(dim=-2).values.masked_fill(~query_mask[:, None, :], 0)
        candidate_mean = candidate_best.sum(-1) / candidate_mask.sum(-1).clamp_min(1)
        query_mean = query_best.sum(-1) / query_mask.sum(-1).clamp_min(1)[:, None]
        candidate_coverage = ((candidate_best > 0.25) & candidate_mask).sum(-1) / candidate_mask.sum(-1).clamp_min(1)
        query_coverage = ((query_best > 0.25) & query_mask[:, None, :]).sum(-1) / query_mask.sum(-1).clamp_min(1)[:, None]
        return torch.stack([candidate_mean, query_mean, candidate_coverage, query_coverage], dim=-1)

    def forward(self, candidates: SignatureEncoding, query: SignatureEncoding, set_mask: Tensor,
                target_index: Tensor | None = None,
                episode_type_index: Tensor | None = None) -> dict[str, Tensor]:
        rank_output = self.rank_branch(candidates, query, set_mask, target_index, episode_type_index)
        candidate_z, candidate_h = self._open_adapt(candidates)
        query_z, query_h = self._open_adapt(query)
        pooled_candidate = masked_mean(candidate_z, set_mask, dim=1)

        query_feature = torch.cat([
            query_z, query.global_sequence, query.global_image,
        ], dim=-1)
        query_log_bayes_factor = self.query_evidence(query_feature).squeeze(-1)

        type_features = rank_output["type_features"].detach()
        global_feature = torch.cat([type_features, query_z, pooled_candidate], dim=-1)
        global_log_bayes_factor = self.global_evidence(global_feature).squeeze(-1)

        best_index = rank_output["rank_logits"].detach().argmax(dim=1)
        batch_index = torch.arange(len(best_index), device=best_index.device)
        best_pair = rank_output["source_pair_embedding"].detach()[batch_index, best_index]
        best_pim = rank_output["pim_statistics"].detach()[batch_index, best_index]
        best_candidate = candidate_z[batch_index, best_index]
        local_statistics = self._local_statistics(
            candidate_h, query_h, candidates.valid_mask, query.valid_mask,
        )[batch_index, best_index]
        local_feature = torch.cat([
            best_pair, best_pim, local_statistics, query_z, best_candidate,
            (query_z - best_candidate).abs(),
        ], dim=-1)
        local_log_bayes_factor = self.local_evidence(local_feature).squeeze(-1)

        zeros = query_log_bayes_factor.new_zeros(query_log_bayes_factor.shape)
        query_increment = torch.stack([
            -0.5 * query_log_bayes_factor, -0.5 * query_log_bayes_factor,
            query_log_bayes_factor,
        ], dim=-1)
        global_increment = torch.stack([
            global_log_bayes_factor, -global_log_bayes_factor, zeros,
        ], dim=-1)
        local_increment = torch.stack([
            local_log_bayes_factor, -local_log_bayes_factor, zeros,
        ], dim=-1)
        prior = self.log_type_prior.expand(len(query_log_bayes_factor), -1)
        stage_logits = torch.stack([
            prior,
            prior + query_increment,
            prior + query_increment + global_increment,
            prior + query_increment + global_increment + local_increment,
        ], dim=1)
        stage_probability = stage_logits.softmax(dim=-1)
        type_logits = stage_logits[:, -1]
        type_probability = stage_probability[:, -1]
        rank_probability = rank_output["rank_logits"].softmax(dim=1)
        exist_probability = type_probability[:, 0]
        unknown_probability = type_probability[:, 1:].sum(dim=1)
        joint_probability = torch.cat([
            exist_probability[:, None] * rank_probability,
            unknown_probability[:, None],
        ], dim=1)
        output = dict(rank_output)
        output.update({
            "query_log_bayes_factor": query_log_bayes_factor,
            "global_log_bayes_factor": global_log_bayes_factor,
            "local_log_bayes_factor": local_log_bayes_factor,
            "bayesian_evidence_increment": torch.stack([
                torch.zeros_like(query_increment), query_increment, global_increment, local_increment,
            ], dim=1),
            "bayesian_stage_logits": stage_logits,
            "bayesian_stage_probability": stage_probability,
            "type_logits": type_logits,
            "type_probability": type_probability,
            "exist_logit": type_logits[:, 0] - torch.logsumexp(type_logits[:, 1:], dim=1),
            "exist_probability": exist_probability,
            "unknown_probability": unknown_probability,
            "joint_probability": joint_probability,
            "open_set_logits": joint_probability.clamp_min(1e-8).log(),
            "rf_logit": type_logits[:, 2] - torch.logsumexp(type_logits[:, :2], dim=1),
            "in_set_logit": type_logits[:, 0] - type_logits[:, 1],
        })
        return output


class StatefulBayesianT2Head(ProgressiveBayesianT2Head):
    """V8 head with an append-only evidence state and prefix-supervised posteriors."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        dim = config.hidden_dim
        packet_dim = dim + 11
        self.incremental_rank = nn.Sequential(
            nn.LayerNorm(packet_dim), nn.Linear(packet_dim, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.state_packet = nn.Sequential(
            nn.LayerNorm(packet_dim), nn.Linear(packet_dim, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, dim),
        )
        self.state_global_evidence = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.state_local_evidence = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"rank", "open"}:
            raise ValueError(f"Unsupported V8 training phase: {phase}")
        self.training_phase = phase
        rank_prefixes = ("rank_branch.", "incremental_rank.")
        open_prefixes = (
            "open_global_adapter.", "open_local_adapter.", "query_evidence.",
            "state_packet.", "state_global_evidence.", "state_local_evidence.",
        )
        for name, parameter in self.named_parameters():
            if phase == "rank":
                parameter.requires_grad_(name.startswith(rank_prefixes))
            else:
                parameter.requires_grad_(name.startswith(open_prefixes))
        if phase == "rank":
            self.rank_branch.train()
            self.incremental_rank.train()
        else:
            self.rank_branch.eval()
            self.incremental_rank.eval()

    def enforce_training_mode(self) -> None:
        if self.training_phase == "open":
            self.rank_branch.eval()
            self.incremental_rank.eval()

    @staticmethod
    def _prefix_masks(set_mask: Tensor, sizes: tuple[int, ...] | list[int]) -> tuple[Tensor, Tensor]:
        width = set_mask.shape[1]
        normalized = sorted({max(1, min(int(size), width)) for size in sizes} | {width})
        positions = torch.arange(width, device=set_mask.device)
        masks = torch.stack([set_mask & (positions[None] < size) for size in normalized], dim=1)
        return masks, torch.tensor(normalized, device=set_mask.device, dtype=torch.long)

    def _stateful_posteriors(self, packet: Tensor, query_z: Tensor, rank_logits: Tensor,
                             set_mask: Tensor, query_increment: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        prefix_masks, prefix_sizes = self._prefix_masks(set_mask, self.config.stateful_prefix_sizes)
        hidden = self.state_packet(packet.detach())
        expanded = hidden[:, None].expand(-1, len(prefix_sizes), -1, -1)
        weights = prefix_masks.to(hidden.dtype).unsqueeze(-1)
        mean = (expanded * weights).sum(2) / weights.sum(2).clamp_min(1)
        maximum = expanded.masked_fill(~prefix_masks.unsqueeze(-1), -1e4).max(2).values
        query_expanded = query_z[:, None].expand(-1, len(prefix_sizes), -1)
        global_factor = self.state_global_evidence(
            torch.cat([query_expanded, mean, maximum], dim=-1),
        ).squeeze(-1)

        prefix_rank = rank_logits[:, None].expand(-1, len(prefix_sizes), -1).masked_fill(~prefix_masks, -1e4)
        best_index = prefix_rank.argmax(dim=-1)
        best_hidden = hidden[:, None].expand(-1, len(prefix_sizes), -1, -1).gather(
            2, best_index[..., None, None].expand(-1, -1, 1, hidden.shape[-1]),
        ).squeeze(2)
        local_factor = self.state_local_evidence(torch.cat([
            query_expanded, best_hidden, (query_expanded - best_hidden).abs(),
        ], dim=-1)).squeeze(-1)

        zeros = global_factor.new_zeros(global_factor.shape)
        global_increment = torch.stack([global_factor, -global_factor, zeros], dim=-1)
        local_increment = torch.stack([local_factor, -local_factor, zeros], dim=-1)
        prior = self.log_type_prior.expand(len(query_z), -1)
        prefix_logits = (
            prior[:, None] + query_increment[:, None] + global_increment + local_increment
        )
        return prefix_logits, prefix_masks, prefix_sizes, global_increment, local_increment

    def forward(self, candidates: SignatureEncoding, query: SignatureEncoding, set_mask: Tensor,
                target_index: Tensor | None = None,
                episode_type_index: Tensor | None = None) -> dict[str, Tensor]:
        output = super().forward(candidates, query, set_mask, target_index, episode_type_index)
        packet = torch.cat([
            output["source_pair_embedding"], output["pim_statistics"], output["absolute_similarity"],
        ], dim=-1)
        rank_input = packet if self.training_phase == "rank" else packet.detach()
        rank_logits = self.incremental_rank(rank_input).squeeze(-1).masked_fill(~set_mask, -1e4)
        query_z, _ = self._open_adapt(query)
        query_factor = output["query_log_bayes_factor"]
        query_increment = torch.stack([
            -0.5 * query_factor, -0.5 * query_factor, query_factor,
        ], dim=-1)
        prefix_logits, prefix_masks, prefix_sizes, global_increment, local_increment = self._stateful_posteriors(
            packet, query_z, rank_logits, set_mask, query_increment,
        )
        prefix_probability = prefix_logits.softmax(dim=-1)
        type_logits = prefix_logits[:, -1]
        type_probability = prefix_probability[:, -1]
        rank_probability = rank_logits.softmax(dim=-1)
        exist_probability = type_probability[:, 0]
        unknown_probability = type_probability[:, 1:].sum(dim=1)
        joint_probability = torch.cat([
            exist_probability[:, None] * rank_probability, unknown_probability[:, None],
        ], dim=1)
        prior = self.log_type_prior.expand(len(query_z), -1)
        stage_logits = torch.stack([
            prior, prior + query_increment,
            prior + query_increment + global_increment[:, -1],
            prior + query_increment + global_increment[:, -1] + local_increment[:, -1],
        ], dim=1)
        output.update({
            "rank_logits": rank_logits,
            "rank_probability": rank_probability,
            "stateful_prefix_logits": prefix_logits,
            "stateful_prefix_probability": prefix_probability,
            "stateful_prefix_masks": prefix_masks,
            "stateful_prefix_sizes": prefix_sizes,
            "bayesian_stage_logits": stage_logits,
            "bayesian_stage_probability": stage_logits.softmax(dim=-1),
            "bayesian_evidence_increment": torch.stack([
                torch.zeros_like(query_increment), query_increment,
                global_increment[:, -1], local_increment[:, -1],
            ], dim=1),
            "type_logits": type_logits,
            "type_probability": type_probability,
            "exist_logit": type_logits[:, 0] - torch.logsumexp(type_logits[:, 1:], dim=1),
            "exist_probability": exist_probability,
            "unknown_probability": unknown_probability,
            "joint_probability": joint_probability,
            "open_set_logits": joint_probability.clamp_min(1e-8).log(),
            "in_set_logit": type_logits[:, 0] - type_logits[:, 1],
        })
        return output

    def begin_state(self, query: SignatureEncoding) -> BayesianEvidenceState:
        query_z, _ = self._open_adapt(query)
        query_feature = torch.cat([query_z, query.global_sequence, query.global_image], dim=-1)
        query_factor = self.query_evidence(query_feature).squeeze(-1)
        query_increment = torch.stack([
            -0.5 * query_factor, -0.5 * query_factor, query_factor,
        ], dim=-1)
        posterior = (self.log_type_prior.expand(len(query_z), -1) + query_increment).softmax(dim=-1).detach()
        return BayesianEvidenceState(query, None, None, posterior, (posterior,))

    def append_state(self, state: BayesianEvidenceState, candidates: SignatureEncoding,
                     set_mask: Tensor) -> tuple[BayesianEvidenceState, dict[str, Tensor]]:
        joined_candidates = candidates if state.candidates is None else _concat_set_encodings(
            state.candidates, candidates,
        )
        joined_mask = set_mask if state.set_mask is None else torch.cat([state.set_mask, set_mask], dim=1)
        output = self.forward(joined_candidates, state.query, joined_mask)
        posterior = output["type_probability"].detach()
        updated = BayesianEvidenceState(
            state.query, joined_candidates, joined_mask, posterior,
            (*state.posterior_history, posterior),
        )
        return updated, output


class UnifiedEvidenceT2Head(nn.Module):
    """V9 candidate/null competition with a query-only RF branch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        packet_dim = dim + 11
        self.query_global_adapter = TaskAdapter(dim)
        self.query_local_adapter = SourceLocalAdapter(dim, dropout=config.dropout)
        self.candidate_global_adapter = TaskAdapter(dim)
        self.candidate_local_adapter = SourceLocalAdapter(dim, dropout=config.dropout)
        self.relation = T2Head(config)
        self.candidate_score = nn.Sequential(
            nn.LayerNorm(packet_dim), nn.Linear(packet_dim, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.null_evidence = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.rf_gate = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim // 2, 1),
        )
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "log_type_prior", torch.tensor([0.5, 0.2, 0.3]).log(), persistent=True,
        )
        self.training_phase = "representation"

    @staticmethod
    def _with_adapted_shared(
        encoding: SignatureEncoding, global_shared: Tensor, local_shared: Tensor,
    ) -> SignatureEncoding:
        return SignatureEncoding(
            global_shared=global_shared, local_shared=local_shared,
            valid_mask=encoding.valid_mask, spatial=encoding.spatial,
            global_sequence=encoding.global_sequence, global_image=encoding.global_image,
            diagnostics=encoding.diagnostics,
        )

    def _adapt_roles(
        self, candidates: SignatureEncoding, query: SignatureEncoding,
    ) -> tuple[SignatureEncoding, SignatureEncoding]:
        candidate = self._with_adapted_shared(
            candidates,
            self.candidate_global_adapter(candidates.global_shared),
            self.candidate_local_adapter(candidates.local_shared, candidates.valid_mask),
        )
        query_adapted = self._with_adapted_shared(
            query,
            self.query_global_adapter(query.global_shared),
            self.query_local_adapter(query.local_shared, query.valid_mask),
        )
        return candidate, query_adapted

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"representation", "unified", "calibration"}:
            raise ValueError(f"Unsupported V9 training phase: {phase}")
        self.training_phase = phase
        calibration_prefixes = ("null_evidence.", "rf_gate.", "log_temperature")
        for name, parameter in self.named_parameters():
            if phase == "representation":
                parameter.requires_grad_(not name.startswith(calibration_prefixes))
            elif phase == "unified":
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(name.startswith(calibration_prefixes))

    def enforce_training_mode(self) -> None:
        if self.training_phase == "representation":
            self.null_evidence.eval()
            self.rf_gate.eval()
        elif self.training_phase == "calibration":
            for module in (
                self.query_global_adapter, self.query_local_adapter,
                self.candidate_global_adapter, self.candidate_local_adapter,
                self.relation, self.candidate_score,
            ):
                module.eval()

    def _query_features(self, query: SignatureEncoding) -> Tensor:
        query_z = self.query_global_adapter(query.global_shared)
        return torch.cat([query_z, query.global_sequence, query.global_image], dim=-1)

    def _prefix_outputs(
        self, candidate_logits: Tensor, null_logit: Tensor, rf_probability: Tensor,
        set_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        prefix_masks, prefix_sizes = StatefulBayesianT2Head._prefix_masks(
            set_mask, self.config.stateful_prefix_sizes,
        )
        expanded = candidate_logits[:, None].expand(-1, len(prefix_sizes), -1)
        masked = expanded.masked_fill(~prefix_masks, -1e4)
        prefix_logits = torch.cat([
            masked, null_logit[:, None, None].expand(-1, len(prefix_sizes), 1),
        ], dim=-1)
        conditional = prefix_logits.softmax(dim=-1)
        null_probability = conditional[..., -1]
        sf_probability = 1 - rf_probability[:, None]
        type_probability = torch.stack([
            sf_probability * (1 - null_probability),
            sf_probability * null_probability,
            rf_probability[:, None].expand_as(null_probability),
        ], dim=-1)
        return prefix_logits, type_probability, prefix_masks, prefix_sizes

    def forward(self, candidates: SignatureEncoding, query: SignatureEncoding, set_mask: Tensor,
                target_index: Tensor | None = None,
                episode_type_index: Tensor | None = None) -> dict[str, Tensor]:
        candidate, query_adapted = self._adapt_roles(candidates, query)
        relation = self.relation(candidate, query_adapted, set_mask, target_index, episode_type_index)
        packet = torch.cat([
            relation["source_pair_embedding"], relation["pim_statistics"],
            relation["absolute_similarity"],
        ], dim=-1)
        temperature = self.log_temperature.exp().clamp(0.5, 2.0)
        candidate_logits = (
            self.candidate_score(packet).squeeze(-1) / temperature
        ).masked_fill(~set_mask, -1e4)
        query_features = self._query_features(query)
        null_logit = self.null_evidence(query_features).squeeze(-1) / temperature
        rf_logit = self.rf_gate(query_features).squeeze(-1)
        rf_probability = torch.sigmoid(rf_logit)

        conditional_logits = torch.cat([candidate_logits, null_logit[:, None]], dim=-1)
        conditional_probability = conditional_logits.softmax(dim=-1)
        sf_probability = 1 - rf_probability
        candidate_probability = sf_probability[:, None] * conditional_probability[:, :-1]
        absent_probability = sf_probability * conditional_probability[:, -1]
        unknown_probability = absent_probability + rf_probability
        joint_probability = torch.cat([candidate_probability, unknown_probability[:, None]], dim=-1)
        exist_probability = candidate_probability.sum(dim=-1)
        type_probability = torch.stack([
            exist_probability, absent_probability, rf_probability,
        ], dim=-1)

        prefix_logits, prefix_type_probability, prefix_masks, prefix_sizes = self._prefix_outputs(
            candidate_logits, null_logit, rf_probability, set_mask,
        )
        query_type_probability = torch.stack([
            0.5 * sf_probability, 0.5 * sf_probability, rf_probability,
        ], dim=-1)
        prior = self.log_type_prior.expand(len(candidate_logits), -1)
        query_type_logits = query_type_probability.clamp_min(1e-8).log()
        final_type_logits = type_probability.clamp_min(1e-8).log()
        stage_logits = torch.stack([
            prior, query_type_logits, final_type_logits, final_type_logits,
        ], dim=1)
        increments = torch.stack([
            torch.zeros_like(prior), query_type_logits - prior,
            final_type_logits - query_type_logits, torch.zeros_like(prior),
        ], dim=1)

        return {
            **relation,
            "rank_logits": candidate_logits,
            "rank_probability": candidate_logits.softmax(dim=-1),
            "candidate_logits": candidate_logits,
            "match_logits": candidate_logits,
            "null_logit": null_logit,
            "rf_logit": rf_logit,
            "rf_probability": rf_probability,
            "conditional_logits": conditional_logits,
            "conditional_probability": conditional_probability,
            "joint_probability": joint_probability,
            "open_set_logits": joint_probability.clamp_min(1e-8).log(),
            "exist_logit": torch.logit(exist_probability.clamp(1e-6, 1 - 1e-6)),
            "exist_probability": exist_probability,
            "unknown_probability": unknown_probability,
            "in_set_logit": torch.logsumexp(candidate_logits, dim=-1) - null_logit,
            "type_probability": type_probability,
            "type_logits": final_type_logits,
            "prefix_conditional_logits": prefix_logits,
            "stateful_prefix_logits": prefix_type_probability.clamp_min(1e-8).log(),
            "stateful_prefix_probability": prefix_type_probability,
            "stateful_prefix_masks": prefix_masks,
            "stateful_prefix_sizes": prefix_sizes,
            "bayesian_stage_logits": stage_logits,
            "bayesian_stage_probability": stage_logits.softmax(dim=-1),
            "bayesian_evidence_increment": increments,
            "evidence_packet": packet,
            "temperature": temperature,
            "training_phase": self.training_phase,
        }

    def begin_state(self, query: SignatureEncoding) -> BayesianEvidenceState:
        query_features = self._query_features(query)
        rf_probability = torch.sigmoid(self.rf_gate(query_features).squeeze(-1))
        posterior = torch.stack([
            0.5 * (1 - rf_probability), 0.5 * (1 - rf_probability), rf_probability,
        ], dim=-1).detach()
        return BayesianEvidenceState(query, None, None, posterior, (posterior,))

    def append_state(self, state: BayesianEvidenceState, candidates: SignatureEncoding,
                     set_mask: Tensor) -> tuple[BayesianEvidenceState, dict[str, Tensor]]:
        joined_candidates = candidates if state.candidates is None else _concat_set_encodings(
            state.candidates, candidates,
        )
        joined_mask = set_mask if state.set_mask is None else torch.cat([state.set_mask, set_mask], dim=1)
        output = self.forward(joined_candidates, state.query, joined_mask)
        posterior = output["type_probability"].detach()
        updated = BayesianEvidenceState(
            state.query, joined_candidates, joined_mask, posterior,
            (*state.posterior_history, posterior),
        )
        return updated, output


class DualEvidenceT2Head(nn.Module):
    """V10: independent ranking and source-existence evidence networks."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        packet_dim = dim + 11
        self.rank_relation = T2Head(config)
        self.open_relation = T2Head(config)
        self.rank_score = nn.Sequential(
            nn.LayerNorm(packet_dim), nn.Linear(packet_dim, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.open_candidate_score = nn.Sequential(
            nn.LayerNorm(packet_dim), nn.Linear(packet_dim, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.open_null = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.rf_gate = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim // 2), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim // 2, 1),
        )
        self.rank_log_temperature = nn.Parameter(torch.zeros(()))
        self.open_log_temperature = nn.Parameter(torch.zeros(()))
        self.query_present_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "log_type_prior", torch.tensor([0.5, 0.2, 0.3]).log(), persistent=True,
        )
        self.training_phase = "rank"

    @staticmethod
    def _packet(output: dict[str, Tensor]) -> Tensor:
        return torch.cat([
            output["source_pair_embedding"], output["pim_statistics"],
            output["absolute_similarity"],
        ], dim=-1)

    @staticmethod
    def _query_features(encoding: SignatureEncoding) -> Tensor:
        return torch.cat([
            encoding.global_shared, encoding.global_sequence, encoding.global_image,
        ], dim=-1)

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"rank", "open", "calibration"}:
            raise ValueError(f"Unsupported V10 training phase: {phase}")
        self.training_phase = phase
        for name, parameter in self.named_parameters():
            if phase == "rank":
                trainable = name.startswith(("rank_relation.", "rank_score.", "rank_log_temperature"))
            elif phase == "open":
                trainable = name.startswith((
                    "open_relation.", "open_candidate_score.", "open_null.",
                    "rf_gate.", "open_log_temperature", "query_present_logit",
                ))
            else:
                trainable = name in {"rank_log_temperature", "open_log_temperature", "query_present_logit"}
            parameter.requires_grad_(trainable)

    def enforce_training_mode(self) -> None:
        rank_modules = (self.rank_relation, self.rank_score)
        open_modules = (
            self.open_relation, self.open_candidate_score, self.open_null, self.rf_gate,
        )
        if self.training_phase != "rank":
            for module in rank_modules:
                module.eval()
        if self.training_phase != "open":
            for module in open_modules:
                module.eval()

    def _prefix_outputs(
        self, presence_logits: Tensor, null_logit: Tensor, rf_probability: Tensor,
        set_mask: Tensor, open_temperature: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prefix_masks, prefix_sizes = StatefulBayesianT2Head._prefix_masks(
            set_mask, self.config.stateful_prefix_sizes,
        )
        expanded = presence_logits[:, None].expand(-1, len(prefix_sizes), -1)
        masked = expanded.masked_fill(~prefix_masks, -1e4)
        valid_count = prefix_masks.sum(dim=-1).clamp_min(1).to(masked.dtype)
        evidence = torch.logsumexp(masked, dim=-1) - valid_count.log()
        in_set_logit = (evidence - null_logit[:, None]) / open_temperature
        in_set_probability = torch.sigmoid(in_set_logit)
        sf_probability = 1 - rf_probability[:, None]
        prefix_type_probability = torch.stack([
            sf_probability * in_set_probability,
            sf_probability * (1 - in_set_probability),
            rf_probability[:, None].expand_as(in_set_probability),
        ], dim=-1)
        return prefix_type_probability, prefix_masks, prefix_sizes

    def forward(
        self, rank_candidates: SignatureEncoding, rank_query: SignatureEncoding,
        open_candidates: SignatureEncoding, open_query: SignatureEncoding, set_mask: Tensor,
        target_index: Tensor | None = None, episode_type_index: Tensor | None = None,
    ) -> dict[str, Tensor]:
        rank_relation = self.rank_relation(
            rank_candidates, rank_query, set_mask, target_index, episode_type_index,
        )
        rank_packet = self._packet(rank_relation)
        rank_temperature = self.rank_log_temperature.exp().clamp(0.5, 2.0)
        rank_logits = (
            self.rank_score(rank_packet).squeeze(-1) / rank_temperature
        ).masked_fill(~set_mask, -1e4)

        open_relation = self.open_relation(
            open_candidates, open_query, set_mask, target_index, episode_type_index,
        )
        open_packet = self._packet(open_relation)
        presence_logits = self.open_candidate_score(open_packet).squeeze(-1).masked_fill(~set_mask, -1e4)
        query_features = self._query_features(open_query)
        null_logit = self.open_null(query_features).squeeze(-1)
        rf_logit = self.rf_gate(query_features).squeeze(-1)
        rf_probability = torch.sigmoid(rf_logit)
        valid_count = set_mask.sum(dim=-1).clamp_min(1).to(presence_logits.dtype)
        set_evidence = torch.logsumexp(presence_logits, dim=-1) - valid_count.log()
        open_temperature = self.open_log_temperature.exp().clamp(0.5, 2.0)
        in_set_logit = (set_evidence - null_logit) / open_temperature
        in_set_probability = torch.sigmoid(in_set_logit)
        sf_probability = 1 - rf_probability
        type_probability = torch.stack([
            sf_probability * in_set_probability,
            sf_probability * (1 - in_set_probability),
            rf_probability,
        ], dim=-1)
        type_logits = type_probability.clamp_min(1e-8).log()

        rank_probability = rank_logits.softmax(dim=-1)
        candidate_probability = type_probability[:, :1] * rank_probability
        unknown_probability = type_probability[:, 1:].sum(dim=-1)
        joint_probability = torch.cat([candidate_probability, unknown_probability[:, None]], dim=-1)
        prefix_type_probability, prefix_masks, prefix_sizes = self._prefix_outputs(
            presence_logits, null_logit, rf_probability, set_mask, open_temperature,
        )

        query_present = torch.sigmoid(self.query_present_logit)
        query_type_probability = torch.stack([
            sf_probability * query_present,
            sf_probability * (1 - query_present),
            rf_probability,
        ], dim=-1)
        prior = self.log_type_prior.expand(len(rank_logits), -1)
        query_type_logits = query_type_probability.clamp_min(1e-8).log()
        stage_logits = torch.stack([prior, query_type_logits, type_logits, type_logits], dim=1)
        evidence_increment = torch.stack([
            torch.zeros_like(prior), query_type_logits - prior,
            type_logits - query_type_logits, torch.zeros_like(prior),
        ], dim=1)

        return {
            "rank_logits": rank_logits,
            "rank_probability": rank_probability,
            "candidate_logits": rank_logits,
            "pim_statistics": rank_relation["pim_statistics"],
            "open_presence_logits": presence_logits,
            "open_in_set_logit": in_set_logit,
            "in_set_logit": in_set_logit,
            "rf_logit": rf_logit,
            "rf_probability": rf_probability,
            "joint_probability": joint_probability,
            "open_set_logits": joint_probability.clamp_min(1e-8).log(),
            "exist_logit": torch.logit(type_probability[:, 0].clamp(1e-6, 1 - 1e-6)),
            "exist_probability": type_probability[:, 0],
            "unknown_probability": unknown_probability,
            "type_probability": type_probability,
            "type_logits": type_logits,
            "stateful_prefix_logits": prefix_type_probability.clamp_min(1e-8).log(),
            "stateful_prefix_probability": prefix_type_probability,
            "stateful_prefix_masks": prefix_masks,
            "stateful_prefix_sizes": prefix_sizes,
            "bayesian_stage_logits": stage_logits,
            "bayesian_stage_probability": stage_logits.softmax(dim=-1),
            "bayesian_evidence_increment": evidence_increment,
            "rank_evidence_packet": rank_packet,
            "open_evidence_packet": open_packet,
            "rank_temperature": rank_temperature,
            "open_temperature": open_temperature,
            "training_phase": self.training_phase,
        }

    def begin_state(
        self, query: DualSignatureEncoding,
    ) -> DualBayesianEvidenceState:
        query_features = self._query_features(query.open)
        rf_probability = torch.sigmoid(self.rf_gate(query_features).squeeze(-1))
        query_present = torch.sigmoid(self.query_present_logit)
        posterior = torch.stack([
            (1 - rf_probability) * query_present,
            (1 - rf_probability) * (1 - query_present),
            rf_probability,
        ], dim=-1).detach()
        return DualBayesianEvidenceState(query, None, None, posterior, (posterior,))

    def append_state(
        self, state: DualBayesianEvidenceState, candidates: DualSignatureEncoding,
        set_mask: Tensor,
    ) -> tuple[DualBayesianEvidenceState, dict[str, Tensor]]:
        if state.candidates is None:
            joined = candidates
            joined_mask = set_mask
        else:
            joined = DualSignatureEncoding(
                _concat_set_encodings(state.candidates.rank, candidates.rank),
                _concat_set_encodings(state.candidates.open, candidates.open),
            )
            assert state.set_mask is not None
            joined_mask = torch.cat([state.set_mask, set_mask], dim=1)
        output = self.forward(
            joined.rank, state.query.rank, joined.open, state.query.open, joined_mask,
        )
        posterior = output["type_probability"].detach()
        updated = DualBayesianEvidenceState(
            state.query, joined, joined_mask, posterior, (*state.posterior_history, posterior),
        )
        return updated, output


class ConditionalABCT2Head(nn.Module):
    """Single-encoder T2 baseline with explicit RF, in-set, and rank factors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_dim
        self.relation = T2Head(config)
        self.rf_head = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.in_set_head = nn.Sequential(
            nn.LayerNorm(dim * 2 + 5), nn.Linear(dim * 2 + 5, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.null_source_head = (
            nn.Sequential(
                nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(dim, 1),
            )
            if config.t2_null_source_enabled else None
        )
        self.null_source_scale = (
            nn.Parameter(torch.zeros(())) if config.t2_null_source_enabled else None
        )

    @staticmethod
    def _query_features(query: SignatureEncoding) -> Tensor:
        return torch.cat([
            query.global_shared, query.global_sequence, query.global_image,
        ], dim=-1)

    def forward_a(self, query: SignatureEncoding) -> dict[str, Tensor]:
        rf_logit = self.rf_head(self._query_features(query)).squeeze(-1)
        return {
            "rf_logit": rf_logit,
            "rf_probability": torch.sigmoid(rf_logit),
        }

    def forward(
        self, candidates: SignatureEncoding, query: SignatureEncoding, set_mask: Tensor,
        target_index: Tensor | None = None, episode_type_index: Tensor | None = None,
    ) -> dict[str, Tensor]:
        relation = self.relation(
            candidates, query, set_mask, target_index, episode_type_index,
        )
        pair = relation["source_pair_embedding"]
        rank_logits = relation["rank_logits"].masked_fill(~set_mask, -1e4)
        rank_probability = rank_logits.softmax(dim=-1)

        mask = set_mask.unsqueeze(-1)
        valid_count = set_mask.sum(dim=-1).clamp_min(1).to(pair.dtype)
        pair_mean = (pair * mask).sum(dim=1) / valid_count[:, None]
        pair_max = pair.masked_fill(~mask, -1e4).max(dim=1).values
        valid_logits = rank_logits.masked_fill(~set_mask, 0)
        score_mean = valid_logits.sum(dim=1) / valid_count
        score_variance = (
            ((rank_logits - score_mean[:, None]).square() * set_mask).sum(dim=1)
            / valid_count
        )
        score_std = score_variance.clamp_min(1e-12).sqrt()
        top = rank_logits.topk(k=min(2, rank_logits.shape[1]), dim=1).values
        gap = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else top[:, 0])
        entropy = -(
            rank_probability * rank_probability.clamp_min(1e-8).log()
        ).sum(dim=-1)
        set_features = torch.cat([
            pair_mean, pair_max, top[:, :1], gap[:, None], score_mean[:, None],
            score_std[:, None], entropy[:, None],
        ], dim=-1)

        rf_output = self.forward_a(query)
        rf_logit = rf_output["rf_logit"]
        in_set_logit = self.in_set_head(set_features).squeeze(-1)
        null_source_evidence = None
        candidate_energy = None
        if self.null_source_head is not None:
            assert self.null_source_scale is not None
            candidate_energy = torch.logsumexp(rank_logits, dim=-1) - valid_count.log()
            null_source_evidence = self.null_source_head(torch.cat([
                self._query_features(query), pair_mean, pair_max,
            ], dim=-1)).squeeze(-1)
            in_set_logit = in_set_logit + torch.tanh(self.null_source_scale) * (
                candidate_energy - null_source_evidence
            )
        rf_probability = torch.sigmoid(rf_logit)
        in_set_probability = torch.sigmoid(in_set_logit)
        sf_probability = 1 - rf_probability
        present_probability = sf_probability * in_set_probability
        absent_probability = sf_probability * (1 - in_set_probability)
        type_probability = torch.stack([
            present_probability, absent_probability, rf_probability,
        ], dim=-1)
        candidate_probability = present_probability[:, None] * rank_probability
        unknown_probability = absent_probability + rf_probability
        joint_probability = torch.cat([
            candidate_probability, unknown_probability[:, None],
        ], dim=-1)
        official_probability = torch.cat([
            candidate_probability, absent_probability[:, None], rf_probability[:, None],
        ], dim=-1)

        relation.update({
            "rank_logits": rank_logits,
            "rank_probability": rank_probability,
            "rf_logit": rf_logit,
            "rf_probability": rf_probability,
            "in_set_logit": in_set_logit,
            "in_set_probability": in_set_probability,
            "exist_logit": torch.logit(present_probability.clamp(1e-6, 1 - 1e-6)),
            "exist_probability": present_probability,
            "candidate_probability": candidate_probability,
            "absent_probability": absent_probability,
            "unknown_probability": unknown_probability,
            "type_probability": type_probability,
            "type_logits": type_probability.clamp_min(1e-8).log(),
            "joint_probability": joint_probability,
            "official_probability": official_probability,
            "open_set_logits": joint_probability.clamp_min(1e-8).log(),
        })
        if null_source_evidence is not None:
            relation.update({
                "candidate_energy": candidate_energy,
                "null_source_evidence": null_source_evidence,
                "null_source_scale": torch.tanh(self.null_source_scale),
            })
        return relation

    def source_existence_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.in_set_head.parameters())
        if self.null_source_head is not None:
            parameters.extend(self.null_source_head.parameters())
            assert self.null_source_scale is not None
            parameters.append(self.null_source_scale)
        return parameters


class T2RelationEnergyAdapter(nn.Module):
    """T2-only residual that compares candidate evidence with a learned null source."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.null_evidence = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.raw_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _query_features(query: SignatureEncoding) -> Tensor:
        return torch.cat([
            query.global_shared, query.global_sequence, query.global_image,
        ], dim=-1)

    def forward(
        self, output: dict[str, Tensor], query: SignatureEncoding, set_mask: Tensor,
    ) -> dict[str, Tensor]:
        pair = output["source_pair_embedding"]
        valid = set_mask.unsqueeze(-1)
        count = set_mask.sum(dim=-1).clamp_min(1).to(pair.dtype)
        pair_mean = (pair * valid).sum(dim=1) / count[:, None]
        pair_max = pair.masked_fill(~valid, -1e4).max(dim=1).values
        rank_logits = output["rank_logits"].masked_fill(~set_mask, -1e4)
        candidate_energy = torch.logsumexp(rank_logits, dim=-1) - count.log()
        null_evidence = self.null_evidence(torch.cat([
            self._query_features(query), pair_mean, pair_max,
        ], dim=-1)).squeeze(-1)
        energy_gap = candidate_energy - null_evidence

        # Zero initialization preserves the migrated V1.0 B decision exactly.
        energy_residual = torch.tanh(self.raw_scale) * energy_gap
        in_set_logit = output["in_set_logit"] + energy_residual
        in_set_probability = torch.sigmoid(in_set_logit)
        rf_probability = output["rf_probability"]
        sf_probability = 1 - rf_probability
        present_probability = sf_probability * in_set_probability
        absent_probability = sf_probability * (1 - in_set_probability)
        rank_probability = output["rank_probability"]
        candidate_probability = present_probability[:, None] * rank_probability
        unknown_probability = absent_probability + rf_probability
        type_probability = torch.stack([
            present_probability, absent_probability, rf_probability,
        ], dim=-1)
        joint_probability = torch.cat([
            candidate_probability, unknown_probability[:, None],
        ], dim=-1)
        official_probability = torch.cat([
            candidate_probability, absent_probability[:, None], rf_probability[:, None],
        ], dim=-1)
        output.update({
            "in_set_logit": in_set_logit,
            "in_set_probability": in_set_probability,
            "exist_logit": torch.logit(present_probability.clamp(1e-6, 1 - 1e-6)),
            "exist_probability": present_probability,
            "candidate_probability": candidate_probability,
            "absent_probability": absent_probability,
            "unknown_probability": unknown_probability,
            "type_probability": type_probability,
            "type_logits": type_probability.clamp_min(1e-8).log(),
            "joint_probability": joint_probability,
            "official_probability": official_probability,
            "open_set_logits": joint_probability.clamp_min(1e-8).log(),
            "relation_candidate_energy": candidate_energy,
            "relation_null_evidence": null_evidence,
            "relation_energy_gap": energy_gap,
            "relation_energy_residual": energy_residual,
        })
        return output


class ConditionalSENT2Head(nn.Module):
    """A/B/C head over the pair evidence shared with T1."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_dim
        self.source_adapter = T2SourceEvidenceAdapter(
            dim, config.t2_adapter_bottleneck, config.conformer_heads, config.dropout,
        )
        self.rank_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim // 2, 1),
        )
        self.rf_head = nn.Sequential(
            nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )
        self.in_set_head = nn.Sequential(
            nn.LayerNorm(dim * 2 + 5), nn.Linear(dim * 2 + 5, dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(dim, 1),
        )

    @staticmethod
    def _query_features(query: SignatureEncoding) -> Tensor:
        return torch.cat([
            query.global_shared, query.global_sequence, query.global_image,
        ], dim=-1)

    def forward_a(self, query: SignatureEncoding) -> dict[str, Tensor]:
        rf_logit = self.rf_head(self._query_features(query)).squeeze(-1)
        return {"rf_logit": rf_logit, "rf_probability": torch.sigmoid(rf_logit)}

    def forward(
        self, relation: dict[str, Tensor], query: SignatureEncoding, set_mask: Tensor,
        target_index: Tensor | None = None, episode_type_index: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del target_index, episode_type_index
        shared_pair = relation["pair_embedding"]
        pair = self.source_adapter(shared_pair, set_mask)
        rank_logits = self.rank_head(pair).squeeze(-1).masked_fill(~set_mask, -1e4)
        rank_probability = rank_logits.softmax(dim=-1)

        valid = set_mask.unsqueeze(-1)
        count = set_mask.sum(dim=-1).clamp_min(1).to(pair.dtype)
        pair_mean = (pair * valid).sum(dim=1) / count[:, None]
        pair_max = pair.masked_fill(~valid, -1e4).max(dim=1).values
        valid_logits = rank_logits.masked_fill(~set_mask, 0)
        score_mean = valid_logits.sum(dim=1) / count
        score_std = (
            ((rank_logits - score_mean[:, None]).square() * set_mask).sum(dim=1) / count
        ).sqrt()
        top = rank_logits.topk(k=min(2, rank_logits.shape[1]), dim=1).values
        gap = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else top[:, 0])
        entropy = -(
            rank_probability * rank_probability.clamp_min(1e-8).log()
        ).sum(dim=-1)
        set_features = torch.cat([
            pair_mean, pair_max, top[:, :1], gap[:, None], score_mean[:, None],
            score_std[:, None], entropy[:, None],
        ], dim=-1)

        rf_logit = self.forward_a(query)["rf_logit"]
        in_set_logit = self.in_set_head(set_features).squeeze(-1)
        rf_probability = torch.sigmoid(rf_logit)
        in_set_probability = torch.sigmoid(in_set_logit)
        sf_probability = 1 - rf_probability
        present_probability = sf_probability * in_set_probability
        absent_probability = sf_probability * (1 - in_set_probability)
        candidate_probability = present_probability[:, None] * rank_probability
        unknown_probability = absent_probability + rf_probability
        type_probability = torch.stack([
            present_probability, absent_probability, rf_probability,
        ], dim=-1)
        joint_probability = torch.cat([
            candidate_probability, unknown_probability[:, None],
        ], dim=-1)
        official_probability = torch.cat([
            candidate_probability, absent_probability[:, None], rf_probability[:, None],
        ], dim=-1)

        return {
            **relation,
            "source_pair_embedding": pair,
            "rank_logits": rank_logits,
            "rank_probability": rank_probability,
            "rf_logit": rf_logit,
            "rf_probability": rf_probability,
            "in_set_logit": in_set_logit,
            "in_set_probability": in_set_probability,
            "exist_logit": torch.logit(present_probability.clamp(1e-6, 1 - 1e-6)),
            "exist_probability": present_probability,
            "candidate_probability": candidate_probability,
            "absent_probability": absent_probability,
            "unknown_probability": unknown_probability,
            "type_probability": type_probability,
            "type_logits": type_probability.clamp_min(1e-8).log(),
            "joint_probability": joint_probability,
            "official_probability": official_probability,
            "open_set_logits": joint_probability.clamp_min(1e-8).log(),
        }


def _gather_set(encoding: SignatureEncoding, index: Tensor) -> SignatureEncoding:
    return SignatureEncoding(
        global_shared=encoding.global_shared[index], local_shared=encoding.local_shared[index],
        valid_mask=encoding.valid_mask[index], spatial=encoding.spatial[index],
        global_sequence=encoding.global_sequence[index], global_image=encoding.global_image[index],
        diagnostics={key: value[index] for key, value in encoding.diagnostics.items()},
    )


def _concat_set_encodings(left: SignatureEncoding, right: SignatureEncoding) -> SignatureEncoding:
    return SignatureEncoding(
        global_shared=torch.cat([left.global_shared, right.global_shared], dim=1),
        local_shared=torch.cat([left.local_shared, right.local_shared], dim=1),
        valid_mask=torch.cat([left.valid_mask, right.valid_mask], dim=1),
        spatial=torch.cat([left.spatial, right.spatial], dim=1),
        global_sequence=torch.cat([left.global_sequence, right.global_sequence], dim=1),
        global_image=torch.cat([left.global_image, right.global_image], dim=1),
        diagnostics={},
    )


class DVSRNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.variant not in {
            "v2", "v3", "v4", "v5", "v5r1", "v6", "v7", "v8", "v9", "v10",
            "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
            "unified_abc_v14", "unified_sen_v20",
        }:
            raise ValueError(f"Unsupported model variant: {config.variant}")
        if config.variant == "unified_sen_v20" and config.t1_variant in {"simple_v1", "simple_v2"}:
            raise ValueError("simple_v1 cannot use the unified_sen_v20 shared local matcher")
        self.config = config
        self.encoder = SignatureEncoder(config)
        self.t1 = T1Head(config, config.t1_variant)
        self.shared_relation = (
            SharedPairEvidenceMatcher(config.hidden_dim, config.conformer_heads, config.dropout)
            if config.variant == "unified_sen_v20" else None
        )
        self.t2_energy_adapter = (
            T2RelationEnergyAdapter(config.hidden_dim, config.dropout)
            if config.t2_relation_energy_enabled else None
        )
        self.t2_encoder = SignatureEncoder(config) if config.variant in {"v9", "t2_abc_v1"} else None
        self.t2_rank_encoder = SignatureEncoder(config) if config.variant == "v10" else None
        self.t2_open_encoder = SignatureEncoder(config) if config.variant == "v10" else None
        if config.variant == "unified_abc_v14":
            self.t2_adapter = T2InternalAdapter(config)
        elif config.variant in {"unified_abc_v12", "unified_abc_v13"}:
            self.t2_adapter = T2ResidualAdapter(config.hidden_dim, config.t2_adapter_bottleneck)
        else:
            self.t2_adapter = None
        if config.variant in {
            "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
            "unified_abc_v14",
        }:
            self.t2 = ConditionalABCT2Head(config)
        elif config.variant == "unified_sen_v20":
            self.t2 = ConditionalSENT2Head(config)
        elif config.variant == "v10":
            self.t2 = DualEvidenceT2Head(config)
        elif config.variant == "v9":
            self.t2 = UnifiedEvidenceT2Head(config)
        elif config.variant == "v8":
            self.t2 = StatefulBayesianT2Head(config)
        elif config.variant == "v7":
            self.t2 = ProgressiveBayesianT2Head(config)
        else:
            self.t2 = T2Head(config)

    def forward(
        self, batch: dict[str, Any], return_encoder_stages: bool = False,
    ) -> dict[str, Any]:
        if batch["protocol"] == "t2_a_query":
            if self.config.variant not in {
                "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
                "unified_abc_v14", "unified_sen_v20",
            }:
                raise RuntimeError("Balanced T2-A queries require a conditional ABC model variant")
            encoder = self.t2_encoder if self.config.variant == "t2_abc_v1" else self.encoder
            assert encoder is not None
            arguments = (
                batch["sequence"], batch["sequence_mask"], batch["image"],
                batch["anchors"], batch["anchor_mask"],
            )
            encoder_stages: dict[str, Tensor] = {}
            if self.config.variant == "unified_abc_v14":
                assert isinstance(self.t2_adapter, T2InternalAdapter)
                if return_encoder_stages:
                    encoding, encoder_stages = encoder.forward_with_intermediates(
                        *arguments, task_adapter=self.t2_adapter,
                    )
                else:
                    encoding = encoder(*arguments, task_adapter=self.t2_adapter)
            else:
                encoding = encoder(*arguments)
            if self.t2_adapter is not None and self.config.variant != "unified_abc_v14":
                encoding = self.t2_adapter(encoding)
            output = self.t2.forward_a(encoding.select(batch["query_index"]))
            if return_encoder_stages:
                output["encoder_stages"] = encoder_stages
            return output
        if self.config.variant == "v10" and not batch["protocol"].startswith("t1"):
            assert self.t2_rank_encoder is not None and self.t2_open_encoder is not None
            arguments = (
                batch["sequence"], batch["sequence_mask"], batch["image"],
                batch["anchors"], batch["anchor_mask"],
            )
            rank_encoding = self.t2_rank_encoder(*arguments)
            rank_query = rank_encoding.select(batch["query_index"])
            rank_members = _gather_set(rank_encoding, batch["set_index"])
            open_encoding = self.t2_open_encoder(*arguments)
            open_query = open_encoding.select(batch["query_index"])
            open_members = _gather_set(open_encoding, batch["set_index"])
            output = self.t2(
                rank_members, rank_query, open_members, open_query, batch["set_mask"],
                batch.get("target_index"), batch.get("episode_type_index"),
            )
            output["view_global_loss"] = rank_encoding.global_shared.new_zeros(())
            output["view_local_loss"] = rank_encoding.global_shared.new_zeros(())
            return output
        encoder = self.encoder
        if self.config.variant in {"v9", "t2_abc_v1"} and not batch["protocol"].startswith("t1"):
            assert self.t2_encoder is not None
            encoder = self.t2_encoder
        arguments = (
            batch["sequence"], batch["sequence_mask"], batch["image"],
            batch["anchors"], batch["anchor_mask"],
        )
        use_internal_adapter = (
            self.config.variant == "unified_abc_v14"
            and not batch["protocol"].startswith("t1")
        )
        encoder_stages: dict[str, Tensor] = {}
        if return_encoder_stages:
            shared_encoding, encoder_stages = encoder.forward_with_intermediates(
                *arguments,
                task_adapter=self.t2_adapter if use_internal_adapter else None,
            )
        else:
            shared_encoding = encoder(
                *arguments,
                task_adapter=self.t2_adapter if use_internal_adapter else None,
            )
        encoding = shared_encoding
        if (
            self.t2_adapter is not None
            and self.config.variant != "unified_abc_v14"
            and not batch["protocol"].startswith("t1")
        ):
            encoding = self.t2_adapter(encoding)
        query = encoding.select(batch["query_index"])
        members = _gather_set(encoding, batch["set_index"])
        if batch["protocol"].startswith("t1"):
            output = self.t1(
                members, query, batch["set_mask"], shared_relation=self.shared_relation,
            )
        else:
            if self.config.variant in {"v6", "v7", "v8"}:
                members, query = members.detach(), query.detach()
            if self.config.variant == "unified_sen_v20":
                assert self.shared_relation is not None
                relation = self.shared_relation(members, query)
                output = self.t2(
                    relation, query, batch["set_mask"],
                    batch.get("target_index"), batch.get("episode_type_index"),
                )
            else:
                output = self.t2(
                    members, query, batch["set_mask"],
                    batch.get("target_index"), batch.get("episode_type_index"),
                )
            if self.t2_energy_adapter is not None:
                output = self.t2_energy_adapter(output, query, batch["set_mask"])
            if self.config.variant in {
                "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                "unified_sen_v20",
            }:
                if self.config.variant in {
                    "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                }:
                    t1_query = shared_encoding.select(batch["query_index"])
                    t1_members = _gather_set(shared_encoding, batch["set_index"])
                else:
                    t1_query, t1_members = query, members
                t1_case_logit = self.t1(
                    t1_members, t1_query, batch["set_mask"],
                    shared_relation=self.shared_relation,
                )["case_logit"]
                t1_genuine_probability = torch.sigmoid(t1_case_logit)
                output["t1_case_logit"] = t1_case_logit
                output["t1_genuine_probability"] = t1_genuine_probability
                output["t1_forgery_probability"] = 1 - t1_genuine_probability
                if "official_probability" in output:
                    full_probability = torch.cat([
                        t1_genuine_probability[:, None],
                        (1 - t1_genuine_probability)[:, None] * output["official_probability"],
                    ], dim=-1)
                    output["full_hierarchical_probability"] = full_probability
                    output["full_hierarchical_logits"] = full_probability.clamp_min(1e-8).log()
                if batch.get("compute_t1_bridge", False):
                    output["bridge_case_logit"] = t1_case_logit
        if self.config.variant in {
            "v6", "v7", "v8", "v9", "t2_abc_v1", "unified_abc_v11", "unified_abc_v12",
            "unified_abc_v13", "unified_abc_v14", "unified_sen_v20",
        } and not batch["protocol"].startswith("t1"):
            output["view_global_loss"] = encoding.global_shared.new_zeros(())
            output["view_local_loss"] = encoding.global_shared.new_zeros(())
            if return_encoder_stages:
                output["encoder_stages"] = encoder_stages
            return output
        if self.config.fusion in {"sequence_only", "image_only"}:
            output["view_global_loss"] = encoding.global_shared.new_zeros(())
        else:
            output["view_global_loss"] = (
                1 - F.cosine_similarity(encoding.global_sequence, encoding.global_image, dim=-1)
            ).mean()
        if "aligned_visual" in encoding.diagnostics:
            valid = encoding.diagnostics["anchor_valid"]
            local_cosine = 1 - F.cosine_similarity(
                encoding.local_shared, encoding.diagnostics["aligned_visual"], dim=-1,
            )
            output["view_local_loss"] = (local_cosine * valid).sum() / valid.sum().clamp_min(1)
        else:
            output["view_local_loss"] = output["view_global_loss"].new_zeros(())
        if return_encoder_stages:
            output["encoder_stages"] = encoder_stages
        return output

    def encode_materials(self, sequence: Tensor, sequence_mask: Tensor, image: Tensor,
                         anchors: Tensor, anchor_mask: Tensor) -> SignatureEncoding | DualSignatureEncoding:
        """Public encoding boundary for stateful T2 inference adapters."""
        if self.config.variant == "v10":
            assert self.t2_rank_encoder is not None and self.t2_open_encoder is not None
            arguments = (sequence, sequence_mask, image, anchors, anchor_mask)
            return DualSignatureEncoding(
                self.t2_rank_encoder(*arguments).detach(),
                self.t2_open_encoder(*arguments).detach(),
            )
        encoder = self.t2_encoder if self.config.variant in {"v9", "t2_abc_v1"} else self.encoder
        assert encoder is not None
        arguments = (sequence, sequence_mask, image, anchors, anchor_mask)
        if self.config.variant == "unified_abc_v14":
            assert isinstance(self.t2_adapter, T2InternalAdapter)
            encoding = encoder(*arguments, task_adapter=self.t2_adapter)
        else:
            encoding = encoder(*arguments)
        if self.t2_adapter is not None and self.config.variant != "unified_abc_v14":
            encoding = self.t2_adapter(encoding)
        return encoding.detach()

    def begin_t2_state(
        self, query: SignatureEncoding | DualSignatureEncoding,
    ) -> BayesianEvidenceState | DualBayesianEvidenceState:
        if self.config.variant not in {"v8", "v9", "v10"}:
            raise RuntimeError("Stateful T2 inference requires model variant v8, v9, or v10")
        if self.config.variant == "v10" and not isinstance(query, DualSignatureEncoding):
            raise TypeError("V10 stateful inference requires DualSignatureEncoding")
        return self.t2.begin_state(query)

    def append_t2_state(
        self, state: BayesianEvidenceState | DualBayesianEvidenceState,
        candidates: SignatureEncoding | DualSignatureEncoding, set_mask: Tensor | None = None,
    ) -> tuple[BayesianEvidenceState | DualBayesianEvidenceState, dict[str, Tensor]]:
        if self.config.variant not in {"v8", "v9", "v10"}:
            raise RuntimeError("Stateful T2 inference requires model variant v8, v9, or v10")
        candidate_rank = candidates.rank if isinstance(candidates, DualSignatureEncoding) else candidates
        if candidate_rank.global_shared.ndim == 2:
            if len(state.posterior) != 1:
                raise ValueError("Flat candidate material input is only valid for a single active session")
            if isinstance(candidates, DualSignatureEncoding):
                candidates = DualSignatureEncoding(
                    candidates.rank.as_single_set(), candidates.open.as_single_set(),
                )
            else:
                candidates = candidates.as_single_set()
            candidate_rank = candidates.rank if isinstance(candidates, DualSignatureEncoding) else candidates
        if set_mask is None:
            set_mask = torch.ones(
                candidate_rank.global_shared.shape[:2], dtype=torch.bool,
                device=candidate_rank.global_shared.device,
            )
        return self.t2.append_state(state, candidates, set_mask)

    def parameter_groups(self, backbone_lr: float, encoder_lr: float, head_lr: float) -> list[dict[str, Any]]:
        image_parameters = list(self.encoder.image.features.parameters())
        if self.t2_encoder is not None:
            image_parameters += list(self.t2_encoder.image.features.parameters())
        for encoder in (self.t2_rank_encoder, self.t2_open_encoder):
            if encoder is not None:
                image_parameters += list(encoder.image.features.parameters())
        image_ids = {id(parameter) for parameter in image_parameters}
        head_parameters = list(self.t1.parameters()) + list(self.t2.parameters())
        if self.t2_adapter is not None:
            head_parameters += list(self.t2_adapter.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        encoder_parameters = [p for p in self.parameters() if id(p) not in image_ids | head_ids]
        return [
            {"params": image_parameters, "lr": backbone_lr, "name": "convnext_backbone"},
            {"params": encoder_parameters, "lr": encoder_lr, "name": "shared_encoder"},
            {"params": head_parameters, "lr": head_lr, "name": "task_heads"},
        ]

    def custom_parameter_count(self) -> int:
        backbone_ids = {id(p) for p in self.encoder.image.features.parameters()}
        if self.t2_encoder is not None:
            backbone_ids.update(id(p) for p in self.t2_encoder.image.features.parameters())
        return sum(p.numel() for p in self.parameters() if id(p) not in backbone_ids)
