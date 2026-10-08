from __future__ import annotations

import copy
import hashlib
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from dvsrc.data import DynamicRGBRenderConfig, DynamicRGBRenderer

from .data import SignatureStore


class ImageStore:
    def __init__(
        self,
        store: SignatureStore,
        rendered_root: str | Path,
        cache_path: str | Path,
        size: int = 160,
    ) -> None:
        self.store = store
        self.rendered_root = Path(rendered_root)
        self.cache_path = Path(cache_path)
        self.size = size
        if self.cache_path.is_file():
            cached = np.load(self.cache_path, allow_pickle=False)
            self.ids = [str(value) for value in cached["ids"]]
            self.images = cached["images"].astype(np.uint8)
        else:
            self.ids = sorted(store.samples)
            renderer = DynamicRGBRenderer(DynamicRGBRenderConfig())
            images: list[np.ndarray] = []
            for index, sample_id in enumerate(self.ids, start=1):
                sample = store.samples[sample_id]
                key = hashlib.sha256((sample["sha256"] + renderer.config.digest).encode()).hexdigest()
                png_path = self.rendered_root / f"{key}.png"
                if png_path.is_file():
                    with Image.open(png_path) as source:
                        image = source.convert("RGB")
                else:
                    image, _ = renderer.render(store.load(sample_id))
                image = image.resize((size, size), Image.Resampling.BILINEAR)
                images.append(np.asarray(image, dtype=np.uint8))
                if index % 1000 == 0:
                    print(f"image cache {index}/{len(self.ids)}", flush=True)
            self.images = np.stack(images)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                ids=np.asarray(self.ids, dtype="U64"),
                images=self.images,
            )
        self.index = {sample_id: index for index, sample_id in enumerate(self.ids)}


class ImagePairDataset(Dataset):
    def __init__(self, store: ImageStore, pairs: list[tuple[str, str]], labels: np.ndarray) -> None:
        self.images = torch.from_numpy(store.images)
        self.first = torch.as_tensor([store.index[pair[0]] for pair in pairs], dtype=torch.long)
        self.second = torch.as_tensor([store.index[pair[1]] for pair in pairs], dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self.images[self.first[index]].permute(2, 0, 1).float().div(255.0)
        second = self.images[self.second[index]].permute(2, 0, 1).float().div(255.0)
        return first, second, self.labels[index]


class ResNet18PairModel(nn.Module):
    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projection = nn.Linear(512, 128)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder((image - self.mean) / self.std))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_embedding = self.embed(first)
        second_embedding = self.embed(second)
        symmetric = torch.cat([
            torch.abs(first_embedding - second_embedding), first_embedding * second_embedding,
        ], dim=1)
        return self.head(symmetric).squeeze(1)


@dataclass(frozen=True)
class ImageConfig:
    image_size: int = 160
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 8
    patience: int = 2


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImagePairScorer:
    def __init__(self, store: ImageStore, model: ResNet18PairModel, device: torch.device, batch_size: int) -> None:
        self.store = store
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        loader = DataLoader(
            ImagePairDataset(self.store, pairs, np.zeros(len(pairs), dtype=np.float32)),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
        values: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for first, second, _ in loader:
                logits = self.model(
                    first.to(self.device, non_blocking=True), second.to(self.device, non_blocking=True),
                )
                values.append(logits.float().cpu().numpy())
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def fit_resnet18_pair_model(
    store: ImageStore,
    train_pairs: list[tuple[str, str]],
    train_labels: np.ndarray,
    validation_pairs: list[tuple[str, str]],
    validation_labels: np.ndarray,
    seed: int,
    weights_path: str | Path,
    checkpoint_path: str | Path,
    config: ImageConfig | None = None,
) -> tuple[ImagePairScorer, dict[str, Any]]:
    config = config or ImageConfig(image_size=store.size)
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18PairModel(weights_path).to(device)
    loader = DataLoader(
        ImagePairDataset(store, train_pairs, train_labels),
        batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
        num_workers=0, pin_memory=True,
    )
    positives = max(float(train_labels.sum()), 1.0)
    negatives = max(float((train_labels == 0).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scorer = ImagePairScorer(store, model, device, config.batch_size)
    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for first, second, labels in loader:
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(first, second), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)
        validation_auc = float(roc_auc_score(validation_labels, scorer.score_pairs(validation_pairs)))
        epoch_loss = total_loss / max(total_examples, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "validation_auc": validation_auc})
        print(f"resnet18 epoch={epoch} loss={epoch_loss:.6f} val_auc={validation_auc:.6f}", flush=True)
        if validation_auc > best_auc + 1e-5:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("ResNet-18 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "resnet18", "state_dict": best_state, "config": asdict(config),
        "seed": seed, "validation_auc": best_auc,
    }, checkpoint_path)
    metadata = {
        "architecture": "resnet18",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_pairs": len(train_pairs),
        "positive_pairs": int(train_labels.sum()),
        "negative_pairs": int((train_labels == 0).sum()),
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "fit_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "history": history,
    }
    return scorer, metadata
