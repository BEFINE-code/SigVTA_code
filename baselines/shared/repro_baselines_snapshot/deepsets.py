from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import BenchmarkRepository
from .evaluation import _finish_t2
from .features import FeatureStore


def episode_pair_features(features: FeatureStore, rows: list[dict[str, Any]]) -> np.ndarray:
    pairs = [
        (candidate, row["query_id"])
        for row in rows
        for candidate in row["candidate_ids"]
    ]
    candidate_counts = {len(row["candidate_ids"]) for row in rows}
    if len(candidate_counts) != 1:
        raise RuntimeError(f"Inconsistent T2 candidate counts: {sorted(candidate_counts)}")
    return features.pair_matrix(pairs).reshape(len(rows), candidate_counts.pop(), -1)


class DeepSetsModel(nn.Module):
    def __init__(self, pair_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        context_dim = hidden_dim * 2
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.unknown_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.phi(pairs)
        context = torch.cat([encoded.mean(dim=1), encoded.max(dim=1).values], dim=1)
        expanded = context[:, None, :].expand(-1, encoded.shape[1], -1)
        candidate_logits = self.rank_head(torch.cat([encoded, expanded], dim=2)).squeeze(2)
        unknown_logit = self.unknown_head(context).squeeze(1)
        return candidate_logits, unknown_logit


@dataclass(frozen=True)
class DeepSetsConfig:
    hidden_dim: int = 128
    batch_size: int = 256
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5
    existence_loss_weight: float = 0.25
    rank_loss_weight: float = 0.25


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _targets(rows: list[dict[str, Any]], candidate_count: int) -> np.ndarray:
    return np.asarray([
        row["target_index"] if row["target_index"] >= 0 else candidate_count for row in rows
    ], dtype=np.int64)


def _score(
    model: DeepSetsModel, matrix: np.ndarray, device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(matrix)), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    candidate_values: list[np.ndarray] = []
    unknown_values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for (pairs,) in loader:
            candidate, unknown = model(pairs.to(device, non_blocking=True))
            candidate_values.append(candidate.float().cpu().numpy())
            unknown_values.append(unknown.float().cpu().numpy())
    return np.concatenate(candidate_values), np.concatenate(unknown_values)


def fit_and_evaluate_deepsets(
    repo: BenchmarkRepository,
    features: FeatureStore,
    output: Path,
    seed: int,
    config: DeepSetsConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or DeepSetsConfig()
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = repo.episodes("t2", "train")
    validation_rows = repo.episodes("t2", "val")
    train_matrix = episode_pair_features(features, train_rows)
    validation_matrix = episode_pair_features(features, validation_rows)
    candidate_count = train_matrix.shape[1]
    train_targets = _targets(train_rows, candidate_count)
    validation_targets = _targets(validation_rows, candidate_count)
    dataset = TensorDataset(
        torch.from_numpy(train_matrix), torch.from_numpy(train_targets),
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=True,
    )
    model = DeepSetsModel(train_matrix.shape[2], config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    joint_criterion = nn.CrossEntropyLoss()
    existence_criterion = nn.BCEWithLogitsLoss()
    rank_criterion = nn.CrossEntropyLoss()
    best_loss = np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for pairs, targets in loader:
            pairs = pairs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            candidate_logits, unknown_logits = model(pairs)
            joint_logits = torch.cat([candidate_logits, unknown_logits[:, None]], dim=1)
            joint_loss = joint_criterion(joint_logits, targets)
            exist_labels = (targets < candidate_count).float()
            existence_logits = candidate_logits.max(dim=1).values - unknown_logits
            existence_loss = existence_criterion(existence_logits, exist_labels)
            present = targets < candidate_count
            rank_loss = (
                rank_criterion(candidate_logits[present], targets[present])
                if present.any() else torch.zeros((), device=device)
            )
            loss = (
                joint_loss
                + config.existence_loss_weight * existence_loss
                + config.rank_loss_weight * rank_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            total_examples += len(targets)
        val_candidate, val_unknown = _score(
            model, validation_matrix, device, config.batch_size,
        )
        logits = np.column_stack([val_candidate, val_unknown])
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        validation_loss = float(-np.log(
            probabilities[np.arange(len(validation_targets)), validation_targets].clip(min=1e-12)
        ).mean())
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_joint_nll": validation_loss,
        })
        print(
            f"deepsets epoch={epoch} loss={history[-1]['train_loss']:.6f} "
            f"val_joint_nll={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("DeepSets training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "deepsets_t2",
        "state_dict": best_state,
        "config": asdict(config),
        "seed": seed,
        "validation_joint_nll": best_loss,
    }, output / "t2_model.pt")

    val_candidate, val_unknown = _score(model, validation_matrix, device, config.batch_size)
    test_rows = repo.episodes("t2", "test", final=True)
    test_matrix = episode_pair_features(features, test_rows)
    test_candidate, test_unknown = _score(model, test_matrix, device, config.batch_size)
    evaluation = _finish_t2(
        validation_rows, test_rows, val_candidate, test_candidate,
        val_candidate.max(axis=1) - val_unknown,
        test_candidate.max(axis=1) - test_unknown,
        output,
    )
    fit_metadata = {
        "architecture": "deepsets_t2",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "pair_feature_dim": int(train_matrix.shape[2]),
        "training_episodes": len(train_rows),
        "best_epoch": best_epoch,
        "best_validation_joint_nll": best_loss,
        "fit_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "history": history,
    }
    return evaluation, fit_metadata
