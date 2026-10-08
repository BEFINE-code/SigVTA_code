from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dtaidistance import dtw, dtw_ndim
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .deep import BiLSTMEncoder, SequenceStore, SiamesePairModel


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.astype(np.float32, copy=False)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    return np.stack([
        np.interp(target, source, values[:, channel]) for channel in range(values.shape[1])
    ], axis=1).astype(np.float32)


class AlignedPairStore:
    def __init__(self, sequences: SequenceStore, window_fraction: float = 0.2) -> None:
        self.sequences = sequences
        self.window_fraction = window_fraction
        self.cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def _align(self, key: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
        first = self.sequences.sequences[self.sequences.index[key[0]]]
        second = self.sequences.sequences[self.sequences.index[key[1]]]
        alignment_channels = [0, 1, 2, 3, 5, 6]
        first_metric = np.ascontiguousarray(first[:, alignment_channels], dtype=np.float64)
        second_metric = np.ascontiguousarray(second[:, alignment_channels], dtype=np.float64)
        window = max(
            abs(len(first) - len(second)),
            int(max(len(first), len(second)) * self.window_fraction),
            1,
        )
        _, paths = dtw_ndim.warping_paths_fast(
            first_metric, second_metric, window=window, inner_dist="euclidean",
        )
        path = dtw.best_path(paths)
        first_aligned = _resample(first[[row for row, _ in path]], self.sequences.length)
        second_aligned = _resample(second[[column for _, column in path]], self.sequences.length)
        return first_aligned, second_aligned

    def arrays(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        missing = sorted({tuple(sorted(pair)) for pair in pairs} - self.cache.keys())
        for index, key in enumerate(missing, start=1):
            self.cache[key] = self._align(key)
            if index % 2000 == 0:
                print(f"TA-RNN align {index}/{len(missing)} new pairs", flush=True)
        first_values: list[np.ndarray] = []
        second_values: list[np.ndarray] = []
        for pair in pairs:
            key = tuple(sorted(pair))
            first, second = self.cache[key]
            if pair != key:
                first, second = second, first
            first_values.append(first)
            second_values.append(second)
        return np.stack(first_values), np.stack(second_values)


class AlignedPairDataset(Dataset):
    def __init__(self, store: AlignedPairStore, pairs: list[tuple[str, str]], labels: np.ndarray) -> None:
        first, second = store.arrays(pairs)
        self.first = torch.from_numpy(first)
        self.second = torch.from_numpy(second)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.first[index], self.second[index], self.labels[index]


@dataclass(frozen=True)
class TARNNConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 10
    patience: int = 3
    window_fraction: float = 0.2


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TARNNScorer:
    def __init__(
        self, store: AlignedPairStore, model: SiamesePairModel,
        device: torch.device, batch_size: int,
    ) -> None:
        self.store = store
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        loader = DataLoader(
            AlignedPairDataset(self.store, pairs, np.zeros(len(pairs), dtype=np.float32)),
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


class MetricTARNNModel(nn.Module):
    def __init__(self, channels: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = BiLSTMEncoder(channels, embedding_dim)

    def embed(self, sequence: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.encoder(sequence), p=2, dim=1)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(self.embed(first) - self.embed(second), dim=1)


def contrastive_loss(
    distances: torch.Tensor, labels: torch.Tensor, margin: float, positive_weight: float,
) -> torch.Tensor:
    positive = labels * distances.square() * positive_weight
    negative = (1.0 - labels) * torch.relu(margin - distances).square()
    weights = labels * positive_weight + (1.0 - labels)
    return (positive + negative).sum() / weights.sum().clamp_min(1.0)


class MetricTARNNScorer:
    def __init__(
        self, store: AlignedPairStore, model: MetricTARNNModel,
        device: torch.device, batch_size: int,
    ) -> None:
        self.store = store
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        loader = DataLoader(
            AlignedPairDataset(self.store, pairs, np.zeros(len(pairs), dtype=np.float32)),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
        values: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for first, second, _ in loader:
                distances = self.model(
                    first.to(self.device, non_blocking=True),
                    second.to(self.device, non_blocking=True),
                )
                values.append((-distances).float().cpu().numpy())
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def fit_metric_tarnn_pair_model(
    aligned_store: AlignedPairStore,
    train_pairs: list[tuple[str, str]],
    train_labels: np.ndarray,
    validation_pairs: list[tuple[str, str]],
    validation_labels: np.ndarray,
    seed: int,
    checkpoint_path: str | Path,
    config: TARNNConfig | None = None,
    margin: float = 1.0,
) -> tuple[MetricTARNNScorer, dict[str, Any]]:
    config = config or TARNNConfig(window_fraction=aligned_store.window_fraction)
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricTARNNModel(aligned_store.sequences.channels).to(device)
    loader = DataLoader(
        AlignedPairDataset(aligned_store, train_pairs, train_labels),
        batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
        num_workers=0, pin_memory=True,
    )
    positives = max(float(train_labels.sum()), 1.0)
    negatives = max(float((train_labels == 0).sum()), 1.0)
    positive_weight = negatives / positives
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scorer = MetricTARNNScorer(aligned_store, model, device, config.batch_size)
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
            loss = contrastive_loss(model(first, second), labels, margin, positive_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)
        validation_auc = float(roc_auc_score(validation_labels, scorer.score_pairs(validation_pairs)))
        epoch_loss = total_loss / max(total_examples, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "validation_auc": validation_auc})
        print(
            f"ta-rnn-contrastive epoch={epoch} loss={epoch_loss:.6f} val_auc={validation_auc:.6f}",
            flush=True,
        )
        if validation_auc > best_auc + 1e-5:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Metric TA-RNN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "ta_rnn_contrastive_adapted",
        "state_dict": best_state,
        "config": asdict(config),
        "margin": margin,
        "seed": seed,
        "validation_auc": best_auc,
    }, checkpoint_path)
    metadata = {
        "architecture": "ta_rnn_contrastive_adapted",
        "adaptation": (
            "DTW alignment on fixed 128-point train-normalized dynamic sequences, "
            "L2-normalized Siamese BiLSTM embeddings, and weighted contrastive loss"
        ),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_pairs": len(train_pairs),
        "positive_pairs": int(train_labels.sum()),
        "negative_pairs": int((train_labels == 0).sum()),
        "positive_weight": positive_weight,
        "margin": margin,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "fit_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "history": history,
    }
    return scorer, metadata


def fit_tarnn_pair_model(
    aligned_store: AlignedPairStore,
    train_pairs: list[tuple[str, str]],
    train_labels: np.ndarray,
    validation_pairs: list[tuple[str, str]],
    validation_labels: np.ndarray,
    seed: int,
    checkpoint_path: str | Path,
    config: TARNNConfig | None = None,
) -> tuple[TARNNScorer, dict[str, Any]]:
    config = config or TARNNConfig(window_fraction=aligned_store.window_fraction)
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamesePairModel(
        "bilstm", aligned_store.sequences.channels, aligned_store.sequences.length,
    ).to(device)
    loader = DataLoader(
        AlignedPairDataset(aligned_store, train_pairs, train_labels),
        batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
        num_workers=0, pin_memory=True,
    )
    positives = max(float(train_labels.sum()), 1.0)
    negatives = max(float((train_labels == 0).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scorer = TARNNScorer(aligned_store, model, device, config.batch_size)
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
        print(f"ta-rnn epoch={epoch} loss={epoch_loss:.6f} val_auc={validation_auc:.6f}", flush=True)
        if validation_auc > best_auc + 1e-5:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("TA-RNN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "ta_rnn_adapted", "state_dict": best_state, "config": asdict(config),
        "seed": seed, "validation_auc": best_auc,
    }, checkpoint_path)
    metadata = {
        "architecture": "ta_rnn_adapted",
        "adaptation": "DTW alignment on fixed 128-point train-normalized dynamic sequences, then Siamese BiLSTM",
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
