from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import SignatureStore


def sequence_features(raw: np.ndarray, length: int = 128) -> np.ndarray:
    time_values, x, y, pressure, speed, direction, pen = raw.T
    points = np.stack([x, y], axis=1).astype(np.float64)
    active = pen > 0.5
    center_source = points[active] if active.any() else points
    points -= center_source.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(center_source - center_source.mean(axis=0, keepdims=True), axis=1).max()
    points /= max(float(scale), 1e-6)
    delta = np.vstack([np.zeros((1, 2)), np.diff(points, axis=0)])
    delta_time = np.r_[0.0, np.maximum(np.diff(time_values), 0.0)]
    values = np.column_stack([
        points,
        delta,
        np.log1p(delta_time),
        pressure,
        np.log1p(np.maximum(speed, 0.0)),
        np.sin(np.deg2rad(direction)),
        np.cos(np.deg2rad(direction)),
        pen,
    ])
    if len(values) == 1:
        return np.repeat(values.astype(np.float32), length, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    result = np.stack([np.interp(target, source, values[:, index]) for index in range(values.shape[1])], axis=1)
    result[:, -1] = (result[:, -1] >= 0.5).astype(np.float64)
    return result.astype(np.float32)


def lnps_features(raw: np.ndarray, length: int = 128) -> np.ndarray:
    active = raw[:, 6] > 0.5
    points = raw[active, 1:3] if active.any() else raw[:, 1:3]
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    increments = np.diff(points, axis=0)
    path_length = np.linalg.norm(increments, axis=1).sum()
    normalized = (points - points[0]) / max(float(path_length), 1e-6)
    source = np.linspace(0.0, 1.0, len(normalized))
    target = np.linspace(0.0, 1.0, length)
    path = np.stack([
        np.interp(target, source, normalized[:, channel]) for channel in range(2)
    ], axis=1)
    signature = np.zeros((length, 6), dtype=np.float64)
    level_one = np.zeros(2, dtype=np.float64)
    level_two = np.zeros((2, 2), dtype=np.float64)
    for index, increment in enumerate(np.diff(path, axis=0), start=1):
        level_two += np.outer(level_one, increment) + 0.5 * np.outer(increment, increment)
        level_one += increment
        signature[index, :2] = level_one
        signature[index, 2:] = level_two.reshape(-1)
    return signature.astype(np.float32)


class SequenceStore:
    def __init__(
        self, store: SignatureStore, train_writers: set[str], cache_path: str | Path, length: int = 128,
    ) -> None:
        self.store = store
        self.cache_path = Path(cache_path)
        self.length = length
        if self.cache_path.is_file():
            cached = np.load(self.cache_path, allow_pickle=False)
            self.ids = [str(value) for value in cached["ids"]]
            self.sequences = cached["sequences"].astype(np.float32)
        else:
            self.ids = sorted(store.samples)
            self.sequences = np.stack([
                sequence_features(store.load(sample_id), length=length) for sample_id in self.ids
            ])
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                ids=np.asarray(self.ids, dtype="U64"),
                sequences=self.sequences,
            )
        self.index = {sample_id: index for index, sample_id in enumerate(self.ids)}
        train_indices = [
            self.index[sample_id] for sample_id, sample in store.samples.items()
            if sample["writer_id"] in train_writers
        ]
        train = self.sequences[train_indices]
        self.mean = train.mean(axis=(0, 1), keepdims=True)
        self.std = train.std(axis=(0, 1), keepdims=True).clip(min=1e-5)
        self.sequences = ((self.sequences - self.mean) / self.std).astype(np.float32)

    @property
    def channels(self) -> int:
        return int(self.sequences.shape[-1])


class LNPSSequenceStore:
    def __init__(
        self, store: SignatureStore, train_writers: set[str], cache_path: str | Path, length: int = 128,
    ) -> None:
        self.store = store
        self.cache_path = Path(cache_path)
        self.length = length
        if self.cache_path.is_file():
            cached = np.load(self.cache_path, allow_pickle=False)
            self.ids = [str(value) for value in cached["ids"]]
            self.sequences = cached["sequences"].astype(np.float32)
        else:
            self.ids = sorted(store.samples)
            self.sequences = np.stack([
                lnps_features(store.load(sample_id), length=length) for sample_id in self.ids
            ])
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.cache_path,
                ids=np.asarray(self.ids, dtype="U64"),
                sequences=self.sequences,
            )
        self.index = {sample_id: index for index, sample_id in enumerate(self.ids)}
        train_indices = [
            self.index[sample_id] for sample_id, sample in store.samples.items()
            if sample["writer_id"] in train_writers
        ]
        train = self.sequences[train_indices]
        self.mean = train.mean(axis=(0, 1), keepdims=True)
        self.std = train.std(axis=(0, 1), keepdims=True).clip(min=1e-5)
        self.sequences = ((self.sequences - self.mean) / self.std).astype(np.float32)

    @property
    def channels(self) -> int:
        return int(self.sequences.shape[-1])


class PairDataset(Dataset):
    def __init__(self, store: SequenceStore, pairs: list[tuple[str, str]], labels: np.ndarray) -> None:
        self.sequences = torch.from_numpy(store.sequences)
        self.first = torch.as_tensor([store.index[pair[0]] for pair in pairs], dtype=torch.long)
        self.second = torch.as_tensor([store.index[pair[1]] for pair in pairs], dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[self.first[index]], self.sequences[self.second[index]], self.labels[index]


class CNNEncoder(nn.Module):
    def __init__(self, channels: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, embedding_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.network(sequence.transpose(1, 2))


class BiLSTMEncoder(nn.Module):
    def __init__(self, channels: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            channels, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1,
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(sequence)
        return self.projection(encoded.mean(dim=1))


class TransformerEncoder(nn.Module):
    def __init__(self, channels: int, length: int, embedding_dim: int = 128) -> None:
        super().__init__()
        width = 64
        self.input_projection = nn.Linear(channels, width)
        self.position = nn.Parameter(torch.zeros(1, length, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=192, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, embedding_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(sequence) + self.position)
        return self.projection(self.norm(encoded).mean(dim=1))


class SiamesePairModel(nn.Module):
    def __init__(self, architecture: str, channels: int, length: int) -> None:
        super().__init__()
        if architecture == "cnn":
            self.encoder = CNNEncoder(channels)
        elif architecture == "bilstm":
            self.encoder = BiLSTMEncoder(channels)
        elif architecture == "transformer":
            self.encoder = TransformerEncoder(channels, length)
        else:
            raise ValueError(f"Unknown deep architecture: {architecture}")
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_embedding = self.encoder(first)
        second_embedding = self.encoder(second)
        symmetric = torch.cat([
            torch.abs(first_embedding - second_embedding), first_embedding * second_embedding,
        ], dim=1)
        return self.head(symmetric).squeeze(1)


@dataclass(frozen=True)
class DeepConfig:
    length: int = 128
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 12
    patience: int = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DeepPairScorer:
    def __init__(
        self, sequence_store: SequenceStore, model: SiamesePairModel,
        device: torch.device, batch_size: int,
    ) -> None:
        self.sequence_store = sequence_store
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        dummy = np.zeros(len(pairs), dtype=np.float32)
        loader = DataLoader(
            PairDataset(self.sequence_store, pairs, dummy),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
        values: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for first, second, _ in loader:
                logits = self.model(
                    first.to(self.device, non_blocking=True),
                    second.to(self.device, non_blocking=True),
                )
                values.append(logits.float().cpu().numpy())
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def fit_deep_pair_model(
    architecture: str,
    sequence_store: SequenceStore,
    train_pairs: list[tuple[str, str]],
    train_labels: np.ndarray,
    validation_pairs: list[tuple[str, str]],
    validation_labels: np.ndarray,
    seed: int,
    checkpoint_path: str | Path,
    config: DeepConfig | None = None,
) -> tuple[DeepPairScorer, dict[str, Any]]:
    config = config or DeepConfig(length=sequence_store.length)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamesePairModel(architecture, sequence_store.channels, sequence_store.length).to(device)
    dataset = PairDataset(sequence_store, train_pairs, train_labels)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=True,
    )
    positives = max(float(train_labels.sum()), 1.0)
    negatives = max(float((train_labels == 0).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scorer = DeepPairScorer(sequence_store, model, device, config.batch_size)
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
            logits = model(first, second)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)
        validation_scores = scorer.score_pairs(validation_pairs)
        validation_auc = float(roc_auc_score(validation_labels, validation_scores))
        epoch_loss = total_loss / max(total_examples, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "validation_auc": validation_auc})
        print(
            f"{architecture} epoch={epoch} loss={epoch_loss:.6f} val_auc={validation_auc:.6f}",
            flush=True,
        )
        if validation_auc > best_auc + 1e-5:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Deep training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": architecture,
        "state_dict": best_state,
        "config": asdict(config),
        "seed": seed,
        "validation_auc": best_auc,
    }, checkpoint_path)
    elapsed = time.perf_counter() - started
    metadata = {
        "architecture": architecture,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_pairs": len(train_pairs),
        "positive_pairs": int(train_labels.sum()),
        "negative_pairs": int((train_labels == 0).sum()),
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "fit_seconds": elapsed,
        "config": asdict(config),
        "history": history,
    }
    return scorer, metadata
