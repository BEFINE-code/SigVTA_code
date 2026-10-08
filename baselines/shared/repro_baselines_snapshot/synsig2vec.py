from __future__ import annotations

import copy
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import SignatureStore


def _normalize_official(raw: np.ndarray) -> np.ndarray:
    result = np.asarray(raw[:, [1, 2, 3]], dtype=np.float32).copy()
    minimum = result[:, :2].min(axis=0)
    maximum = result[:, :2].max(axis=0)
    result[:, :2] = (result[:, :2] - (maximum + minimum) / 2.0) / max(
        float((maximum - minimum).max()), 1e-8,
    )
    result[:, 2] /= max(float(result[:, 2].max()), 1e-8)
    return result


def load_official_modules(official_root: str | Path) -> dict[str, Any]:
    official_root = Path(official_root).resolve()
    sigma_root = str(official_root / "sigma_lognormal")
    model_root = str(official_root / "Sig2Vec")
    for path in (sigma_root, model_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    import scipy
    from scipy import signal

    if not hasattr(scipy, "convolve"):
        scipy.convolve = signal.convolve  # type: ignore[attr-defined]
    from dataset import datasetTrainAll_SLN as train_dataset  # type: ignore[import-not-found]
    from dataset import utils as dataset_utils  # type: ignore[import-not-found]
    from network import Sig2Vec  # type: ignore[import-not-found]
    return {"dataset": train_dataset, "utils": dataset_utils, "Sig2Vec": Sig2Vec}


class OfficialSequenceStore:
    def __init__(self, signatures: SignatureStore, feature_function) -> None:
        self.signatures = signatures
        self.feature_function = feature_function
        self.cache: dict[str, np.ndarray] = {}
        self.repaired_nonfinite: set[str] = set()

    def get(self, sample_id: str) -> np.ndarray:
        if sample_id not in self.cache:
            path = _normalize_official(self.signatures.load(sample_id))
            features: list[np.ndarray] = []
            self.feature_function([path], features)
            value = features[0]
            if not np.isfinite(value).all():
                value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                self.repaired_nonfinite.add(sample_id)
            self.cache[sample_id] = value
        return self.cache[sample_id]


class OfficialSynSig2VecScorer:
    def __init__(
        self, store: OfficialSequenceStore, model: nn.Module,
        device: torch.device, batch_size: int = 128,
    ) -> None:
        self.store = store
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.embedding_cache: dict[str, np.ndarray] = {}

    def clear_embeddings(self) -> None:
        self.embedding_cache.clear()

    def _encode(self, sample_ids: list[str]) -> None:
        missing = sorted(set(sample_ids) - self.embedding_cache.keys())
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                batch_ids = missing[start:start + self.batch_size]
                sequences = [self.store.get(sample_id) for sample_id in batch_ids]
                lengths = np.asarray([len(value) for value in sequences], dtype=np.float32)
                padded = np.zeros(
                    (len(sequences), int(lengths.max()), sequences[0].shape[1]), dtype=np.float32,
                )
                for index, sequence in enumerate(sequences):
                    padded[index, :len(sequence)] = sequence
                mask = self.model.getOutputMask(lengths)
                embeddings, _ = self.model(
                    torch.from_numpy(padded).to(self.device),
                    torch.from_numpy(mask).to(self.device),
                    torch.from_numpy(lengths).to(self.device),
                )
                values = embeddings.float().cpu().numpy().reshape(len(sequences), -1, 32)
                values /= np.linalg.norm(values, axis=2, keepdims=True).clip(min=1e-8)
                values = values.reshape(len(sequences), -1)
                values /= np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)
                self.embedding_cache.update(zip(batch_ids, values, strict=True))

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        sample_ids = [sample_id for pair in pairs for sample_id in pair]
        self._encode(sample_ids)
        return np.asarray([
            -float(np.square(self.embedding_cache[first] - self.embedding_cache[second]).sum())
            for first, second in pairs
        ])


@dataclass(frozen=True)
class SynSig2VecConfig:
    epochs: int = 200
    validation_interval: int = 25
    patience_intervals: int = 3
    learning_rate: float = 1e-3
    task_size: int = 4
    genuine_shots: int = 5
    forgery_shots: int = 10
    synthesis_per_signature: int = 10
    synthesis_level: float = 0.4
    inference_batch_size: int = 128


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_official_synsig2vec(
    signatures: SignatureStore,
    validation_pairs: list[tuple[str, str]],
    validation_labels: np.ndarray,
    official_root: str | Path,
    prepared_root: str | Path,
    checkpoint_path: str | Path,
    seed: int,
    config: SynSig2VecConfig | None = None,
) -> tuple[OfficialSynSig2VecScorer, dict[str, Any]]:
    config = config or SynSig2VecConfig()
    _set_seed(seed)
    official_root = Path(official_root).resolve()
    prepared_root = Path(prepared_root).resolve()
    manifest_path = prepared_root / "manifest.json"
    if not manifest_path.is_file() or not (prepared_root / "train_genuine.pkl").is_file():
        raise FileNotFoundError("Complete official SynSig2Vec preprocessing is required")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("Official SynSig2Vec preprocessing manifest is incomplete")
    modules = load_official_modules(official_root)
    previous_cwd = Path.cwd()
    try:
        os.chdir(official_root / "Sig2Vec")
        with (prepared_root / "train_genuine.pkl").open("rb") as stream:
            signature_dict = pickle.load(stream)
        dataset = modules["dataset"].dataset(
            sigDict=signature_dict,
            slnPath=str(prepared_root / "params"),
            slnLevel=config.synthesis_level,
            taskSize=config.task_size,
            taskNumGen=config.genuine_shots,
            taskNumNeg=config.forgery_shots,
            numSynthesis=config.synthesis_per_signature,
            prefix="PaperDatabaseTrain",
        )
        dataset.computeStats()
    finally:
        os.chdir(previous_cwd)
    training_nonfinite_sequences = 0
    for index, feature in enumerate(dataset.feats):
        if not np.isfinite(feature).all():
            dataset.feats[index] = np.nan_to_num(
                feature, nan=0.0, posinf=0.0, neginf=0.0,
            )
            training_nonfinite_sequences += 1
    sampler = modules["dataset"].batchSampler(dataset, loop=False)
    loader = DataLoader(
        dataset, num_workers=0, batch_sampler=sampler,
        collate_fn=modules["dataset"].collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modules["Sig2Vec"](
        n_in=dataset.featDim,
        n_classes=len(dataset),
        n_task=config.task_size,
        n_shot_g=config.genuine_shots,
        n_shot_f=config.forgery_shots,
        APAlpha=6.0,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=0.9,
        weight_decay=1e-5, nesterov=True,
    )
    sequence_store = OfficialSequenceStore(signatures, modules["utils"].featExt)
    scorer = OfficialSynSig2VecScorer(
        sequence_store, model, device, config.inference_batch_size,
    )
    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        map_sum = 0.0
        batches = 0
        for signatures_batch, lengths, labels in loader:
            mask = model.getOutputMask(lengths)
            values = torch.from_numpy(signatures_batch).to(device)
            masks = torch.from_numpy(mask).to(device)
            length_values = torch.from_numpy(lengths).to(device)
            label_values = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(values, masks, length_values)
            loss_ce = model.smoothCEloss(logits, label_values, eps=0.10)
            loss_ap, mean_ap = model.APLoss_DLM(embeddings)
            (loss_ap + loss_ce).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss_ap.item())
            ce_sum += float(loss_ce.item())
            map_sum += float(mean_ap.item())
            batches += 1
        row = {
            "epoch": epoch,
            "ap_loss": loss_sum / max(batches, 1),
            "classification_loss": ce_sum / max(batches, 1),
            "train_map": map_sum / max(batches, 1),
        }
        should_validate = epoch == 1 or epoch % config.validation_interval == 0
        if should_validate:
            scorer.clear_embeddings()
            validation_auc = float(roc_auc_score(
                validation_labels, scorer.score_pairs(validation_pairs),
            ))
            row["validation_auc"] = validation_auc
            print(
                f"synsig2vec epoch={epoch} ap={row['ap_loss']:.6f} "
                f"ce={row['classification_loss']:.6f} map={row['train_map']:.6f} "
                f"val_auc={validation_auc:.6f}", flush=True,
            )
            if validation_auc > best_auc + 1e-5:
                best_auc = validation_auc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= config.validation_interval * config.patience_intervals:
                history.append(row)
                break
        history.append(row)
    if best_state is None:
        raise RuntimeError("Official SynSig2Vec training did not produce a checkpoint")
    model.load_state_dict(best_state)
    scorer.clear_embeddings()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "official_synsig2vec_adapted",
        "official_commit": manifest["official_commit"],
        "state_dict": best_state,
        "config": asdict(config),
        "seed": seed,
        "validation_auc": best_auc,
    }, checkpoint_path)
    return scorer, {
        "architecture": "official_synsig2vec_adapted",
        "official_commit": manifest["official_commit"],
        "official_license": manifest["official_license"],
        "prepared_split_digest": manifest.get("split_digest"),
        "prepared_cache_scope": "Train-only; reusable only when the frozen Train writer set is identical",
        "adaptation": (
            "official sigma-lognormal synthesis, official selective-pooling 1D CNN and "
            "AP/classification objective; frozen benchmark validation selects checkpoint"
        ),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_writers": len(dataset),
        "genuine_signatures": manifest["genuine_signatures"],
        "training_nonfinite_sequences_repaired": training_nonfinite_sequences,
        "validation_nonfinite_sequences_repaired": len(sequence_store.repaired_nonfinite),
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "fit_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "history": history,
    }
