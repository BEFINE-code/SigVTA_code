from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .model import ConditionalABCT2Head, SignatureEncoder, T1Head, _gather_set


TaskName = Literal["t1", "t2"]
PACKAGE_SCHEMA = "dvsrc.task-inference.v1"


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    allowed = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in raw.items() if key in allowed})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TaskInferenceModel(nn.Module):
    """One deployed encoder instance plus the head required by one task."""

    def __init__(self, config: ModelConfig, task: TaskName):
        super().__init__()
        if task not in {"t1", "t2"}:
            raise ValueError(f"Unsupported task package: {task}")
        if task == "t2" and config.variant != "t2_abc_v1":
            raise ValueError(
                "T2 task packages require the dual-weight conditional ABC variant t2_abc_v1"
            )
        self.config = config
        self.task = task
        self.encoder = SignatureEncoder(config)
        self.head = (
            T1Head(config, config.t1_variant or config.variant)
            if task == "t1" else ConditionalABCT2Head(config)
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        protocol = str(batch["protocol"])
        if self.task == "t1" and not protocol.startswith("t1"):
            raise ValueError(f"T1 package cannot execute protocol {protocol}")
        if self.task == "t2" and protocol.startswith("t1"):
            raise ValueError(f"T2 package cannot execute protocol {protocol}")

        encoding = self.encoder(
            batch["sequence"], batch["sequence_mask"], batch["image"],
            batch["anchors"], batch["anchor_mask"],
        )
        query = encoding.select(batch["query_index"])
        if self.task == "t2" and protocol == "t2_a_query":
            assert isinstance(self.head, ConditionalABCT2Head)
            return self.head.forward_a(query)

        members = _gather_set(encoding, batch["set_index"])
        if self.task == "t1":
            assert isinstance(self.head, T1Head)
            return self.head(members, query, batch["set_mask"])
        assert isinstance(self.head, ConditionalABCT2Head)
        return self.head(
            members, query, batch["set_mask"],
            batch.get("target_index"), batch.get("episode_type_index"),
        )


def export_task_package(
    checkpoint_path: str | Path,
    task: TaskName,
    output_path: str | Path,
    calibration_path: str | Path | None = None,
    initialized_from_t1: str | Path | None = None,
) -> dict[str, Any]:
    """Export a training checkpoint into a strict, single-task inference package."""
    source_path = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config", {}).get("model")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint does not contain config.model metadata")
    config = _model_config(raw_config)
    if task == "t2" and config.variant != "t2_abc_v1":
        raise ValueError(f"T2 export requires t2_abc_v1, found {config.variant}")

    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    encoder_prefix = "encoder." if task == "t1" else "t2_encoder."
    head_prefix = "t1." if task == "t1" else "t2."
    package_state: dict[str, Tensor] = {}
    for key, value in source_state.items():
        if key.startswith(encoder_prefix):
            package_state["encoder." + key.removeprefix(encoder_prefix)] = value
        elif key.startswith(head_prefix):
            package_state["head." + key.removeprefix(head_prefix)] = value

    model = TaskInferenceModel(config, task)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(package_state))
    unexpected = sorted(set(package_state) - set(expected))
    incompatible = sorted(
        key for key in set(expected) & set(package_state)
        if expected[key].shape != package_state[key].shape
    )
    if missing or unexpected or incompatible:
        raise ValueError(
            "Task package state is incomplete or incompatible: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"incompatible={incompatible[:5]}"
        )
    model.load_state_dict(package_state, strict=True)

    calibration = None
    calibration_source = None
    if calibration_path is not None:
        calibration_file = Path(calibration_path).resolve()
        calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
        calibration_source = {
            "path": str(calibration_file),
            "sha256": _sha256(calibration_file),
        }

    parent = None
    if initialized_from_t1 is not None:
        parent_path = Path(initialized_from_t1).resolve()
        parent = {"path": str(parent_path), "sha256": _sha256(parent_path)}

    payload = {
        "schema": PACKAGE_SCHEMA,
        "task": task,
        "model_config": raw_config,
        "model": package_state,
        "calibration": calibration,
        "provenance": {
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": _sha256(source_path),
            "source_stage": checkpoint.get("stage"),
            "source_epoch": checkpoint.get("epoch"),
            "initialized_from_t1": parent,
            "calibration_source": calibration_source,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "schema": PACKAGE_SCHEMA,
        "task": task,
        "output": str(output),
        "output_sha256": _sha256(output),
        "parameter_tensors": len(package_state),
        "parameters": sum(tensor.numel() for tensor in package_state.values()),
        "source_checkpoint": str(source_path),
        "initialized_from_t1": parent,
    }


def load_task_package(
    package_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TaskInferenceModel, dict[str, Any]]:
    """Load one task package without constructing the unused task branch."""
    path = Path(package_path).resolve()
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("schema") != PACKAGE_SCHEMA:
        raise ValueError(f"Unsupported task package schema: {package.get('schema')}")
    task = package.get("task")
    if task not in {"t1", "t2"}:
        raise ValueError(f"Unsupported task package: {task}")
    raw_config = package.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Task package does not contain model_config")
    state = package.get("model")
    if not isinstance(state, dict):
        raise ValueError("Task package does not contain model weights")

    model = TaskInferenceModel(_model_config(raw_config), task)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {
        "schema": package["schema"],
        "task": task,
        "calibration": package.get("calibration"),
        "provenance": package.get("provenance", {}),
        "package_path": str(path),
        "package_sha256": _sha256(path),
    }
    return model, metadata
