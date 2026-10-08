from __future__ import annotations

import itertools
import gc
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from .calibration import Calibrator, eer_threshold, fit_binary_temperature, fit_multiclass_temperature
from .config import ExperimentConfig, ModelConfig
from .data import (
    EpisodeCollator, EpisodeDataset, make_preprocessor,
    t2_balanced_a_query_view, t2_balanced_b_view,
)
from .losses import T1Loss, T2Loss, project_conflicting
from .metrics import (
    balanced_a_metrics, best_balanced_a_threshold, conditional_t2_factor_metrics,
    t1_metrics, t2_metrics, writer_bootstrap,
)
from .model import DVSRNet, SignatureEncoder
from .utils import atomic_json, seed_everything, write_jsonl


class BalancedEpisodeBatchSampler(Sampler[list[int]]):
    """Oversample only within predeclared episode strata, never across splits."""

    def __init__(self, dataset: EpisodeDataset, batch_size: int, seed: int = 42, drop_last: bool = False,
                  t2_unknown_fraction: float = 0.5, pair_source_absent: bool = False):
        self.batch_size, self.seed, self.drop_last, self.epoch = batch_size, seed, drop_last, 0
        self.cycle = 0
        if not 0.0 <= t2_unknown_fraction <= 1.0:
            raise ValueError("t2_unknown_fraction must be between 0 and 1")
        self.t2_unknown_fraction = t2_unknown_fraction
        self.pair_source_absent = pair_source_absent
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.episodes):
            if row["protocol"].startswith("t1"):
                key = "genuine" if row["label"] else row["attack_type"]
            else:
                key = row["episode_type"]
            groups[key].append(index)
        self.groups = dict(groups)
        self.protocol = dataset.episodes[0]["protocol"] if dataset.episodes else ""
        self.present_by_query: dict[str, list[int]] = defaultdict(list)
        if self.protocol == "t2_source_ranking" and pair_source_absent:
            for index, row in enumerate(dataset.episodes):
                if row["episode_type"] == "source_present":
                    key = row.get("counterfactual_pair_id", row["query_id"])
                    self.present_by_query[key].append(index)
            self.query_by_index = {
                index: row.get("counterfactual_pair_id", row["query_id"])
                for index, row in enumerate(dataset.episodes)
            }
        else:
            self.query_by_index = {}
        self.batches = math.ceil(len(dataset) / batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.cycle = 0

    def set_cycle(self, cycle: int) -> None:
        self.cycle = cycle

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009 + self.cycle * 9176)
        names = sorted(self.groups)
        pools = {name: rng.sample(values, len(values)) for name, values in self.groups.items()}
        cursors = {name: 0 for name in names}
        unknown_names = [name for name in names if name != "source_present"]
        unknown_cursor = 0
        paired_pools = {
            query_id: rng.sample(indices, len(indices))
            for query_id, indices in self.present_by_query.items()
        }
        paired_cursors = defaultdict(int)

        def take(name: str) -> int:
            if cursors[name] >= len(pools[name]):
                pools[name] = rng.sample(self.groups[name], len(self.groups[name]))
                cursors[name] = 0
            index = pools[name][cursors[name]]
            cursors[name] += 1
            return index

        def take_paired_present(query_id: str) -> int:
            candidates = paired_pools.get(query_id)
            if not candidates:
                return take("source_present")
            cursor = paired_cursors[query_id]
            if cursor >= len(candidates):
                candidates = rng.sample(self.present_by_query[query_id], len(self.present_by_query[query_id]))
                paired_pools[query_id] = candidates
                cursor = 0
            paired_cursors[query_id] = cursor + 1
            return candidates[cursor]

        for batch_index in range(self.batches):
            if self.protocol == "t2_source_ranking" and self.pair_source_absent:
                start = batch_index * self.batch_size
                stop = start + self.batch_size
                unknown_count = math.floor(stop * self.t2_unknown_fraction) - math.floor(
                    start * self.t2_unknown_fraction
                )
                unknown_indices = []
                for _ in range(unknown_count):
                    name = unknown_names[unknown_cursor % len(unknown_names)]
                    unknown_cursor += 1
                    unknown_indices.append((name, take(name)))
                present_count = self.batch_size - unknown_count
                batch = []
                for name, unknown_index in unknown_indices:
                    batch.append(unknown_index)
                    if name == "source_absent" and present_count > 0:
                        batch.append(take_paired_present(self.query_by_index[unknown_index]))
                        present_count -= 1
                batch.extend(take("source_present") for _ in range(present_count))
                rng.shuffle(batch)
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
                continue
            batch = []
            for offset in range(self.batch_size):
                sample_index = batch_index * self.batch_size + offset
                if self.protocol == "t2_source_ranking":
                    if "source_present" not in self.groups or not unknown_names:
                        raise ValueError("T2 training requires source-present and Unknown episode strata")
                    unknown_before = math.floor(sample_index * self.t2_unknown_fraction)
                    unknown_after = math.floor((sample_index + 1) * self.t2_unknown_fraction)
                    if unknown_after > unknown_before:
                        name = unknown_names[unknown_cursor % len(unknown_names)]
                        unknown_cursor += 1
                    else:
                        name = "source_present"
                else:
                    name = names[sample_index % len(names)]
                batch.append(take(name))
            rng.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _cycle(loader: DataLoader, reshuffle: bool = False) -> Iterator[dict[str, Any]]:
    cycle = 0
    while True:
        if reshuffle and hasattr(loader.batch_sampler, "set_cycle"):
            loader.batch_sampler.set_cycle(cycle)
        yield from loader
        cycle += 1


def deterministic_stratified_subset(
    episodes: list[dict[str, Any]], fraction: float, seed: int,
) -> list[dict[str, Any]]:
    """Reduce only a training split while retaining every declared episode stratum."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("train_episode_fraction must be in (0, 1]")
    if fraction == 1.0 or not episodes:
        return list(episodes)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        if row["protocol"].startswith("t1"):
            key = "genuine" if row["label"] else row["attack_type"]
        else:
            key = row["episode_type"]
        groups[key].append(row)
    selected_ids: set[str] = set()
    for name, rows in groups.items():
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{name}:{row['episode_id']}".encode("utf-8")
            ).digest(),
        )
        keep = max(1, int(round(len(ranked) * fraction)))
        selected_ids.update(row["episode_id"] for row in ranked[:keep])
    return [row for row in episodes if row["episode_id"] in selected_ids]


def counterfactual_t2_pairs(
    episodes: list[dict[str, Any]], seed: int,
) -> list[dict[str, Any]]:
    """Make one source-present/source-absent pair differ by only the source candidate."""
    rows = [{**row, "candidate_ids": list(row.get("candidate_ids", []))} for row in episodes]
    present_by_query: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("episode_type") == "source_present":
            present_by_query[row["query_id"]].append(index)
    used_present: set[int] = set()
    for absent_index, absent in enumerate(rows):
        if absent.get("episode_type") != "source_absent":
            continue
        choices = [index for index in present_by_query.get(absent["query_id"], []) if index not in used_present]
        if not choices:
            continue
        choices.sort(key=lambda index: hashlib.sha256(
            f"{seed}:{absent['episode_id']}:{rows[index]['episode_id']}".encode("utf-8")
        ).digest())
        present_index = choices[0]
        present = rows[present_index]
        target = int(present["target_index"])
        shared = list(present["candidate_ids"])
        replacement_pool = sorted(set(absent["candidate_ids"]) - set(shared))
        if not replacement_pool:
            continue
        replacement = replacement_pool[
            int.from_bytes(hashlib.sha256(
                f"{seed}:{absent['episode_id']}:replacement".encode("utf-8")
            ).digest()[:8], "big") % len(replacement_pool)
        ]
        counterfactual = list(shared)
        counterfactual[target] = replacement
        pair_id = f"CF-{present['episode_id']}-{absent['episode_id']}"
        present.update({"counterfactual_pair_id": pair_id, "counterfactual_role": "present"})
        absent.update({
            "candidate_ids": counterfactual,
            "counterfactual_pair_id": pair_id,
            "counterfactual_role": "absent",
            "counterfactual_replaced_index": target,
        })
        used_present.add(present_index)
    return rows


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        seed_everything(config.train.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DVSRNet(config.model).to(self.device)
        self.t1_loss = T1Loss(genuine_identity_only=config.train.t1_genuine_identity_only)
        self.t2_loss = T2Loss(
            mode=config.train.t2_loss,
            pair_margin=config.train.t2_pair_margin,
            pair_weight=config.train.t2_pair_loss_weight,
            query_weight=config.train.t2_query_loss_weight,
            monotonic_weight=config.train.t2_monotonic_loss_weight,
            stateful_weight=config.train.t2_stateful_loss_weight,
            arrival_weight=config.train.t2_arrival_loss_weight,
            bridge_weight=(
                config.train.unified_bridge_loss_weight
                if config.model.variant == "unified_abc_v11" else 0.0
            ),
            official_weight=(
                config.train.unified_official_loss_weight
                if config.model.variant in {
                    "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                    "unified_sen_v20",
                } else 0.0
            ),
            joint_weight=config.train.unified_joint_loss_weight,
            factor_a_weight=config.train.unified_factor_a_loss_weight,
            factor_b_weight=config.train.unified_factor_b_loss_weight,
            rank_weight=config.train.unified_rank_loss_weight,
            rank_margin_weight=config.train.unified_rank_margin_loss_weight,
            rank_distillation_weight=config.train.t2_rank_distillation_weight,
            b_distillation_weight=config.train.t2_b_distillation_weight,
            hard_negative_weight=config.train.t2_hard_negative_loss_weight,
            distillation_temperature=config.train.t2_distillation_temperature,
        )
        self.output = Path(config.train.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_json(self.output / "resolved_config.json", config.to_dict())
        self.store, preprocessor = make_preprocessor(config.data)
        self.collator = EpisodeCollator(preprocessor)
        self.fold_dir = Path(config.data.benchmark_root) / f"episodes/fold_{config.data.fold}"
        self.t2_fold_dir = Path(
            config.data.t2_benchmark_root or config.data.benchmark_root
        ) / f"episodes/fold_{config.data.fold}"
        self.optimizer = AdamW(
            self.model.parameter_groups(config.train.backbone_lr, config.train.encoder_lr, config.train.head_lr),
            weight_decay=config.train.weight_decay,
            fused=self.device.type == "cuda",
        )
        self.history: list[dict[str, Any]] = []
        self.test_history: list[dict[str, Any]] = []
        self.held_out_forger: str | None = None
        self.t2_relation_teacher: DVSRNet | None = None

    def _dataset(self, split: str, protocol: str, held_out_forger: str | None = None) -> EpisodeDataset:
        filenames = {
            "t1_1v1": f"{split}_t1_1v1.jsonl",
            "t1_5v1": f"{split}_t1_5v1.jsonl",
            "t2": f"{split}_t2.jsonl",
            "t2_a": f"{split}_t2.jsonl",
            "t2_b": f"{split}_t2.jsonl",
            "t2_c": f"{split}_t2.jsonl",
        }
        fold_dir = self.t2_fold_dir if protocol.startswith("t2") else self.fold_dir
        dataset = EpisodeDataset(fold_dir / filenames[protocol])
        held_out_forger = held_out_forger or self.held_out_forger
        if held_out_forger and split in {"train", "val"}:
            dataset.episodes = [row for row in dataset.episodes if row.get("forger_id") != held_out_forger]
        view_seed = self.config.train.seed + {"train": 1103, "val": 1709, "test": 2309}[split]
        if protocol == "t2_a":
            dataset.episodes = t2_balanced_a_query_view(
                dataset.episodes, view_seed, balance=split != "train",
            )
        elif protocol == "t2_b":
            dataset.episodes = t2_balanced_b_view(dataset.episodes, view_seed)
        elif protocol == "t2_c":
            dataset.episodes = [
                row for row in dataset.episodes if row["episode_type"] == "source_present"
            ]
        if split == "train":
            if protocol != "t2_a":
                dataset.episodes = deterministic_stratified_subset(
                    dataset.episodes,
                    self.config.data.train_episode_fraction,
                    self.config.data.train_subset_seed,
                )
            if protocol == "t2" and self.config.train.t2_counterfactual_pairs:
                dataset.episodes = counterfactual_t2_pairs(
                    dataset.episodes, self.config.train.seed,
                )
        return dataset

    def _stage_b_protocols(self) -> tuple[str, ...]:
        return (("t1_1v1", "t1_5v1", "t2")
                if self.config.train.stage_b_train_t1 else ("t2",))

    def loader(self, split: str, protocol: str, train: bool = False, held_out_forger: str | None = None,
               batch_size: int | None = None, num_workers: int | None = None) -> DataLoader:
        dataset = self._dataset(split, protocol, held_out_forger)
        batch_sizes = {
            "t1_1v1": self.config.train.batch_t1_1v1,
            "t1_5v1": self.config.train.batch_t1_5v1,
            "t2": self.config.train.batch_t2,
            "t2_a": self.config.train.batch_t2_a,
            "t2_b": self.config.train.batch_t2_b,
            "t2_c": self.config.train.batch_t2,
        }
        if batch_size is not None:
            batch_sizes[protocol] = batch_size
        elif not train:
            batch_sizes[protocol] *= self.config.train.eval_batch_multiplier
        workers = self.config.data.num_workers if num_workers is None else num_workers
        common = {"num_workers": workers, "collate_fn": self.collator, "pin_memory": self.device.type == "cuda",
                  "persistent_workers": workers > 0}
        if workers > 0:
            common["prefetch_factor"] = 4
        if train:
            if protocol.startswith("t1") and not self.config.train.t1_balance_strata:
                generator = torch.Generator()
                generator.manual_seed(self.config.train.seed)
                return DataLoader(
                    dataset, batch_size=batch_sizes[protocol], shuffle=True,
                    generator=generator, drop_last=False, **common,
                )
            sampler = BalancedEpisodeBatchSampler(
                dataset, batch_sizes[protocol], self.config.train.seed,
                t2_unknown_fraction=self.config.train.t2_unknown_fraction,
                pair_source_absent=self.config.train.t2_pair_source_absent,
            )
            return DataLoader(dataset, batch_sampler=sampler, **common)
        return DataLoader(dataset, batch_size=batch_sizes[protocol], shuffle=False, **common)

    def manifest_loader(self, path: str | Path, batch_size: int | None = None) -> DataLoader:
        workers = self.config.data.num_workers
        options = {
            "num_workers": workers, "collate_fn": self.collator,
            "pin_memory": self.device.type == "cuda", "persistent_workers": workers > 0,
        }
        if workers > 0:
            options["prefetch_factor"] = 4
        return DataLoader(
            EpisodeDataset(path), batch_size=batch_size or self.config.train.batch_t2,
            shuffle=False, **options,
        )

    def _autocast(self):
        enabled = self.config.train.amp and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled)

    @staticmethod
    def _require_finite(value: Tensor, context: str) -> None:
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite tensor detected during {context}")

    def _finite_optimizer_step(self, parameters: Iterable[Tensor], context: str) -> None:
        active = [parameter for parameter in parameters if parameter.requires_grad]
        for parameter in active:
            self._require_finite(parameter, f"{context} parameters before update")
            if parameter.grad is not None:
                self._require_finite(parameter.grad, f"{context} gradients")
        norm = torch.nn.utils.clip_grad_norm_(active, self.config.train.grad_clip)
        self._require_finite(norm, f"{context} gradient norm")
        self.optimizer.step()
        for parameter in active:
            self._require_finite(parameter, f"{context} parameters after update")

    def _freeze_early_image(self, frozen: bool) -> None:
        for index, layer in enumerate(self.model.encoder.image.features):
            if index <= 3:
                for parameter in layer.parameters():
                    parameter.requires_grad_(not frozen)

    def _prepare_stage_b_modules(self, stage_epoch: int | None = None) -> None:
        if self.config.train.stage_b_strict_t1_isolation and (
            not self.config.train.stage_b_freeze_encoder
            or self.config.train.stage_b_unfreeze_late_encoder
            or self.config.train.stage_b_train_t1
        ):
            raise ValueError(
                "Strict T1 isolation requires a frozen encoder, frozen T1 head, and no late unfreeze"
            )
        encoder_frozen = self.config.train.stage_b_freeze_encoder
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(not encoder_frozen)
        if encoder_frozen and self.config.train.stage_b_unfreeze_late_encoder:
            for module in self._late_stage_b_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for parameter in self.model.t1.parameters():
            parameter.requires_grad_(self.config.train.stage_b_train_t1)
        for parameter in self.model.t2.parameters():
            parameter.requires_grad_(True)
        if self.config.model.variant == "unified_abc_v14":
            phase = self._stage_b_phase(stage_epoch or 0)
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.model.t1.parameters():
                parameter.requires_grad_(False)
            assert self.model.t2_adapter is not None
            if self.config.train.t2_scratch_from_stage_a:
                for parameter in self.model.t2.parameters():
                    parameter.requires_grad_(True)
                for parameter in self.model.t2_adapter.parameters():
                    parameter.requires_grad_(phase == "joint")
                if self.model.t2_energy_adapter is not None:
                    for parameter in self.model.t2_energy_adapter.parameters():
                        parameter.requires_grad_(True)
                self.t2_loss.set_training_phase("joint")
            else:
                for parameter in self.model.t2.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.model.t2_adapter.parameters():
                    parameter.requires_grad_(phase == "rank_recovery")
                active_t2_module = (
                    self.model.t2.relation
                    if phase == "rank_recovery" else self.model.t2.in_set_head
                )
                for parameter in active_t2_module.parameters():
                    parameter.requires_grad_(True)
                if self.model.t2_energy_adapter is not None:
                    for parameter in self.model.t2_energy_adapter.parameters():
                        parameter.requires_grad_(phase == "b_calibration")
                self.t2_loss.set_training_phase(phase)
        elif self.config.model.variant in {"unified_abc_v12", "unified_abc_v13"}:
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.model.t1.parameters():
                parameter.requires_grad_(False)
            assert self.model.t2_adapter is not None
            for parameter in self.model.t2_adapter.parameters():
                parameter.requires_grad_(True)
        elif self.config.model.variant == "unified_abc_v11":
            phase = self._stage_b_phase(stage_epoch or 0)
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.model.t1.parameters():
                parameter.requires_grad_(phase == "joint")
            if phase == "joint":
                for module in self._late_stage_b_modules():
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
        elif self.config.model.variant == "t2_abc_v1":
            phase = self._stage_b_phase(stage_epoch or 0)
            assert self.model.t2_encoder is not None
            trainability = self.config.train.t2_encoder_trainability
            if trainability not in {"full", "head_only", "late_partial"}:
                raise ValueError(
                    "t2_encoder_trainability must be full, head_only, or late_partial"
                )
            for parameter in self.model.t2_encoder.parameters():
                parameter.requires_grad_(trainability == "full" and phase == "joint")
            if trainability == "late_partial" and phase == "joint":
                for module in self._late_t2_encoder_modules():
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
            inactive_modules: tuple[torch.nn.Module, ...] = ()
            if (
                self.config.train.t2_freeze_inactive_modalities
                and self.config.model.fusion == "sequence_only"
            ):
                inactive_modules = (
                    self.model.t2_encoder.image,
                    self.model.t2_encoder.fusion,
                    self.model.t2_encoder.late,
                )
            elif (
                self.config.train.t2_freeze_inactive_modalities
                and self.config.model.fusion == "image_only"
            ):
                inactive_modules = (
                    self.model.t2_encoder.sequence,
                    self.model.t2_encoder.fusion,
                    self.model.t2_encoder.late,
                )
            elif self.config.train.t2_freeze_inactive_modalities:
                inactive_modules = (
                    (self.model.t2_encoder.late,)
                    if self.config.model.fusion == "ms_caf"
                    else (self.model.t2_encoder.fusion,)
                    if self.config.model.fusion == "late"
                    else ()
                )
            for module in inactive_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            if self.config.train.t2_freeze_inactive_relation_modules:
                relation = self.model.t2.relation
                disabled_relation_modules: tuple[torch.nn.Module, ...] = (
                    (relation.simple_rank, relation.simple_exist)
                    if self.config.model.use_ocsr else (relation.ocsr,)
                )
                if not self.config.model.use_pim:
                    disabled_relation_modules = (*disabled_relation_modules, relation.pim)
                for module in disabled_relation_modules:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        elif self.config.model.variant == "v10":
            phase = self._stage_b_phase(stage_epoch or 0)
            assert self.model.t2_rank_encoder is not None
            assert self.model.t2_open_encoder is not None
            for parameter in self.model.t2_rank_encoder.parameters():
                parameter.requires_grad_(phase == "rank")
            for parameter in self.model.t2_open_encoder.parameters():
                parameter.requires_grad_(phase == "open")
            self.model.t2.set_training_phase(phase)
        elif self.config.model.variant == "v9":
            phase = self._stage_b_phase(stage_epoch or 0)
            assert self.model.t2_encoder is not None
            for parameter in self.model.t2_encoder.parameters():
                parameter.requires_grad_(phase != "calibration")
            self.model.t2.set_training_phase(phase)
        elif self.config.model.variant in {"v7", "v8"}:
            rank_epochs = self.config.train.t2_rank_phase_epochs
            phase = "rank" if stage_epoch is None or stage_epoch < rank_epochs else "open"
            self.model.t2.set_training_phase(phase)

    def _stage_b_phase(self, stage_epoch: int) -> str:
        if self.config.model.variant == "unified_abc_v14":
            if self.config.train.t2_scratch_from_stage_a:
                return (
                    "head_warmup"
                    if stage_epoch < self.config.train.t2_head_warmup_epochs else "joint"
                )
            return (
                "rank_recovery"
                if stage_epoch < self.config.train.t2_rank_recovery_epochs
                else "b_calibration"
            )
        if self.config.model.variant == "unified_abc_v13":
            return (
                "relation_distillation"
                if stage_epoch < self.config.train.t2_relation_distillation_epochs
                else "t2_finetune"
            )
        if self.config.model.variant == "unified_abc_v12":
            return "t2_finetune"
        if self.config.model.variant in {"t2_abc_v1", "unified_abc_v11"}:
            return (
                "head_warmup"
                if stage_epoch < self.config.train.t2_head_warmup_epochs else "joint"
            )
        if self.config.model.variant == "v10":
            rank_end = self.config.train.t2_rank_phase_epochs
            open_end = rank_end + self.config.train.t2_open_phase_epochs
            if stage_epoch < rank_end:
                return "rank"
            return "open" if stage_epoch < open_end else "calibration"
        if self.config.model.variant == "v9":
            representation_end = self.config.train.t2_representation_phase_epochs
            unified_end = representation_end + self.config.train.t2_unified_phase_epochs
            if stage_epoch < representation_end:
                return "representation"
            return "unified" if stage_epoch < unified_end else "calibration"
        if self.config.model.variant in {"v7", "v8"}:
            return "rank" if stage_epoch < self.config.train.t2_rank_phase_epochs else "open"
        return "joint"

    def _frozen_t1_digest(self) -> str:
        digest = hashlib.sha256()
        for prefix, module in (("encoder", self.model.encoder), ("t1", self.model.t1)):
            for name, tensor in sorted(module.state_dict().items()):
                digest.update(f"{prefix}.{name}".encode("utf-8"))
                digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _prune_v10_frozen_optimizer_state(self) -> int:
        """Release Adam moments for a V10 path once that path is permanently frozen."""
        if self.config.model.variant != "v10":
            return 0
        removed = 0
        for parameter in self.model.parameters():
            if not parameter.requires_grad and parameter in self.optimizer.state:
                self.optimizer.state.pop(parameter)
                removed += 1
        return removed

    def _set_t2_release(self, stage_epoch: int) -> tuple[float, float]:
        if self.config.model.variant != "v6":
            return 1.0, 1.0
        warmup = self.config.train.t2_release_warmup_epochs
        duration = max(1, self.config.train.t2_release_epochs)
        alpha = min(1.0, max(0.0, (stage_epoch + 1 - warmup) / duration))
        beta = min(1.0, max(0.0, (stage_epoch + 1 - warmup - 2) / duration))
        self.model.t2.set_release(alpha, beta)
        return alpha, beta

    def _late_stage_b_modules(self) -> tuple[torch.nn.Module, ...]:
        encoder = self.model.encoder
        return (
            *tuple(encoder.sequence.blocks[-2:]), encoder.sequence.norm,
            *tuple(encoder.image.features[-2:]),
            encoder.image.stage3_projection, encoder.image.stage4_projection,
            encoder.image.global_projection, encoder.fusion, encoder.late,
        )

    def _late_t2_encoder_modules(self) -> tuple[torch.nn.Module, ...]:
        if self.model.t2_encoder is None:
            raise ValueError("Late T2 encoder modules require an independent T2 encoder")
        encoder = self.model.t2_encoder
        return (
            *tuple(encoder.sequence.blocks[-2:]), encoder.sequence.norm,
            encoder.image.stage3_projection, encoder.image.stage4_projection,
            encoder.image.global_projection, encoder.fusion,
        )

    def _set_stage_b_train_mode(self) -> None:
        self.model.train()
        if self.config.train.stage_b_freeze_encoder:
            self.model.encoder.eval()
            if self.config.train.stage_b_unfreeze_late_encoder:
                for module in self._late_stage_b_modules():
                    module.train()
        if not self.config.train.stage_b_train_t1:
            self.model.t1.eval()
        if self.config.model.variant == "unified_abc_v14":
            phase = self._stage_b_phase(getattr(self, "_active_stage_b_epoch", 0))
            self.model.encoder.eval()
            self.model.t1.eval()
            assert self.model.t2_adapter is not None
            if self.config.train.t2_scratch_from_stage_a:
                self.model.t2.train()
                if phase == "head_warmup":
                    self.model.t2_adapter.eval()
                else:
                    self.model.t2_adapter.train()
                if self.model.t2_energy_adapter is not None:
                    self.model.t2_energy_adapter.train()
            elif phase == "rank_recovery":
                self.model.t2_adapter.train()
                self.model.t2.relation.train()
                self.model.t2.in_set_head.eval()
                if self.model.t2_energy_adapter is not None:
                    self.model.t2_energy_adapter.eval()
            else:
                self.model.t2_adapter.eval()
                self.model.t2.relation.eval()
                self.model.t2.in_set_head.train()
                if self.model.t2_energy_adapter is not None:
                    self.model.t2_energy_adapter.train()
            self.model.t2.rf_head.eval()
        elif self.config.model.variant in {"unified_abc_v12", "unified_abc_v13"}:
            self.model.encoder.eval()
            self.model.t1.eval()
            assert self.model.t2_adapter is not None
            self.model.t2_adapter.train()
        elif self.config.model.variant == "t2_abc_v1":
            phase = self._stage_b_phase(getattr(self, "_active_stage_b_epoch", 0))
            assert self.model.t2_encoder is not None
            trainability = self.config.train.t2_encoder_trainability
            if phase != "joint" or trainability == "head_only":
                self.model.t2_encoder.eval()
            elif trainability == "late_partial":
                self.model.t2_encoder.eval()
                for module in self._late_t2_encoder_modules():
                    module.train()
            if (
                self.config.train.t2_freeze_inactive_modalities
                and self.config.model.fusion == "sequence_only"
            ):
                self.model.t2_encoder.image.eval()
                self.model.t2_encoder.fusion.eval()
                self.model.t2_encoder.late.eval()
            elif (
                self.config.train.t2_freeze_inactive_modalities
                and self.config.model.fusion == "image_only"
            ):
                self.model.t2_encoder.sequence.eval()
                self.model.t2_encoder.fusion.eval()
                self.model.t2_encoder.late.eval()
        if self.config.model.variant in {"v7", "v8", "v9", "v10"}:
            self.model.t2.enforce_training_mode()
        if self.config.model.variant == "v10":
            phase = self._stage_b_phase(getattr(self, "_active_stage_b_epoch", 0))
            assert self.model.t2_rank_encoder is not None
            assert self.model.t2_open_encoder is not None
            if phase != "rank":
                self.model.t2_rank_encoder.eval()
            if phase != "open":
                self.model.t2_open_encoder.eval()
        if self.config.model.variant == "v9" and self._stage_b_phase(
            getattr(self, "_active_stage_b_epoch", 0),
        ) == "calibration":
            assert self.model.t2_encoder is not None
            self.model.t2_encoder.eval()
        if self.config.model.variant == "t2_abc_v1" and self._stage_b_phase(
            getattr(self, "_active_stage_b_epoch", 0),
        ) == "head_warmup":
            assert self.model.t2_encoder is not None
            self.model.t2_encoder.eval()
        if self.config.model.variant == "unified_abc_v11" and self._stage_b_phase(
            getattr(self, "_active_stage_b_epoch", 0),
        ) == "head_warmup":
            self.model.encoder.eval()
            self.model.t1.eval()

    def _scheduler(self, total_steps: int) -> LambdaLR:
        warmup = max(1, int(total_steps * self.config.train.warmup_fraction))

        def scale(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

        return LambdaLR(self.optimizer, scale)

    def train(self, held_out_forger: str | None = None, resume_path: str | Path | None = None,
              initialize_path: str | Path | None = None) -> dict[str, Any]:
        if resume_path is not None and initialize_path is not None:
            raise ValueError("resume_path and initialize_path are mutually exclusive")
        self.held_out_forger = held_out_forger
        if initialize_path is not None:
            self.initialize_from_checkpoint(initialize_path)
        stage_a_loaders = {
            "t1_1v1": self.loader("train", "t1_1v1", train=True, held_out_forger=held_out_forger,
                                   batch_size=self.config.train.stage_a_batch_t1_1v1),
            "t1_5v1": self.loader("train", "t1_5v1", train=True, held_out_forger=held_out_forger,
                                   batch_size=self.config.train.stage_a_batch_t1_5v1),
        }
        # Stage B loaders are only metadata until iterated; workers start after Stage A is released.
        stage_b_loaders = {protocol: self.loader("train", protocol, train=True, held_out_forger=held_out_forger)
                           for protocol in self._stage_b_protocols()}
        stage_a_steps = math.ceil(
            (len(stage_a_loaders["t1_1v1"]) + len(stage_a_loaders["t1_5v1"]))
            / self.config.train.stage_a_grad_accumulation
        )
        stage_b_steps = math.ceil(max(map(len, stage_b_loaders.values())) / self.config.train.grad_accumulation)
        total_steps = self.config.train.stage_a_epochs * stage_a_steps + self.config.train.stage_b_epochs * stage_b_steps
        scheduler = self._scheduler(total_steps)
        start_stage_a, global_epoch = 0, 0
        best_score, stale = -float("inf"), 0
        if resume_path:
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            self._validate_checkpoint_config(checkpoint)
            if checkpoint.get("epoch", -1) >= self.config.train.stage_a_epochs:
                raise ValueError("This resume path currently supports Stage A checkpoints only")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            start_stage_a = int(checkpoint["epoch"]) + 1
            global_epoch = start_stage_a
            history_path = self.output / "history.jsonl"
            if history_path.exists():
                self.history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
                best_score = max((row["selection_score"] for row in self.history if row["stage"] == "stage_a"), default=-float("inf"))
            scheduler.last_epoch = start_stage_a * stage_a_steps
            scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]
            print(f"resumed stage_a at epoch={start_stage_a + 1} optimizer_step={scheduler.last_epoch}", flush=True)
        for epoch in range(start_stage_a, self.config.train.stage_a_epochs):
            self._freeze_early_image(epoch < 5)
            train_log = self._stage_a_epoch(stage_a_loaders, scheduler, epoch)
            validation = self.evaluate_split("val", protocols=("t1_1v1", "t1_5v1"))
            score = self._stage_a_selection_score(
                validation, self.config.train.stage_a_selection_policy,
            )
            self._record("stage_a", global_epoch, train_log, validation, score)
            self._checkpoint("last.pt", global_epoch, validation)
            if score > best_score:
                best_score, stale = score, 0
                self._checkpoint("best_stage_a.pt", global_epoch, validation)
            else:
                stale += 1
            global_epoch += 1
        del stage_a_loaders
        gc.collect()
        self.load_checkpoint(self.output / "best_stage_a.pt", restore_optimizer=True)
        for index, group in enumerate(self.optimizer.param_groups):
            if group.get("name") in {"convnext_backbone", "shared_encoder"}:
                group["lr"] *= 0.3
                scheduler.base_lrs[index] *= 0.3
        stage_a_sf_far = self.evaluate_split("val", protocols=("t1_5v1",))["t1_5v1"].get(
            "genuine_vs_SF", {}
        ).get("far", 1.0)
        best_score, stale = -float("inf"), 0
        for epoch in range(self.config.train.stage_b_epochs):
            self._freeze_early_image(False)
            self._prepare_stage_b_modules()
            train_log = self._stage_b_epoch(stage_b_loaders, scheduler, epoch)
            validation = self.evaluate_split("val")
            score = self._selection_score(validation, self.config.train.selection_policy)
            self._record("stage_b", global_epoch, train_log, validation, score)
            self._checkpoint("last.pt", global_epoch, validation)
            sf_far = validation["t1_5v1"].get("genuine_vs_SF", {}).get("far", 1.0)
            eligible = sf_far <= stage_a_sf_far + 0.05
            if score > best_score and eligible:
                best_score, stale = score, 0
                self._checkpoint("best.pt", global_epoch, validation)
            else:
                stale += 1
            global_epoch += 1
            if stale >= self.config.train.patience:
                break
        best_path = self.output / ("best.pt" if (self.output / "best.pt").exists() else "best_stage_a.pt")
        self.load_checkpoint(best_path)
        calibrator = self.fit_calibration()
        test = self.evaluate_split("test", calibrator=calibrator, export=True)
        atomic_json(self.output / "test_metrics.json", test)
        return test

    def train_t1(self, resume_path: str | Path | None = None) -> dict[str, Any]:
        """Train and evaluate only Stage A on a T1-only benchmark."""
        protocols = ("t1_1v1", "t1_5v1")
        loaders = {
            "t1_1v1": self.loader(
                "train", "t1_1v1", train=True,
                batch_size=self.config.train.stage_a_batch_t1_1v1,
            ),
            "t1_5v1": self.loader(
                "train", "t1_5v1", train=True,
                batch_size=self.config.train.stage_a_batch_t1_5v1,
            ),
        }
        steps_per_epoch = math.ceil(
            (len(loaders["t1_1v1"]) + len(loaders["t1_5v1"]))
            / self.config.train.stage_a_grad_accumulation
        )
        scheduler = self._scheduler(self.config.train.stage_a_epochs * steps_per_epoch)
        if resume_path is None:
            atomic_json(self.output / "fresh_initialization.json", {
                "dvsr_checkpoint": None,
                "resume_checkpoint": None,
                "model_variant": self.config.model.variant,
                "t1_variant": self.config.model.t1_variant or self.config.model.variant,
                "image_backbone": self.config.model.image_backbone,
                "image_backbone_pretrained": self.config.model.pretrained,
                "training_seed": self.config.train.seed,
                "dataset_split_digest": json.loads(
                    (Path(self.config.data.benchmark_root) / "folds/single_split.json").read_text(
                        encoding="utf-8"
                    )
                )["split_digest"],
            })
        best_score = -float("inf")
        stale = 0
        completed_epochs = 0
        start_epoch = 0
        previous_score: float | None = None
        early_stop_reason: str | None = None
        history_path = self.output / "history.jsonl"
        test_history_path = self.output / "test_history.jsonl"
        if history_path.exists():
            self.history = [
                json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if test_history_path.exists():
            self.test_history = [
                json.loads(line) for line in test_history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            self._validate_checkpoint_config(checkpoint)
            if checkpoint.get("stage") != "stage_a_t1_only":
                raise ValueError("T1 resume requires a stage_a_t1_only checkpoint")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("stage_epoch", checkpoint.get("epoch", -1))) + 1
            state = checkpoint.get("training_state", {})
            best_score = float(state.get("best_score", -float("inf")))
            stale = int(state.get("stale", 0))
            early_stop_reason = state.get("early_stop_reason")
            previous_score = state.get("previous_score")
            if previous_score is None and self.history:
                previous_score = float(self.history[-1]["selection_score"])
            completed_epochs = start_epoch
            if (
                early_stop_reason is None
                and self.config.train.stage_a_stop_on_validation_decline
                and self._stage_a_history_declined(self.history)
            ):
                early_stop_reason = "validation_declined_before_resume"
            atomic_json(self.output / "stage_a_resume.json", {
                "checkpoint": str(Path(resume_path).resolve()),
                "start_epoch": start_epoch,
                "restored_scheduler": "scheduler" in checkpoint,
                "best_score": best_score,
                "stale": stale,
                "early_stop_reason": early_stop_reason,
            })
            print(
                f"resumed T1 Stage-A at epoch={start_epoch + 1} "
                f"optimizer_step={scheduler.last_epoch}", flush=True,
            )
        remaining_epochs = (
            range(start_epoch, self.config.train.stage_a_epochs)
            if early_stop_reason is None else ()
        )
        for epoch in remaining_epochs:
            self._freeze_early_image(epoch < 5)
            train_log = self._stage_a_epoch(loaders, scheduler, epoch)
            validation = self.evaluate_split("val", protocols=protocols)
            score = self._stage_a_selection_score(
                validation, self.config.train.stage_a_selection_policy,
            )
            test_selection: dict[str, Any] | None = None
            test_target_reached = False
            test_target_checks: dict[str, bool] = {}
            if self.config.train.stage_a_evaluate_test_each_epoch:
                epoch_calibrator = self.fit_t1_calibration()
                test_selection = self.evaluate_split(
                    "test", calibrator=epoch_calibrator, protocols=protocols,
                )
                test_target_reached, test_target_checks = self._stage_a_test_target_checks(
                    test_selection, self.config.train,
                )
                self.test_history.append({
                    "stage": "stage_a_t1_only",
                    "epoch": epoch,
                    "metrics": test_selection,
                    "target_reached": test_target_reached,
                    "target_checks": test_target_checks,
                    "selection_role": "early_stopping_authorized",
                    "timestamp": time.time(),
                })
                write_jsonl(test_history_path, self.test_history)
            validation_declined = (
                self.config.train.stage_a_stop_on_validation_decline
                and previous_score is not None
                and score < previous_score
            )
            self._record("stage_a_t1_only", epoch, train_log, validation, score)
            improved = score > best_score
            next_stale = 0 if improved else stale + 1
            next_best_score = max(best_score, score)
            if test_target_reached:
                early_stop_reason = "test_target_reached"
            elif validation_declined:
                early_stop_reason = "validation_declined"
            elif next_stale >= self.config.train.patience:
                early_stop_reason = "patience_exhausted"
            state = {
                "best_score": next_best_score,
                "stale": next_stale,
                "previous_score": score,
                "steps_per_epoch": steps_per_epoch,
                "manifest_distribution_preserved": not self.config.train.t1_balance_strata,
                "test_target_reached": test_target_reached,
                "test_target_checks": test_target_checks,
                "early_stop_reason": early_stop_reason,
            }
            self._checkpoint(
                "last_t1.pt", epoch, validation, stage="stage_a_t1_only",
                stage_epoch=epoch, scheduler=scheduler, training_state=state,
            )
            if improved:
                best_score = score
                stale = 0
            else:
                stale += 1
            if improved or test_target_reached:
                self._checkpoint(
                    "best_t1.pt", epoch, validation, stage="stage_a_t1_only",
                    stage_epoch=epoch, scheduler=scheduler, training_state=state,
                )
            completed_epochs = epoch + 1
            previous_score = score
            atomic_json(self.output / "t1_training_status.json", {
                "running": True,
                "completed_epochs": completed_epochs,
                "maximum_epochs": self.config.train.stage_a_epochs,
                "best_score": best_score,
                "stale": stale,
                "last_validation": validation,
                "last_test_selection": test_selection,
                "test_target_checks": test_target_checks,
                "early_stop_reason": early_stop_reason,
            })
            if early_stop_reason is not None:
                break
        del loaders
        gc.collect()
        self.load_checkpoint(self.output / "best_t1.pt")
        calibrator = self.fit_t1_calibration()
        validation = self.evaluate_split("val", calibrator=calibrator, export=True, protocols=protocols)
        result = {
            "model_variant": self.config.model.variant,
            "fresh_dvsr_initialization": True,
            "completed_epochs": completed_epochs,
            "maximum_epochs": self.config.train.stage_a_epochs,
            "early_stopped": completed_epochs < self.config.train.stage_a_epochs,
            "early_stop_reason": early_stop_reason,
            "best_selection_score": best_score,
            "validation": validation,
            "test_executed": self.config.train.stage_a_finalize_test,
        }
        if self.config.train.stage_a_finalize_test:
            test = self.evaluate_split(
                "test", calibrator=calibrator, export=True, protocols=protocols,
            )
            result["test"] = test
            atomic_json(self.output / "test_metrics.json", test)
        atomic_json(self.output / "t1_metrics.json", result)
        atomic_json(self.output / "validation_metrics.json", validation)
        atomic_json(self.output / "t1_training_status.json", {**result, "running": False})
        return result

    def _stage_a_epoch(self, loaders: dict[str, DataLoader], scheduler: LambdaLR, epoch: int) -> dict[str, float]:
        self.model.train()
        totals: dict[str, list[Tensor | float]] = defaultdict(list)
        accumulation = self.config.train.stage_a_grad_accumulation
        micro_step = 0
        self.optimizer.zero_grad(set_to_none=True)
        for protocol in loaders:
            sampler = loaders[protocol].batch_sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for protocol_step, batch in enumerate(loaders[protocol], start=1):
                batch = _move(batch, self.device)
                with self._autocast():
                    output = self.model(batch)
                    loss, parts = self.t1_loss(output, batch)
                (loss / accumulation).backward()
                micro_step += 1
                if micro_step % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                for key, value in parts.items():
                    totals[key].append(float(value.detach()))
                if protocol_step % 100 == 0:
                    print(
                        f"stage_a epoch={epoch + 1} protocol={protocol} step={protocol_step}/{len(loaders[protocol])} "
                        f"loss={float(loss.detach()):.4f}", flush=True,
                    )
        if micro_step % accumulation:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        summary = {key: float(np.mean(values)) for key, values in totals.items()}
        if getattr(self.model.config, "t1_variant", None) == "simple_v2":
            summary["t1_fusion_alpha"] = float(self.model.encoder.fusion_mix_logit.sigmoid().detach())
            summary["t1_product_beta"] = float(self.model.t1.relation.product_mix_logit.sigmoid().detach())
        return summary

    def _prepare_t2_a_modules(self) -> None:
        if self.config.model.variant not in {
            "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
            "unified_abc_v14", "unified_sen_v20",
        }:
            raise ValueError("Balanced T2-A pretraining requires a conditional A/B/C model")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.t2.rf_head.parameters():
            parameter.requires_grad_(True)
        self.model.eval()
        self.model.t2.rf_head.train()

    def _t2_a_epoch(self, loader: DataLoader, scheduler: LambdaLR, epoch: int) -> dict[str, float]:
        self._prepare_t2_a_modules()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        accumulation = self.config.train.t2_a_grad_accumulation
        totals: list[Tensor] = []
        self.optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(loader, start=1):
            batch = _move(batch, self.device)
            with self._autocast():
                output = self.model(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["rf_logit"], batch["rf_label"],
                )
            self._require_finite(loss, "balanced T2-A pretraining loss")
            (loss / accumulation).backward()
            totals.append(loss.detach())
            if micro_step % accumulation == 0:
                self._finite_optimizer_step(
                    self.model.t2.rf_head.parameters(), "balanced T2-A pretraining",
                )
                scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            if micro_step % 25 == 0:
                print(
                    f"t2_a epoch={epoch + 1} step={micro_step}/{len(loader)} "
                    f"loss={float(loss.detach()):.4f}", flush=True,
                )
        if len(loader) % accumulation:
            self._finite_optimizer_step(
                self.model.t2.rf_head.parameters(), "balanced T2-A pretraining",
            )
            scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        return {
            "t2_a_bce": float(torch.stack(totals).mean().cpu()),
            "micro_steps": float(len(loader)),
        }

    def _run_t2_a_pretraining(
        self, loader: DataLoader, scheduler: LambdaLR, global_epoch: int,
        start_epoch: int = 0, best_score: float = -float("inf"),
        best_threshold: float = 0.5, stale: int = 0,
    ) -> tuple[int, float, float, int]:
        """Fit the RF/SF head on one row per Query before joint T2 training."""
        completed = start_epoch
        for epoch in range(start_epoch, self.config.train.t2_a_pretrain_epochs):
            if stale >= self.config.train.t2_a_patience:
                break
            train_log = self._t2_a_epoch(loader, scheduler, epoch)
            threshold, metrics = self.evaluate_balanced_a("val")
            score = float(metrics["balanced_accuracy"])
            improved = score > best_score
            if improved:
                best_score = score
                best_threshold = threshold
                stale = 0
            else:
                stale += 1
            validation = {"t2_a_balanced": metrics}
            self._record("t2_a", global_epoch, train_log, validation, score)
            state = {
                "best_a_balanced_accuracy": best_score,
                "best_a_threshold": best_threshold,
                "stale": stale,
            }
            self._checkpoint(
                "last_a.pt", global_epoch, validation, stage="t2_a", stage_epoch=epoch,
                scheduler=scheduler, training_state=state,
            )
            if improved:
                self._checkpoint(
                    "best_a.pt", global_epoch, validation, stage="t2_a", stage_epoch=epoch,
                    scheduler=scheduler, training_state=state,
                )
            global_epoch += 1
            completed = epoch + 1
            atomic_json(self.output / "t2_a_status.json", {
                "stage": "t2_a",
                "running": True,
                "completed_epochs": completed,
                "stale": stale,
                "best_balanced_accuracy": best_score,
                "best_threshold": best_threshold,
                "last_validation": metrics,
            })

        best_checkpoint = torch.load(
            self.output / "best_a.pt", map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(best_checkpoint["model"])
        result = {
            "stage": "t2_a",
            "running": False,
            "completed_epochs": completed,
            "best_balanced_accuracy": best_score,
            "best_threshold": best_threshold,
            "best_checkpoint": str((self.output / "best_a.pt").resolve()),
        }
        atomic_json(self.output / "t2_a_status.json", result)
        return global_epoch, best_score, best_threshold, completed

    def _prepare_t2_b_modules(self) -> None:
        if self.config.model.variant not in {
            "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
            "unified_abc_v14", "unified_sen_v20",
        }:
            raise ValueError("Balanced T2-B pretraining requires a conditional A/B/C model")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self._t2_b_parameters():
            parameter.requires_grad_(True)
        if self.model.t2_energy_adapter is not None:
            for parameter in self.model.t2_energy_adapter.parameters():
                parameter.requires_grad_(True)
        self.model.eval()
        self.model.t2.in_set_head.train()
        null_source_head = getattr(self.model.t2, "null_source_head", None)
        if null_source_head is not None:
            null_source_head.train()
        if self.model.t2_energy_adapter is not None:
            self.model.t2_energy_adapter.train()

    def _t2_b_parameters(self) -> list[Tensor]:
        source_parameters = getattr(self.model.t2, "source_existence_parameters", None)
        parameters = (
            list(source_parameters())
            if source_parameters is not None else list(self.model.t2.in_set_head.parameters())
        )
        if self.model.t2_energy_adapter is not None:
            parameters.extend(self.model.t2_energy_adapter.parameters())
        return parameters

    def _t2_b_epoch(self, loader: DataLoader, scheduler: LambdaLR, epoch: int) -> dict[str, float]:
        self._prepare_t2_b_modules()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        accumulation = self.config.train.t2_b_grad_accumulation
        totals: list[Tensor] = []
        present_count = absent_count = 0
        self.optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(loader, start=1):
            batch = _move(batch, self.device)
            if bool((batch["episode_type_index"] == 2).any()):
                raise ValueError("Balanced T2-B training must contain SF episodes only")
            with self._autocast():
                output = self.model(batch)
                target = batch["exist_label"].to(output["in_set_logit"].dtype)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["in_set_logit"], target,
                )
            self._require_finite(loss, "balanced T2-B pretraining loss")
            (loss / accumulation).backward()
            totals.append(loss.detach())
            present_count += int((target == 1).sum())
            absent_count += int((target == 0).sum())
            if micro_step % accumulation == 0:
                self._finite_optimizer_step(
                    self._t2_b_parameters(), "balanced T2-B pretraining",
                )
                scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            if micro_step % 50 == 0:
                print(
                    f"t2_b epoch={epoch + 1} step={micro_step}/{len(loader)} "
                    f"loss={float(loss.detach()):.4f}", flush=True,
                )
            del batch, output, target, loss
        if len(loader) % accumulation:
            self._finite_optimizer_step(
                self._t2_b_parameters(), "balanced T2-B pretraining",
            )
            scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        return {
            "t2_b_bce": float(torch.stack(totals).mean().cpu()),
            "t2_b_present_episodes": float(present_count),
            "t2_b_absent_episodes": float(absent_count),
            "micro_steps": float(len(loader)),
        }

    def _run_t2_b_pretraining(
        self, loader: DataLoader, scheduler: LambdaLR, global_epoch: int,
        best_a_score: float, best_a_threshold: float,
        start_epoch: int = 0, best_score: float = -float("inf"), stale: int = 0,
    ) -> tuple[int, float, int]:
        """Fit the source-presence head on balanced SF Present/Absent episodes."""
        completed = start_epoch
        for epoch in range(start_epoch, self.config.train.t2_b_pretrain_epochs):
            if stale >= self.config.train.t2_b_patience:
                break
            train_log = self._t2_b_epoch(loader, scheduler, epoch)
            metrics = self.evaluate_balanced_b("val")
            score = float(metrics["accuracy"])
            improved = score > best_score
            if improved:
                best_score = score
                stale = 0
            else:
                stale += 1
            validation = {"t2_b_balanced": metrics}
            self._record("t2_b", global_epoch, train_log, validation, score)
            state = {
                "best_a_balanced_accuracy": best_a_score,
                "best_a_threshold": best_a_threshold,
                "best_b_balanced_accuracy": best_score,
                "stale": stale,
            }
            self._checkpoint(
                "last_b.pt", global_epoch, validation, stage="t2_b", stage_epoch=epoch,
                scheduler=scheduler, training_state=state,
            )
            if improved:
                self._checkpoint(
                    "best_b.pt", global_epoch, validation, stage="t2_b", stage_epoch=epoch,
                    scheduler=scheduler, training_state=state,
                )
            global_epoch += 1
            completed = epoch + 1
            atomic_json(self.output / "t2_b_status.json", {
                "stage": "t2_b",
                "running": True,
                "completed_epochs": completed,
                "stale": stale,
                "best_balanced_accuracy": best_score,
                "last_validation": metrics,
            })

        best_checkpoint = torch.load(
            self.output / "best_b.pt", map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(best_checkpoint["model"])
        result = {
            "stage": "t2_b",
            "running": False,
            "completed_epochs": completed,
            "best_balanced_accuracy": best_score,
            "best_checkpoint": str((self.output / "best_b.pt").resolve()),
        }
        atomic_json(self.output / "t2_b_status.json", result)
        return global_epoch, best_score, completed

    def _balanced_a_backward(self, stream: Iterator[dict[str, Any]]) -> tuple[Tensor, dict[str, Any]]:
        batch = _move(next(stream), self.device)
        with self._autocast():
            output = self.model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["rf_logit"], batch["rf_label"],
            )
        self._require_finite(loss, "balanced T2-A auxiliary loss")
        (self.config.train.t2_a_loss_weight * loss).backward()
        metadata = {
            "loss": loss.detach(),
            "sf": int((batch["rf_label"] == 0).sum()),
            "rf": int((batch["rf_label"] == 1).sum()),
        }
        del batch, output
        return loss.detach(), metadata

    def _balanced_b_backward(self, stream: Iterator[dict[str, Any]]) -> tuple[Tensor, dict[str, Any]]:
        batch = _move(next(stream), self.device)
        if bool((batch["episode_type_index"] == 2).any()):
            raise ValueError("Balanced T2-B auxiliary batches must contain SF episodes only")
        with self._autocast():
            output = self.model(batch)
            target = batch["exist_label"].to(output["in_set_logit"].dtype)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["in_set_logit"], target,
            )
        self._require_finite(loss, "balanced T2-B auxiliary loss")
        (self.config.train.t2_b_loss_weight * loss).backward()
        metadata = {
            "loss": loss.detach(),
            "present": int((target == 1).sum()),
            "absent": int((target == 0).sum()),
        }
        del batch, output, target
        return loss.detach(), metadata

    @staticmethod
    def _mean_training_values(values: list[Tensor | float]) -> float:
        if not values:
            return float("nan")
        if isinstance(values[0], Tensor):
            return float(torch.stack(values).mean().cpu())
        return float(np.mean(values))

    def _stage_b_epoch(self, loaders: dict[str, DataLoader], scheduler: LambdaLR, epoch: int,
                       max_steps: int | None = None,
                       balanced_a_loader: DataLoader | None = None,
                       balanced_b_loader: DataLoader | None = None) -> dict[str, float]:
        # PyTorch fused Adam cannot mix restored Stage A state with newly active T2 parameters.
        self.optimizer.defaults["fused"] = False
        for group in self.optimizer.param_groups:
            group["fused"] = False
        self._active_stage_b_epoch = epoch
        self._set_stage_b_train_mode()
        for loader in loaders.values():
            if hasattr(loader.batch_sampler, "set_epoch"):
                loader.batch_sampler.set_epoch(epoch)
        streams = {
            name: _cycle(loader, reshuffle=self.config.train.reshuffle_on_cycle)
            for name, loader in loaders.items()
        }
        balanced_a_stream = None
        if balanced_a_loader is not None:
            if hasattr(balanced_a_loader.batch_sampler, "set_epoch"):
                balanced_a_loader.batch_sampler.set_epoch(epoch)
            balanced_a_stream = _cycle(balanced_a_loader, reshuffle=True)
        balanced_b_stream = None
        if balanced_b_loader is not None:
            if hasattr(balanced_b_loader.batch_sampler, "set_epoch"):
                balanced_b_loader.batch_sampler.set_epoch(epoch)
            balanced_b_stream = _cycle(balanced_b_loader, reshuffle=True)
        full_steps = max(map(len, loaders.values()))
        steps = min(full_steps, max_steps) if max_steps is not None else full_steps
        accumulation = self.config.train.grad_accumulation
        totals: dict[str, list[Tensor | float]] = defaultdict(list)
        train_t1_this_epoch = self.config.train.stage_b_train_t1 and any(
            parameter.requires_grad for parameter in self.model.t1.parameters()
        )
        if not train_t1_this_epoch:
            self.optimizer.zero_grad(set_to_none=True)
            for micro_step in range(steps):
                batch_2 = _move(next(streams["t2"]), self.device)
                with self._autocast():
                    output_2 = self.model(batch_2)
                    self._attach_t2_teacher_targets(output_2, batch_2, epoch)
                    loss_t2, parts_2 = self.t2_loss(output_2, batch_2)
                self._require_finite(loss_t2, "Stage B T2 loss")
                (loss_t2 / accumulation).backward()
                loss_t2_value = loss_t2.detach()
                for key, value in parts_2.items():
                    totals[key].append(value.detach())
                del batch_2, output_2, loss_t2, parts_2
                if (micro_step + 1) % accumulation == 0:
                    if balanced_a_stream is not None:
                        a_loss, a_info = self._balanced_a_backward(balanced_a_stream)
                        totals["t2_a_balanced_bce"].append(a_loss)
                        totals["t2_a_balanced_sf_queries"].append(float(a_info["sf"]))
                        totals["t2_a_balanced_rf_queries"].append(float(a_info["rf"]))
                    if balanced_b_stream is not None:
                        b_loss, b_info = self._balanced_b_backward(balanced_b_stream)
                        totals["t2_b_balanced_bce"].append(b_loss)
                        totals["t2_b_balanced_present_episodes"].append(float(b_info["present"]))
                        totals["t2_b_balanced_absent_episodes"].append(float(b_info["absent"]))
                    active_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                    self._finite_optimizer_step(active_parameters, "Stage B T2 update")
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                if (micro_step + 1) % 50 == 0:
                    print(
                        f"stage_b epoch={epoch + 1} step={micro_step + 1}/{steps} "
                        f"t2={loss_t2_value:.4f}", flush=True,
                    )
            if steps % accumulation:
                if balanced_a_stream is not None:
                    a_loss, a_info = self._balanced_a_backward(balanced_a_stream)
                    totals["t2_a_balanced_bce"].append(a_loss)
                    totals["t2_a_balanced_sf_queries"].append(float(a_info["sf"]))
                    totals["t2_a_balanced_rf_queries"].append(float(a_info["rf"]))
                if balanced_b_stream is not None:
                    b_loss, b_info = self._balanced_b_backward(balanced_b_stream)
                    totals["t2_b_balanced_bce"].append(b_loss)
                    totals["t2_b_balanced_present_episodes"].append(float(b_info["present"]))
                    totals["t2_b_balanced_absent_episodes"].append(float(b_info["absent"]))
                active_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
                self._finite_optimizer_step(active_parameters, "Stage B T2 update")
                scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            result = {
                key: self._mean_training_values(values)
                for key, values in totals.items()
            }
            result["micro_steps"] = float(steps)
            return result
        shared_parameters = [p for p in self.model.encoder.parameters() if p.requires_grad]
        if self.model.shared_relation is not None:
            shared_parameters.extend(
                parameter for parameter in self.model.shared_relation.parameters()
                if parameter.requires_grad
            )
        t1_parameters = [p for p in self.model.t1.parameters() if p.requires_grad]
        t2_parameters = [p for p in self.model.t2.parameters() if p.requires_grad]
        self.optimizer.zero_grad(set_to_none=True)
        for micro_step in range(steps):
            if self.config.train.pcgrad:
                batch_1 = _move(next(streams["t1_1v1"]), self.device)
                with self._autocast():
                    output_1 = self.model(batch_1)
                    loss_1, parts_1 = self.t1_loss(output_1, batch_1)
                self._require_finite(loss_1, "Stage B T1 1v1 loss")
                grad_1_shared, grad_1_head = self._task_gradients(
                    loss_1 * (0.5 / accumulation), shared_parameters, t1_parameters,
                    "Stage B T1 1v1",
                )
                loss_1_value = loss_1.detach()
                parts_1_values = {key: value.detach() for key, value in parts_1.items()}
                del batch_1, output_1, loss_1, parts_1

                batch_5 = _move(next(streams["t1_5v1"]), self.device)
                with self._autocast():
                    output_5 = self.model(batch_5)
                    loss_5, parts_5 = self.t1_loss(output_5, batch_5)
                self._require_finite(loss_5, "Stage B T1 5v1 loss")
                grad_5_shared, grad_5_head = self._task_gradients(
                    loss_5 * (0.5 / accumulation), shared_parameters, t1_parameters,
                    "Stage B T1 5v1",
                )
                loss_5_value = loss_5.detach()
                parts_5_values = {key: value.detach() for key, value in parts_5.items()}
                del batch_5, output_5, loss_5, parts_5

                grad_t1 = self._sum_gradients(grad_1_shared, grad_5_shared)
                head_t1 = self._sum_gradients(grad_1_head, grad_5_head)
                del grad_1_shared, grad_5_shared, grad_1_head, grad_5_head

                batch_2 = _move(next(streams["t2"]), self.device)
                if self.config.model.variant == "unified_abc_v11":
                    batch_2["compute_t1_bridge"] = True
                with self._autocast():
                    output_2 = self.model(batch_2)
                    loss_t2, parts_2 = self.t2_loss(output_2, batch_2)
                self._require_finite(loss_t2, "Stage B T2 loss")
                grad_t2, head_t2 = self._task_gradients(
                    loss_t2 / accumulation, shared_parameters, t2_parameters,
                    "Stage B T2",
                )
                loss_t2_value = loss_t2.detach()
                parts_2_values = {key: value.detach() for key, value in parts_2.items()}
                del batch_2, output_2, loss_t2, parts_2

                projected, cosine = project_conflicting(grad_t1, grad_t2)
                for gradient in projected:
                    if gradient is not None:
                        self._require_finite(gradient, "Stage B PCGrad projected gradients")
                self._require_finite(cosine, "Stage B PCGrad cosine")
                self._accumulate_gradients(shared_parameters, projected)
                self._accumulate_gradients(t1_parameters, head_t1)
                self._accumulate_gradients(t2_parameters, head_t2)
                totals["gradient_cosine"].append(cosine.detach())
                loss_t1_value = 0.5 * (loss_1_value + loss_5_value)
            else:
                batch_1 = _move(next(streams["t1_1v1"]), self.device)
                with self._autocast():
                    output_1 = self.model(batch_1)
                    loss_1, parts_1 = self.t1_loss(output_1, batch_1)
                self._require_finite(loss_1, "Stage B T1 1v1 loss")
                (loss_1 * (0.5 / accumulation)).backward()
                loss_1_value = loss_1.detach()
                parts_1_values = {key: value.detach() for key, value in parts_1.items()}
                del batch_1, output_1, loss_1, parts_1

                batch_5 = _move(next(streams["t1_5v1"]), self.device)
                with self._autocast():
                    output_5 = self.model(batch_5)
                    loss_5, parts_5 = self.t1_loss(output_5, batch_5)
                self._require_finite(loss_5, "Stage B T1 5v1 loss")
                (loss_5 * (0.5 / accumulation)).backward()
                loss_5_value = loss_5.detach()
                parts_5_values = {key: value.detach() for key, value in parts_5.items()}
                del batch_5, output_5, loss_5, parts_5

                batch_2 = _move(next(streams["t2"]), self.device)
                if self.config.model.variant == "unified_abc_v11":
                    batch_2["compute_t1_bridge"] = True
                with self._autocast():
                    output_2 = self.model(batch_2)
                    loss_t2, parts_2 = self.t2_loss(output_2, batch_2)
                self._require_finite(loss_t2, "Stage B T2 loss")
                (loss_t2 / accumulation).backward()
                loss_t1_value = 0.5 * (loss_1_value + loss_5_value)
                loss_t2_value = loss_t2.detach()
                parts_2_values = {key: value.detach() for key, value in parts_2.items()}
                del batch_2, output_2, loss_t2, parts_2
            if (micro_step + 1) % accumulation == 0:
                self._finite_optimizer_step(self.model.parameters(), "Stage B joint update")
                scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            for parts in (parts_1_values, parts_5_values, parts_2_values):
                for key, value in parts.items():
                    totals[key].append(value)
            if (micro_step + 1) % 50 == 0:
                print(
                    f"stage_b epoch={epoch + 1} step={micro_step + 1}/{steps} "
                    f"t1={loss_t1_value:.4f} t2={loss_t2_value:.4f}", flush=True,
                )
        if steps % accumulation:
            self._finite_optimizer_step(self.model.parameters(), "Stage B joint update")
            scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        result = {
            key: self._mean_training_values(values)
            for key, values in totals.items()
        }
        result["micro_steps"] = float(steps)
        return result

    def profile_stage_b(
        self, checkpoint_path: str | Path, steps: int = 100, stage_epoch: int = 0,
    ) -> dict[str, Any]:
        if steps < 1:
            raise ValueError("steps must be positive")
        if stage_epoch < 0 or stage_epoch >= self.config.train.stage_b_epochs:
            raise ValueError("stage_epoch must be within the configured Stage B range")
        if self.config.model.variant in {
            "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
        }:
            self.initialize_unified_stage_b(checkpoint_path)
        elif self.config.model.variant == "unified_sen_v20":
            self.initialize_sen_stage_b(checkpoint_path)
        elif self.config.train.stage_b_strict_t1_isolation:
            self.initialize_isolated_stage_b(checkpoint_path)
        else:
            self.load_checkpoint(checkpoint_path, restore_optimizer=True)
        anchor_digest = self._frozen_t1_digest()
        loaders = {protocol: self.loader("train", protocol, train=True)
                   for protocol in self._stage_b_protocols()}
        balanced_a_loader = (
            self.loader("train", "t2_a", train=True)
            if self.config.train.t2_balanced_a_enabled else None
        )
        balanced_b_loader = (
            self.loader("train", "t2_b", train=True)
            if self.config.train.t2_balanced_b_enabled else None
        )
        full_micro_steps = max(map(len, loaders.values()))
        optimizer_steps = math.ceil(full_micro_steps / self.config.train.grad_accumulation)
        scheduler = self._scheduler(max(1, self.config.train.stage_b_epochs * optimizer_steps))
        for index, group in enumerate(self.optimizer.param_groups):
            if group.get("name") in {"convnext_backbone", "shared_encoder"}:
                group["lr"] *= 0.3
                scheduler.base_lrs[index] *= 0.3
        self._active_stage_b_epoch = stage_epoch
        self._prepare_stage_b_modules(stage_epoch)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        train_log = self._stage_b_epoch(
            loaders, scheduler, epoch=stage_epoch, max_steps=steps,
            balanced_a_loader=balanced_a_loader,
            balanced_b_loader=balanced_b_loader,
        )
        frozen_t1_unchanged = self._frozen_t1_digest() == anchor_digest
        if self.config.train.stage_b_strict_t1_isolation and not frozen_t1_unchanged:
            raise RuntimeError("Frozen encoder/T1 state changed during Stage B profile")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actual_steps = int(train_log["micro_steps"])
        result = {
            "pcgrad": self.config.train.pcgrad,
            "requested_steps": steps,
            "stage_epoch": stage_epoch,
            "training_phase": self._stage_b_phase(stage_epoch),
            "actual_steps": actual_steps,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / actual_steps,
            "full_epoch_steps": full_micro_steps,
            "projected_epoch_minutes": elapsed / actual_steps * full_micro_steps / 60,
            "train": train_log,
            "frozen_t1_unchanged": frozen_t1_unchanged,
        }
        if self.device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024 ** 3
        atomic_json(self.output / "stage_b_profile.json", result)
        return result

    def _run_t2_adapter_distillation(
        self, loader: DataLoader, scheduler: LambdaLR,
    ) -> dict[str, Any]:
        if self.config.model.variant == "unified_abc_v14":
            return self._run_t2_internal_adapter_distillation(loader, scheduler)
        if self.config.model.variant not in {"unified_abc_v12", "unified_abc_v13"}:
            return {"completed_steps": 0, "enabled": False}
        steps = self.config.train.t2_adapter_distillation_steps
        if steps < 1:
            raise ValueError(
                f"{self.config.model.variant} requires t2_adapter_distillation_steps >= 1"
            )
        checkpoint_path = self.config.train.t2_initialization_checkpoint
        if not checkpoint_path:
            raise ValueError("V1.2 adapter distillation requires a T2 initialization checkpoint")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        teacher = SignatureEncoder(replace(self.config.model, pretrained=False))
        teacher_state = {
            key.removeprefix("t2_encoder."): value
            for key, value in checkpoint["model"].items()
            if key.startswith("t2_encoder.")
        }
        expected = teacher.state_dict()
        missing = sorted(key for key in expected if key not in teacher_state)
        incompatible = sorted(
            key for key in expected
            if key in teacher_state and expected[key].shape != teacher_state[key].shape
        )
        if missing or incompatible:
            raise ValueError(
                "T2 teacher encoder is incomplete: "
                f"missing={missing[:5]}, incompatible={incompatible[:5]}"
            )
        teacher.load_state_dict(teacher_state)
        teacher.to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        del checkpoint, teacher_state

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        assert self.model.t2_adapter is not None
        for parameter in self.model.t2_adapter.parameters():
            parameter.requires_grad_(True)
        self.model.eval()
        self.model.t2_adapter.train()
        stream = _cycle(loader, reshuffle=True)
        totals: list[float] = []
        global_totals: list[float] = []
        local_totals: list[float] = []
        self.optimizer.zero_grad(set_to_none=True)
        for step in range(steps):
            batch = _move(next(stream), self.device)
            arguments = (
                batch["sequence"], batch["sequence_mask"], batch["image"],
                batch["anchors"], batch["anchor_mask"],
            )
            with torch.no_grad(), self._autocast():
                shared = self.model.encoder(*arguments)
                target = teacher(*arguments)
            with self._autocast():
                adapted = self.model.t2_adapter(shared)
                global_loss = sum(
                    torch.nn.functional.smooth_l1_loss(left, right)
                    for left, right in (
                        (adapted.global_shared, target.global_shared),
                        (adapted.global_sequence, target.global_sequence),
                        (adapted.global_image, target.global_image),
                    )
                ) / 3
                valid = target.valid_mask
                local_difference = torch.nn.functional.smooth_l1_loss(
                    adapted.local_shared, target.local_shared, reduction="none",
                )
                local_weight = valid.unsqueeze(-1).to(local_difference.dtype)
                local_loss = (local_difference * local_weight).sum() / (
                    local_weight.sum() * local_difference.shape[-1]
                ).clamp_min(1)
                loss = (
                    self.config.train.t2_adapter_global_distillation_weight * global_loss
                    + self.config.train.t2_adapter_local_distillation_weight * local_loss
                )
            self._require_finite(loss, "T2 adapter distillation loss")
            loss.backward()
            self._finite_optimizer_step(
                self.model.t2_adapter.parameters(), "T2 adapter distillation",
            )
            scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            totals.append(float(loss.detach()))
            global_totals.append(float(global_loss.detach()))
            local_totals.append(float(local_loss.detach()))
            if (step + 1) % 25 == 0 or step + 1 == steps:
                print(
                    f"t2_adapter_distillation step={step + 1}/{steps} "
                    f"loss={totals[-1]:.6f}", flush=True,
                )
                atomic_json(self.output / "t2_adapter_status.json", {
                    "stage": "t2_adapter_distillation", "running": True,
                    "completed_steps": step + 1, "total_steps": steps,
                    "latest_loss": totals[-1],
                })
            del batch, shared, target, adapted, loss, global_loss, local_loss

        result = {
            "stage": "t2_adapter_distillation", "running": False, "enabled": True,
            "completed_steps": steps, "total_steps": steps,
            "mean_loss": float(np.mean(totals)),
            "mean_global_loss": float(np.mean(global_totals)),
            "mean_local_loss": float(np.mean(local_totals)),
            "teacher_checkpoint": str(Path(checkpoint_path).resolve()),
        }
        atomic_json(self.output / "t2_adapter_status.json", result)
        self._checkpoint(
            "adapter_distilled.pt", -1, {"t2_adapter_distillation": result},
            stage="t2_adapter_distillation", stage_epoch=-1, scheduler=scheduler,
            training_state={"adapter_distillation_completed": True},
        )
        del teacher
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def _run_t2_internal_adapter_distillation(
        self, loader: DataLoader, scheduler: LambdaLR,
    ) -> dict[str, Any]:
        steps = self.config.train.t2_adapter_distillation_steps
        if steps < 1:
            raise ValueError("Unified V1.4 requires t2_adapter_distillation_steps >= 1")
        checkpoint_path = self.config.train.t2_initialization_checkpoint
        if not checkpoint_path:
            raise ValueError("Unified V1.4 distillation requires a T2 initialization checkpoint")

        self._set_t2_relation_teacher(True)
        assert self.t2_relation_teacher is not None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        assert self.model.t2_adapter is not None
        for parameter in self.model.t2_adapter.parameters():
            parameter.requires_grad_(True)
        self.model.eval()
        self.model.t2_adapter.train()

        stream = _cycle(loader, reshuffle=True)
        totals: list[float] = []
        internal_totals: list[float] = []
        global_totals: list[float] = []
        local_totals: list[float] = []
        rank_totals: list[float] = []
        b_totals: list[float] = []
        self.optimizer.zero_grad(set_to_none=True)
        for step in range(steps):
            batch = _move(next(stream), self.device)
            with self._autocast():
                student_output = self.model(batch, return_encoder_stages=True)
            with torch.no_grad(), self._autocast():
                teacher_output = self.t2_relation_teacher(
                    batch, return_encoder_stages=True,
                )
            student_stages = student_output.pop("encoder_stages")
            teacher_stages = teacher_output.pop("encoder_stages")
            common = sorted(student_stages.keys() & teacher_stages.keys())
            internal_keys = [key for key in common if not key.startswith("encoding.")]
            if not internal_keys:
                raise RuntimeError("Unified V1.4 teacher/student have no common internal stages")
            internal_loss = torch.stack([
                torch.nn.functional.smooth_l1_loss(
                    student_stages[key], teacher_stages[key].detach(),
                )
                for key in internal_keys
            ]).mean()
            global_keys = [
                key for key in common
                if key in {
                    "encoding.global_shared", "encoding.global_sequence", "encoding.global_image",
                }
            ]
            global_loss = torch.stack([
                torch.nn.functional.smooth_l1_loss(
                    student_stages[key], teacher_stages[key].detach(),
                )
                for key in global_keys
            ]).mean()
            local_loss = torch.nn.functional.smooth_l1_loss(
                student_stages["encoding.local_shared"],
                teacher_stages["encoding.local_shared"].detach(),
            )
            present = batch["target_index"] >= 0
            if present.any():
                temperature = self.config.train.t2_distillation_temperature
                rank_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(
                        student_output["rank_logits"][present] / temperature, dim=-1,
                    ),
                    torch.nn.functional.softmax(
                        teacher_output["rank_logits"][present].detach() / temperature, dim=-1,
                    ),
                    reduction="batchmean",
                ) * temperature ** 2
            else:
                rank_loss = internal_loss.new_zeros(())
            sf = batch["episode_type_index"] != 2
            b_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    student_output["in_set_logit"][sf],
                    torch.sigmoid(teacher_output["in_set_logit"][sf].detach()),
                )
                if sf.any() else internal_loss.new_zeros(())
            )
            loss = (
                self.config.train.t2_adapter_internal_distillation_weight * internal_loss
                + self.config.train.t2_adapter_global_distillation_weight * global_loss
                + self.config.train.t2_adapter_local_distillation_weight * local_loss
                + self.config.train.t2_rank_distillation_weight * rank_loss
                + self.config.train.t2_b_distillation_weight * b_loss
            )
            self._require_finite(loss, "T2 internal adapter distillation loss")
            loss.backward()
            self._finite_optimizer_step(
                self.model.t2_adapter.parameters(), "T2 internal adapter distillation",
            )
            scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            totals.append(float(loss.detach()))
            internal_totals.append(float(internal_loss.detach()))
            global_totals.append(float(global_loss.detach()))
            local_totals.append(float(local_loss.detach()))
            rank_totals.append(float(rank_loss.detach()))
            b_totals.append(float(b_loss.detach()))
            if (step + 1) % 25 == 0 or step + 1 == steps:
                print(
                    f"t2_internal_adapter_distillation step={step + 1}/{steps} "
                    f"loss={totals[-1]:.6f}", flush=True,
                )
                atomic_json(self.output / "t2_adapter_status.json", {
                    "stage": "t2_internal_adapter_distillation", "running": True,
                    "completed_steps": step + 1, "total_steps": steps,
                    "latest_loss": totals[-1],
                })

        result = {
            "stage": "t2_internal_adapter_distillation", "running": False, "enabled": True,
            "completed_steps": steps, "total_steps": steps,
            "mean_loss": float(np.mean(totals)),
            "mean_internal_loss": float(np.mean(internal_totals)),
            "mean_global_loss": float(np.mean(global_totals)),
            "mean_local_loss": float(np.mean(local_totals)),
            "mean_rank_distillation_loss": float(np.mean(rank_totals)),
            "mean_b_distillation_loss": float(np.mean(b_totals)),
            "teacher_checkpoint": str(Path(checkpoint_path).resolve()),
        }
        reconstruction_validation = self.evaluate_split("val", protocols=("t2",))
        source_present = reconstruction_validation["t2"]["source_present"]
        reconstruction_count = int(source_present["n"])
        tolerance_samples = self.config.train.t2_adapter_reconstruction_tolerance_samples
        reconstruction_checks = {
            "rank_1": self._metric_meets_sample_tolerant_floor(
                source_present["rank_1"], self.config.train.t2_adapter_min_rank_1,
                reconstruction_count, tolerance_samples,
            ),
            "rank_3": self._metric_meets_sample_tolerant_floor(
                source_present["rank_3"], self.config.train.t2_adapter_min_rank_3,
                reconstruction_count, tolerance_samples,
            ),
        }
        result["reconstruction_validation"] = reconstruction_validation
        result["reconstruction_checks"] = reconstruction_checks
        result["reconstruction_gate"] = {
            "n": reconstruction_count,
            "tolerance_samples": tolerance_samples,
            "rank_1_floor": self.config.train.t2_adapter_min_rank_1,
            "rank_3_floor": self.config.train.t2_adapter_min_rank_3,
        }
        atomic_json(self.output / "t2_adapter_status.json", result)
        self._set_t2_relation_teacher(False)
        self._prepare_stage_b_modules(0)
        reconstruction_passed = all(reconstruction_checks.values())
        self._checkpoint(
            "adapter_distilled.pt", -1, {"t2_adapter_distillation": result},
            stage="t2_adapter_distillation", stage_epoch=-1, scheduler=scheduler,
            training_state={"adapter_distillation_completed": reconstruction_passed},
        )
        if not reconstruction_passed:
            raise RuntimeError(
                "Unified V1.4 internal adapter did not meet the Validation rank reconstruction "
                f"floors: {reconstruction_checks}"
            )
        return result

    @staticmethod
    def _metric_meets_sample_tolerant_floor(
        value: float, floor: float, sample_count: int, tolerance_samples: int,
    ) -> bool:
        if sample_count < 1:
            raise ValueError("reconstruction floor requires at least one sample")
        if tolerance_samples < 0:
            raise ValueError("reconstruction tolerance samples must be non-negative")
        observed_correct = float(value) * sample_count
        required_correct = float(floor) * sample_count
        return observed_correct + tolerance_samples + 1e-9 >= required_correct

    def _set_t2_relation_teacher(self, enabled: bool) -> None:
        if not enabled:
            if self.t2_relation_teacher is not None:
                del self.t2_relation_teacher
                self.t2_relation_teacher = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            return
        if self.t2_relation_teacher is not None:
            return
        checkpoint_path = self.config.train.t2_initialization_checkpoint
        if not checkpoint_path:
            raise ValueError("Relation distillation requires t2_initialization_checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_model = dict(checkpoint.get("config", {}).get("model", {}))
        if not saved_model:
            raise ValueError("T2 teacher checkpoint does not contain a model configuration")
        saved_model["pretrained"] = False
        teacher = DVSRNet(ModelConfig(**saved_model))
        teacher.load_state_dict(checkpoint["model"])
        teacher.to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.t2_relation_teacher = teacher
        del checkpoint

    def _attach_t2_teacher_targets(
        self, output: dict[str, Tensor], batch: dict[str, Any], stage_epoch: int,
    ) -> None:
        enabled = (
            self.config.model.variant == "unified_abc_v13"
            and stage_epoch < self.config.train.t2_relation_distillation_epochs
        ) or (
            self.config.model.variant == "unified_abc_v14"
            and not self.config.train.t2_scratch_from_stage_a
        )
        self._set_t2_relation_teacher(enabled)
        if not enabled:
            return
        assert self.t2_relation_teacher is not None
        with torch.no_grad(), self._autocast():
            teacher_output = self.t2_relation_teacher(batch)
        output["teacher_rank_logits"] = teacher_output["rank_logits"].detach()
        output["teacher_in_set_logit"] = teacher_output["in_set_logit"].detach()

    def train_stage_b(self, stage_a_checkpoint: str | Path | None = None,
                      resume_path: str | Path | None = None, epochs: int | None = None,
                      held_out_forger: str | None = None, finalize: bool = False,
                      reset_patience: bool = False) -> dict[str, Any]:
        if (stage_a_checkpoint is None) == (resume_path is None):
            raise ValueError("Provide exactly one of stage_a_checkpoint or resume_path")
        self.held_out_forger = held_out_forger
        stage_b_workers = self.config.train.stage_b_workers_per_loader
        stage_a_loaders = {
            "t1_1v1": self.loader("train", "t1_1v1", train=True, held_out_forger=held_out_forger,
                                   batch_size=self.config.train.stage_a_batch_t1_1v1,
                                   num_workers=stage_b_workers),
            "t1_5v1": self.loader("train", "t1_5v1", train=True, held_out_forger=held_out_forger,
                                   batch_size=self.config.train.stage_a_batch_t1_5v1,
                                   num_workers=stage_b_workers),
        }
        stage_b_loaders = {
            protocol: self.loader(
                "train", protocol, train=True, held_out_forger=held_out_forger,
                num_workers=stage_b_workers,
            )
            for protocol in self._stage_b_protocols()
        }
        balanced_a_loader = (
            self.loader(
                "train", "t2_a", train=True, held_out_forger=held_out_forger,
                num_workers=stage_b_workers,
            )
            if self.config.train.t2_balanced_a_enabled else None
        )
        balanced_b_loader = (
            self.loader(
                "train", "t2_b", train=True, held_out_forger=held_out_forger,
                num_workers=stage_b_workers,
            )
            if self.config.train.t2_balanced_b_enabled else None
        )
        stage_a_steps = math.ceil(
            (len(stage_a_loaders["t1_1v1"]) + len(stage_a_loaders["t1_5v1"]))
            / self.config.train.stage_a_grad_accumulation
        )
        stage_b_steps = math.ceil(max(map(len, stage_b_loaders.values())) / self.config.train.grad_accumulation)
        t2_a_steps = (
            math.ceil(len(balanced_a_loader) / self.config.train.t2_a_grad_accumulation)
            if balanced_a_loader is not None else 0
        )
        t2_b_steps = (
            math.ceil(len(balanced_b_loader) / self.config.train.t2_b_grad_accumulation)
            if balanced_b_loader is not None else 0
        )
        total_steps = (
            self.config.train.stage_a_epochs * stage_a_steps
            + self.config.train.t2_adapter_distillation_steps
            + self.config.train.t2_a_pretrain_epochs * t2_a_steps
            + self.config.train.t2_b_pretrain_epochs * t2_b_steps
            + self.config.train.stage_b_epochs * stage_b_steps
        )
        scheduler = self._scheduler(total_steps)
        history_path = self.output / "history.jsonl"
        if history_path.exists():
            self.history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()]
        test_history_path = self.output / "test_history.jsonl"
        if test_history_path.exists():
            self.test_history = [
                json.loads(line)
                for line in test_history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        unified_t1_baseline_eer: dict[str, float] = {}
        unified_t1_baseline_validation: dict[str, Any] = {}
        best_test_diagnostic_score = -float("inf")

        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            self._validate_checkpoint_config(checkpoint)
            stage = checkpoint.get("stage")
            if stage not in {"stage_b", "t2_adapter_distillation", "t2_a", "t2_b"}:
                raise ValueError(
                    "T2 resume requires an adapter, t2_a, t2_b, or stage_b checkpoint "
                    "written by train-stage-b"
                )
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            global_epoch = int(checkpoint.get("global_epoch", checkpoint["epoch"])) + 1
            state = checkpoint.get("training_state", {})
            if stage in {"t2_adapter_distillation", "t2_a", "t2_b"}:
                start_epoch = 0
                best_score, stale = -float("inf"), 0
                best_rank_score = best_open_score = best_diagnostic_score = -float("inf")
                best_test_diagnostic_score = -float("inf")
                best_a_score = float(state.get("best_a_balanced_accuracy", -float("inf")))
                best_a_threshold = float(state.get("best_a_threshold", 0.5))
                best_b_score = float(state.get("best_b_balanced_accuracy", -float("inf")))
                anchor_digest = self._frozen_t1_digest()
                stage_a_sf_far = self.evaluate_split(
                    "val", protocols=("t1_5v1",),
                )["t1_5v1"].get("genuine_vs_SF", {}).get("far", 1.0)
                if stage == "t2_adapter_distillation":
                    if self.config.train.t2_balanced_a_enabled:
                        global_epoch, best_a_score, best_a_threshold, _ = self._run_t2_a_pretraining(
                            balanced_a_loader, scheduler, global_epoch,
                        )
                    if self.config.train.t2_balanced_b_enabled:
                        global_epoch, best_b_score, _ = self._run_t2_b_pretraining(
                            balanced_b_loader, scheduler, global_epoch,
                            best_a_score, best_a_threshold,
                        )
                elif stage == "t2_a":
                    if not self.config.train.t2_balanced_a_enabled:
                        raise ValueError("A t2_a checkpoint requires t2_balanced_a_enabled=true")
                    a_start_epoch = int(checkpoint["stage_epoch"]) + 1
                    global_epoch, best_a_score, best_a_threshold, _ = self._run_t2_a_pretraining(
                        balanced_a_loader, scheduler, global_epoch,
                        start_epoch=a_start_epoch,
                        best_score=best_a_score,
                        best_threshold=best_a_threshold,
                        stale=int(state.get("stale", 0)),
                    )
                    best_b_score = -float("inf")
                    if self.config.train.t2_balanced_b_enabled:
                        global_epoch, best_b_score, _ = self._run_t2_b_pretraining(
                            balanced_b_loader, scheduler, global_epoch,
                            best_a_score, best_a_threshold,
                        )
                else:
                    if not self.config.train.t2_balanced_b_enabled:
                        raise ValueError("A t2_b checkpoint requires t2_balanced_b_enabled=true")
                    b_start_epoch = int(checkpoint["stage_epoch"]) + 1
                    global_epoch, best_b_score, _ = self._run_t2_b_pretraining(
                        balanced_b_loader, scheduler, global_epoch,
                        best_a_score, best_a_threshold,
                        start_epoch=b_start_epoch,
                        best_score=best_b_score,
                        stale=int(state.get("stale", 0)),
                    )
                print(
                    f"resumed {stage}; continuing with stage_b",
                    flush=True,
                )
            else:
                start_epoch = int(checkpoint["stage_epoch"]) + 1
                best_score = float(state.get("best_score", -float("inf")))
                best_rank_score = float(state.get("best_rank_score", -float("inf")))
                best_open_score = float(state.get("best_open_score", -float("inf")))
                best_diagnostic_score = float(state.get("best_diagnostic_score", -float("inf")))
                best_test_diagnostic_score = float(
                    state.get("best_test_diagnostic_score", -float("inf"))
                )
                stale = int(state.get("stale", 0))
                stage_a_sf_far = float(state.get("stage_a_sf_far", 1.0))
                best_a_score = float(state.get("best_a_balanced_accuracy", -float("inf")))
                best_a_threshold = float(state.get("best_a_threshold", 0.5))
                best_b_score = float(state.get("best_b_balanced_accuracy", -float("inf")))
                unified_t1_baseline_eer = {
                    key: float(value)
                    for key, value in state.get("unified_t1_baseline_eer", {}).items()
                }
                anchor_digest = state.get("frozen_t1_digest", self._frozen_t1_digest())
                if reset_patience:
                    best_score = -float("inf")
                    stale = 0
                    print("reset stage_b selection baseline after training-policy change", flush=True)
                print(
                    f"resumed stage_b at epoch={start_epoch + 1} "
                    f"optimizer_step={scheduler.last_epoch}", flush=True,
                )
        else:
            if self.config.model.variant in {
                "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
            }:
                initialized = (
                    self.initialize_unified_scratch_stage_b(stage_a_checkpoint)
                    if self.config.train.t2_scratch_from_stage_a
                    else self.initialize_unified_stage_b(stage_a_checkpoint)
                )
                checkpoint = initialized["checkpoint"]
            elif self.config.model.variant == "unified_sen_v20":
                initialized = self.initialize_sen_stage_b(stage_a_checkpoint)
                checkpoint = initialized["checkpoint"]
            elif self.config.train.stage_b_strict_t1_isolation:
                initialized = self.initialize_isolated_stage_b(stage_a_checkpoint)
                checkpoint = initialized["checkpoint"]
            else:
                checkpoint = torch.load(stage_a_checkpoint, map_location=self.device, weights_only=False)
                self._validate_checkpoint_config(checkpoint)
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.last_epoch = self.config.train.stage_a_epochs * stage_a_steps
            scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]
            if self.config.model.variant not in {
                "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
                "unified_abc_v14",
            }:
                for index, group in enumerate(self.optimizer.param_groups):
                    if group.get("name") in {"convnext_backbone", "shared_encoder"}:
                        group["lr"] *= 0.3
                        scheduler.base_lrs[index] *= 0.3
            start_epoch = 0
            global_epoch = self.config.train.stage_a_epochs
            best_score, stale = -float("inf"), 0
            best_rank_score = best_open_score = best_diagnostic_score = -float("inf")
            best_test_diagnostic_score = -float("inf")
            best_a_score, best_a_threshold = -float("inf"), 0.5
            best_b_score = -float("inf")
            anchor_digest = self._frozen_t1_digest()
            stage_a_sf_far = checkpoint.get("validation", {}).get("t1_5v1", {}).get(
                "genuine_vs_SF", {}
            ).get("far")
            if stage_a_sf_far is None:
                stage_a_sf_far = self.evaluate_split("val", protocols=("t1_5v1",))["t1_5v1"].get(
                    "genuine_vs_SF", {}
                ).get("far", 1.0)
            if self.config.model.variant in {
                "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                "unified_sen_v20",
            }:
                t1_baseline = self.evaluate_split(
                    "val", protocols=("t1_1v1", "t1_5v1"),
                )
                unified_t1_baseline_validation = t1_baseline
                unified_t1_baseline_eer = {
                    protocol: float(t1_baseline[protocol]["overall"]["eer"])
                    for protocol in ("t1_1v1", "t1_5v1")
                }
            source_path = self.output / "stage_b_source.json"
            source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
            source.update({
                "stage_a_checkpoint": str(Path(stage_a_checkpoint).resolve()),
                "stage_a_checkpoint_epoch": checkpoint.get("epoch"),
                "stage_a_sf_far": stage_a_sf_far,
                "frozen_t1_digest": anchor_digest,
                "unified_t1_baseline_eer": unified_t1_baseline_eer,
            })
            atomic_json(source_path, source)
            if self.config.model.variant in {
                "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
            } and not self.config.train.t2_scratch_from_stage_a:
                self._run_t2_adapter_distillation(stage_b_loaders["t2"], scheduler)
            if self.config.train.t2_balanced_a_enabled:
                if self.config.train.t2_a_pretrain_epochs < 1:
                    raise ValueError(
                        "t2_balanced_a_enabled requires t2_a_pretrain_epochs >= 1 for a fresh run"
                    )
                global_epoch, best_a_score, best_a_threshold, _ = self._run_t2_a_pretraining(
                    balanced_a_loader, scheduler, global_epoch,
                )
            if self.config.train.t2_balanced_b_enabled:
                if self.config.train.t2_b_pretrain_epochs < 1:
                    raise ValueError(
                        "t2_balanced_b_enabled requires t2_b_pretrain_epochs >= 1 for a fresh run"
                    )
                global_epoch, best_b_score, _ = self._run_t2_b_pretraining(
                    balanced_b_loader, scheduler, global_epoch,
                    best_a_score, best_a_threshold,
                )
            if self.config.model.variant == "v10" and self.config.train.t2_initialization_checkpoint:
                rank_anchor = torch.load(
                    self.config.train.t2_initialization_checkpoint,
                    map_location=self.device, weights_only=False,
                )
                rank_validation = rank_anchor.get("validation", {})
                rank_t2 = rank_validation.get("t2", {})
                rank_present = rank_t2.get("source_present", {})
                if "rank_1" in rank_present and "mrr" in rank_present:
                    best_rank_score = float(rank_present["rank_1"]) + 0.25 * float(rank_present["mrr"])
                    self._checkpoint(
                        "best_rank.pt", global_epoch, rank_validation,
                        stage="stage_b", stage_epoch=-1, scheduler=scheduler,
                        training_state={
                            "best_rank_score": best_rank_score,
                            "frozen_t1_digest": anchor_digest,
                            "training_phase": "rank_anchor",
                        },
                    )
                    print(
                        f"preserved V10 rank initialization as best_rank.pt score={best_rank_score:.6f}",
                        flush=True,
                    )
            print(f"starting stage_b from {stage_a_checkpoint} optimizer_step={scheduler.last_epoch}", flush=True)

        if self.config.model.variant in {
            "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
            "unified_sen_v20",
        } and (
            not unified_t1_baseline_eer or not unified_t1_baseline_validation
        ):
            t1_baseline = self.evaluate_split(
                "val", protocols=("t1_1v1", "t1_5v1"),
            )
            unified_t1_baseline_validation = t1_baseline
            unified_t1_baseline_eer = {
                protocol: float(t1_baseline[protocol]["overall"]["eer"])
                for protocol in ("t1_1v1", "t1_5v1")
            }

        if start_epoch >= self.config.train.stage_b_epochs:
            raise ValueError("Stage B is already complete for the configured epoch count")
        run_epochs = epochs if epochs is not None else self.config.train.stage_b_epochs - start_epoch
        stop_epoch = min(self.config.train.stage_b_epochs, start_epoch + run_epochs)
        last_validation: dict[str, Any] = {}
        for epoch in range(start_epoch, stop_epoch):
            if (
                self.config.model.variant == "unified_abc_v14"
                and epoch == self.config.train.t2_rank_recovery_epochs
                and (self.output / "best_rank.pt").exists()
            ):
                rank_checkpoint = torch.load(
                    self.output / "best_rank.pt", map_location=self.device, weights_only=False,
                )
                self.model.load_state_dict(rank_checkpoint["model"])
                print("loaded V1.4 best_rank.pt before B-only calibration", flush=True)
            if (
                self.config.model.variant in {"v7", "v8", "v10"}
                and epoch == self.config.train.t2_rank_phase_epochs
                and (self.output / "best_rank.pt").exists()
            ):
                rank_checkpoint = torch.load(
                    self.output / "best_rank.pt", map_location=self.device, weights_only=False,
                )
                self.model.load_state_dict(rank_checkpoint["model"])
                print("loaded best_rank.pt and froze the ranking branch", flush=True)
            self._freeze_early_image(False)
            self._prepare_stage_b_modules(epoch)
            pruned_optimizer_states = self._prune_v10_frozen_optimizer_state()
            if pruned_optimizer_states:
                print(
                    f"pruned {pruned_optimizer_states} frozen V10 optimizer states at phase boundary",
                    flush=True,
                )
            release_alpha, release_beta = self._set_t2_release(epoch)
            train_log = self._stage_b_epoch(
                stage_b_loaders, scheduler, epoch,
                balanced_a_loader=balanced_a_loader,
                balanced_b_loader=balanced_b_loader,
            )
            frozen_t1_unchanged = self._frozen_t1_digest() == anchor_digest
            if self.config.train.stage_b_strict_t1_isolation and not frozen_t1_unchanged:
                raise RuntimeError("Frozen encoder/T1 state changed during isolated Stage B training")
            validation = self.evaluate_split(
                "val",
                protocols=("t2",) if self.config.model.variant in {
                    "t2_abc_v1", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                } else (
                    "t1_1v1", "t1_5v1", "t2",
                ),
            )
            if self.config.model.variant in {
                "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
            }:
                validation.update(unified_t1_baseline_validation)
            if self.config.train.t2_balanced_a_enabled:
                current_a_threshold, current_a_metrics = self.evaluate_balanced_a("val")
                validation["t2_a_balanced"] = current_a_metrics
            else:
                current_a_threshold, current_a_metrics = 0.5, None
            if (
                self.config.train.t2_balanced_b_enabled
                or self.config.train.t2_factor_b_evaluation_enabled
            ):
                current_b_metrics = self.evaluate_balanced_b("val")
                validation["t2_b_balanced"] = current_b_metrics
            else:
                current_b_metrics = None
            test_diagnostic: dict[str, Any] | None = None
            test_diagnostic_score = -float("inf")
            is_best_test_diagnostic = False
            if self.config.train.evaluate_test_each_epoch:
                test_diagnostic = self.evaluate_split("test", protocols=("t2",))
                if self.config.train.t2_balanced_a_enabled:
                    _, test_a_metrics = self.evaluate_balanced_a(
                        "test", threshold=current_a_threshold,
                    )
                    test_diagnostic["t2_a_balanced"] = test_a_metrics
                if self.config.train.t2_balanced_b_enabled:
                    test_diagnostic["t2_b_balanced"] = self.evaluate_balanced_b("test")
                test_diagnostic_score = self._selection_score(
                    test_diagnostic, self.config.train.selection_policy,
                )
                is_best_test_diagnostic = test_diagnostic_score > best_test_diagnostic_score
                if is_best_test_diagnostic:
                    best_test_diagnostic_score = test_diagnostic_score
                self.test_history.append({
                    "stage": "stage_b",
                    "epoch": global_epoch,
                    "stage_epoch": epoch,
                    "metrics": test_diagnostic,
                    "diagnostic_score": test_diagnostic_score,
                    "selection_role": "diagnostic_only",
                    "timestamp": time.time(),
                })
                write_jsonl(test_history_path, self.test_history)
                atomic_json(self.output / "latest_test_diagnostic.json", {
                    "epoch": global_epoch,
                    "stage_epoch": epoch,
                    "metrics": test_diagnostic,
                    "diagnostic_score": test_diagnostic_score,
                    "selection_role": "diagnostic_only",
                    "test_used_for_checkpoint_selection": False,
                })
            score = self._selection_score(validation, self.config.train.selection_policy)
            sf_far = (
                stage_a_sf_far if self.config.model.variant == "t2_abc_v1" else
                validation["t1_5v1"].get("genuine_vs_SF", {}).get("far", 1.0)
            )
            phase = self._stage_b_phase(epoch)
            if self.config.train.selection_policy == "dual_evidence_v10":
                eligible, gate_report = self._v9_checkpoint_eligible(
                    validation, frozen_t1_unchanged, self.config,
                )
                eligible = eligible and phase in {"open", "calibration"}
                gate_report["eligible"] = eligible
                gate_report["phase"] = phase
            elif self.config.train.selection_policy == "unified_evidence_v9":
                eligible, gate_report = self._v9_checkpoint_eligible(
                    validation, frozen_t1_unchanged, self.config,
                )
                eligible = eligible and phase in {"unified", "calibration"}
                gate_report["eligible"] = eligible
                gate_report["phase"] = phase
            elif self.config.train.selection_policy == "stateful_bayesian_v8":
                eligible, gate_report = self._v8_checkpoint_eligible(
                    validation, frozen_t1_unchanged, self.config,
                )
                eligible = eligible and phase == "open"
                gate_report["eligible"] = eligible
                gate_report["phase"] = phase
            elif self.config.train.selection_policy == "progressive_bayesian_v7":
                eligible, gate_report = self._v7_checkpoint_eligible(
                    validation, frozen_t1_unchanged, self.config,
                )
                eligible = eligible and phase == "open"
                gate_report["eligible"] = eligible
                gate_report["phase"] = phase
            elif self.config.train.selection_policy == "isolated_bayesian_v6":
                eligible, gate_report = self._v6_checkpoint_eligible(
                    validation, frozen_t1_unchanged, self.config,
                )
            elif self.config.model.variant in {
                "unified_abc_v11", "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                "unified_sen_v20",
            }:
                max_increase = self.config.train.unified_t1_max_eer_increase
                t1_checks = {
                    protocol: (
                        validation[protocol]["overall"]["eer"]
                        <= unified_t1_baseline_eer[protocol] + max_increase
                    )
                    for protocol in ("t1_1v1", "t1_5v1")
                }
                eligible = (
                    frozen_t1_unchanged and all(t1_checks.values())
                    if self.config.model.variant in {
                        "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                    }
                    else phase == "joint" and all(t1_checks.values())
                )
                gate_report = {
                    "eligible": eligible,
                    "phase": phase,
                    "t1_eer_preserved": t1_checks,
                    "t1_baseline_eer": unified_t1_baseline_eer,
                    "t1_max_eer_increase": max_increase,
                    "single_shared_encoder": self.model.t2_encoder is None,
                    "t2_adapter": self.model.t2_adapter is not None,
                }
                if self.config.train.t2_balanced_a_enabled:
                    a_floor = best_a_score - self.config.train.t2_a_max_balanced_accuracy_drop
                    a_preserved = (
                        current_a_metrics["balanced_accuracy"] >= a_floor
                        and current_a_metrics["rf_recall"] >= self.config.train.t2_a_min_class_recall
                        and current_a_metrics["sf_recall"] >= self.config.train.t2_a_min_class_recall
                    )
                    eligible = eligible and a_preserved
                    gate_report["a_factor_preserved"] = a_preserved
                    gate_report["a_required_balanced_accuracy"] = a_floor
                if self.config.train.t2_balanced_b_enabled:
                    b_floor = best_b_score - self.config.train.t2_b_max_balanced_accuracy_drop
                    b_preserved = (
                        current_b_metrics["accuracy"] >= b_floor
                        and current_b_metrics["present_recall"] >= self.config.train.t2_b_min_class_recall
                        and current_b_metrics["absent_recall"] >= self.config.train.t2_b_min_class_recall
                    )
                    eligible = eligible and b_preserved
                    gate_report["b_factor_preserved"] = b_preserved
                    gate_report["b_required_balanced_accuracy"] = b_floor
                if self.config.model.variant in {"unified_abc_v14", "unified_sen_v20"}:
                    source_present = validation["t2"]["source_present"]
                    rank_checks = {
                        "rank_1": (
                            source_present["rank_1"]
                            >= self.config.train.checkpoint_min_rank_1
                        ),
                        "rank_3": (
                            source_present["rank_3"]
                            >= self.config.train.checkpoint_min_rank_3
                        ),
                    }
                    required_phase = (
                        "joint"
                        if self.config.train.t2_scratch_from_stage_a
                        else "b_calibration"
                        if self.config.model.variant == "unified_abc_v14"
                        else "joint"
                    )
                    eligible = eligible and phase == required_phase and all(rank_checks.values())
                    gate_report["rank_preserved"] = rank_checks
                    gate_report["required_rank_1"] = self.config.train.checkpoint_min_rank_1
                    gate_report["required_rank_3"] = self.config.train.checkpoint_min_rank_3
                gate_report["eligible"] = eligible
            elif self.config.model.variant == "t2_abc_v1":
                eligible = frozen_t1_unchanged
                gate_report = {
                    "eligible": eligible,
                    "frozen_t1_unchanged": frozen_t1_unchanged,
                    "selection_policy": self.config.train.selection_policy,
                    "factor_performance_gates_enabled": (
                        self.config.train.checkpoint_factor_gates_enabled
                    ),
                }
                if (
                    self.config.train.checkpoint_factor_gates_enabled
                    and self.config.train.t2_balanced_a_enabled
                ):
                    minimum_score = (
                        best_a_score - self.config.train.t2_a_max_balanced_accuracy_drop
                    )
                    score_preserved = current_a_metrics["balanced_accuracy"] >= minimum_score
                    recalls_preserved = (
                        current_a_metrics["rf_recall"] >= self.config.train.t2_a_min_class_recall
                        and current_a_metrics["sf_recall"] >= self.config.train.t2_a_min_class_recall
                    )
                    eligible = eligible and score_preserved and recalls_preserved
                    gate_report.update({
                        "eligible": eligible,
                        "balanced_a_enabled": True,
                        "a_balanced_accuracy": current_a_metrics["balanced_accuracy"],
                        "a_required_balanced_accuracy": minimum_score,
                        "a_rf_recall": current_a_metrics["rf_recall"],
                        "a_sf_recall": current_a_metrics["sf_recall"],
                        "a_min_class_recall": self.config.train.t2_a_min_class_recall,
                        "a_score_preserved": score_preserved,
                        "a_recalls_preserved": recalls_preserved,
                    })
                if (
                    self.config.train.checkpoint_factor_gates_enabled
                    and self.config.train.t2_balanced_b_enabled
                ):
                    b_minimum_score = (
                        best_b_score - self.config.train.t2_b_max_balanced_accuracy_drop
                    )
                    b_score_preserved = current_b_metrics["accuracy"] >= b_minimum_score
                    b_recalls_preserved = (
                        current_b_metrics["present_recall"] >= self.config.train.t2_b_min_class_recall
                        and current_b_metrics["absent_recall"] >= self.config.train.t2_b_min_class_recall
                    )
                    eligible = eligible and b_score_preserved and b_recalls_preserved
                    gate_report.update({
                        "eligible": eligible,
                        "balanced_b_enabled": True,
                        "b_balanced_accuracy": current_b_metrics["accuracy"],
                        "b_required_balanced_accuracy": b_minimum_score,
                        "b_present_recall": current_b_metrics["present_recall"],
                        "b_absent_recall": current_b_metrics["absent_recall"],
                        "b_min_class_recall": self.config.train.t2_b_min_class_recall,
                        "b_score_preserved": b_score_preserved,
                        "b_recalls_preserved": b_recalls_preserved,
                    })
            else:
                eligible = sf_far <= stage_a_sf_far + 0.05
                gate_report = {"eligible": eligible, "sf_far_preserved": eligible}
            if score > best_score and eligible:
                best_score, stale = score, 0
                is_best = True
            else:
                scratch_selection_active = (
                    self.config.train.t2_scratch_from_stage_a and phase == "joint"
                )
                legacy_selection_active = (
                    not self.config.train.t2_scratch_from_stage_a
                    and (
                        self.config.model.variant != "unified_abc_v14"
                        or phase == "b_calibration"
                    )
                )
                if (
                    scratch_selection_active or legacy_selection_active
                ) and epoch + 1 >= self.config.train.checkpoint_gate_warmup_epochs:
                    stale += 1
                is_best = False
            t2 = validation["t2"]
            rank_score = (
                0.60 * t2["source_present"]["rank_1"]
                + 0.40 * t2["source_present"]["rank_3"]
                if self.config.model.variant in {
                    "unified_abc_v13", "unified_abc_v14", "unified_sen_v20",
                }
                else t2["source_present"]["rank_1"] + 0.25 * t2["source_present"]["mrr"]
            )
            subtype = t2["hierarchical_episode_type_accuracy"]
            open_score = (
                0.4 * t2["hierarchical_accuracy"]
                + 0.2 * subtype.get("source_present", 0.0)
                + 0.2 * subtype.get("source_absent", 0.0)
                + 0.2 * subtype.get("rf_no_source", 0.0)
            )
            rank_phase = (
                phase in {"rank", "representation"}
                or self.config.model.variant == "unified_abc_v13"
                or self.config.model.variant == "unified_sen_v20"
                or (
                    self.config.model.variant == "unified_abc_v14"
                    and phase in {"rank_recovery", "joint"}
                )
            )
            open_phase = phase in {"open", "unified", "calibration"}
            is_best_rank = rank_phase and rank_score > best_rank_score
            is_best_open = open_phase and open_score > best_open_score
            is_best_diagnostic = score > best_diagnostic_score
            if is_best_rank:
                best_rank_score = rank_score
            if is_best_open:
                best_open_score = open_score
            if is_best_diagnostic:
                best_diagnostic_score = score
            train_log.update({
                "release_alpha": release_alpha,
                "release_beta": release_beta,
                "frozen_t1_unchanged": frozen_t1_unchanged,
                "training_phase": phase,
                "pruned_optimizer_states": pruned_optimizer_states,
            })
            self._record("stage_b", global_epoch, train_log, validation, score)
            state = {
                "best_score": best_score,
                "best_rank_score": best_rank_score,
                "best_open_score": best_open_score,
                "best_diagnostic_score": best_diagnostic_score,
                "best_test_diagnostic_score": best_test_diagnostic_score,
                "stale": stale,
                "stage_a_sf_far": stage_a_sf_far,
                "t2_unknown_fraction": self.config.train.t2_unknown_fraction,
                "frozen_t1_digest": anchor_digest,
                "frozen_t1_unchanged": frozen_t1_unchanged,
                "checkpoint_gate": gate_report,
                "release_alpha": release_alpha,
                "release_beta": release_beta,
                "training_phase": phase,
                "best_a_balanced_accuracy": best_a_score,
                "best_a_threshold": best_a_threshold,
                "current_a_threshold": current_a_threshold,
                "best_b_balanced_accuracy": best_b_score,
                "unified_t1_baseline_eer": unified_t1_baseline_eer,
                "current_test_diagnostic": test_diagnostic,
            }
            self._checkpoint("last.pt", global_epoch, validation, stage="stage_b", stage_epoch=epoch,
                             scheduler=scheduler, training_state=state)
            if is_best:
                self._checkpoint("best.pt", global_epoch, validation, stage="stage_b", stage_epoch=epoch,
                                 scheduler=scheduler, training_state=state)
            if is_best_rank:
                self._checkpoint("best_rank.pt", global_epoch, validation, stage="stage_b", stage_epoch=epoch,
                                 scheduler=scheduler, training_state=state)
            if is_best_open:
                self._checkpoint("best_open.pt", global_epoch, validation, stage="stage_b", stage_epoch=epoch,
                                 scheduler=scheduler, training_state=state)
            if is_best_diagnostic:
                self._checkpoint("best_diagnostic.pt", global_epoch, validation, stage="stage_b", stage_epoch=epoch,
                                 scheduler=scheduler, training_state=state)
            if is_best_test_diagnostic:
                self._checkpoint(
                    "best_test_diagnostic.pt", global_epoch, validation,
                    stage="stage_b", stage_epoch=epoch, scheduler=scheduler,
                    training_state=state,
                )
                atomic_json(self.output / "best_test_diagnostic.json", {
                    "epoch": global_epoch,
                    "stage_epoch": epoch,
                    "metrics": test_diagnostic,
                    "diagnostic_score": test_diagnostic_score,
                    "selection_role": "diagnostic_only",
                    "test_used_for_checkpoint_selection": False,
                    "official_checkpoint": "best.pt",
                })
            global_epoch += 1
            last_validation = validation
            atomic_json(self.output / "stage_b_status.json", {
                "stage": "stage_b", "running": True,
                "completed_stage_b_epochs": epoch + 1, "stale": stale,
                "best_score": best_score, "last_selection_score": score,
                "best_rank_score": best_rank_score, "best_open_score": best_open_score,
                "best_diagnostic_score": best_diagnostic_score,
                "best_test_diagnostic_score": best_test_diagnostic_score,
                "training_phase": phase,
                "checkpoint_gate": gate_report,
                "release_alpha": release_alpha, "release_beta": release_beta,
                "best_a_balanced_accuracy": best_a_score,
                "best_a_threshold": best_a_threshold,
                "best_b_balanced_accuracy": best_b_score,
                "last_validation": validation,
                "last_test_diagnostic": test_diagnostic,
            })
            if stale >= self.config.train.patience and self.config.model.variant not in {"v7", "v8", "v9", "v10"}:
                break

        completed_epochs = sum(row.get("stage") == "stage_b" for row in self.history)
        result: dict[str, Any] = {
            "stage": "stage_b",
            "completed_stage_b_epochs": completed_epochs,
            "stale": stale,
            "best_score": best_score,
            "best_rank_score": best_rank_score,
            "best_open_score": best_open_score,
            "best_diagnostic_score": best_diagnostic_score,
            "best_test_diagnostic_score": best_test_diagnostic_score,
            "best_a_balanced_accuracy": best_a_score,
            "best_b_balanced_accuracy": best_b_score,
            "last_validation": last_validation,
            "running": False,
            "best_checkpoint_exists": (self.output / "best.pt").exists(),
            "best_rank_checkpoint_exists": (self.output / "best_rank.pt").exists(),
            "best_open_checkpoint_exists": (self.output / "best_open.pt").exists(),
            "best_test_diagnostic_checkpoint_exists": (
                self.output / "best_test_diagnostic.pt"
            ).exists(),
        }
        self._set_t2_relation_teacher(False)
        reached_end = completed_epochs >= self.config.train.stage_b_epochs or stale >= self.config.train.patience
        if finalize and reached_end and (self.output / "best.pt").exists():
            self.load_checkpoint(self.output / "best.pt")
            calibrator = self.fit_calibration()
            test = self.evaluate_split("test", calibrator=calibrator, export=True)
            atomic_json(self.output / "test_metrics.json", test)
            result["test"] = test
        atomic_json(self.output / "stage_b_status.json", result)
        return result

    def _task_gradients(
        self, loss: Tensor, shared: list[Tensor], head: list[Tensor], context: str,
    ) -> tuple[list[Tensor | None], list[Tensor | None]]:
        gradients = torch.autograd.grad(loss, [*shared, *head], allow_unused=True)
        for gradient in gradients:
            if gradient is not None:
                self._require_finite(gradient, f"{context} task gradients")
        split = len(shared)
        return list(gradients[:split]), list(gradients[split:])

    @staticmethod
    def _sum_gradients(first: list[Tensor | None], second: list[Tensor | None]) -> list[Tensor | None]:
        result: list[Tensor | None] = []
        for left, right in zip(first, second):
            if left is None:
                result.append(right)
            elif right is None:
                result.append(left)
            else:
                result.append(left + right)
        return result

    @staticmethod
    def _accumulate_gradients(parameters: list[Tensor], gradients: list[Tensor | None]) -> None:
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                parameter.grad = gradient if parameter.grad is None else parameter.grad + gradient

    @torch.inference_mode()
    def predict(self, loader: DataLoader, calibrator: Calibrator | None = None) -> list[dict[str, Any]]:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        for batch in loader:
            batch = _move(batch, self.device)
            with self._autocast():
                output = self.model(batch)
            if batch["protocol"] == "t2_a_query":
                for index, metadata in enumerate(batch["metadata"]):
                    rows.append({
                        **metadata,
                        "rf_probability": float(output["rf_probability"][index]),
                    })
            elif batch["protocol"].startswith("t1"):
                references = batch["set_index"].shape[1]
                raw_logit = output["case_logit"].float()
                score = (calibrator.calibrate_t1(raw_logit, references)
                         if calibrator else torch.sigmoid(raw_logit))
                pair_scores = torch.sigmoid(output["pair_logits"].float())
                for index, metadata in enumerate(batch["metadata"]):
                    row = {**metadata, "score": float(score[index]), "raw_logit": float(raw_logit[index]),
                           "pair_scores": pair_scores[index, batch["set_mask"][index]].float().cpu().tolist()}
                    if "reliability" in output:
                        row["reference_reliability"] = output["reliability"][
                            index, batch["set_mask"][index]
                        ].float().cpu().tolist()
                    rows.append(row)
            else:
                probabilities = (calibrator.calibrate_t2(
                    output["rank_logits"], output["exist_logit"], output.get("type_logits"),
                )
                                 if calibrator else output["joint_probability"])
                if calibrator and "type_logits" in output:
                    type_probability = torch.softmax(
                        output["type_logits"] / calibrator.t2_type_temperature, dim=-1,
                    )
                    exist_probability = type_probability[:, 0]
                else:
                    type_probability = output.get("type_probability")
                    exist_probability = (
                        torch.sigmoid(output["exist_logit"] / calibrator.t2_exist_temperature)
                        if calibrator else output["exist_probability"]
                    )
                exist_threshold = calibrator.t2_exist_threshold if calibrator else 0.5
                for index, metadata in enumerate(batch["metadata"]):
                    count = int(batch["set_mask"][index].sum())
                    row = {**metadata,
                                 "joint_probabilities": probabilities[index, :count + 1].float().cpu().tolist(),
                                 "rank_logits": output["rank_logits"][index, :count].float().cpu().tolist(),
                                 "rank_probabilities": output["rank_probability"][index, :count].float().cpu().tolist(),
                                 "exist_logit": float(output["exist_logit"][index]),
                                 "exist_probability": float(exist_probability[index]),
                                 "exist_threshold": exist_threshold}
                    if "pim_statistics" in output:
                        row["pim_statistics"] = (
                            output["pim_statistics"][index, :count].float().cpu().tolist()
                        )
                    if "rf_probability" in output:
                        row["rf_probability"] = float(output["rf_probability"][index])
                    if "in_set_probability" in output:
                        row["in_set_probability"] = float(output["in_set_probability"][index])
                    if "official_probability" in output:
                        row["official_probabilities"] = (
                            torch.cat([
                                output["official_probability"][index, :count],
                                output["official_probability"][index, -2:],
                            ])
                            .float().cpu().tolist()
                        )
                    if "t1_genuine_probability" in output:
                        row["t1_genuine_probability"] = float(
                            output["t1_genuine_probability"][index]
                        )
                        row["t1_forgery_probability"] = float(
                            output["t1_forgery_probability"][index]
                        )
                    if "match_logits" in output:
                        row["match_logits"] = output["match_logits"][index, :count].float().cpu().tolist()
                    if type_probability is not None:
                        row["type_probabilities"] = type_probability[index].float().cpu().tolist()
                    if "bayesian_stage_probability" in output:
                        row["bayesian_stage_probabilities"] = (
                            output["bayesian_stage_probability"][index].float().cpu().tolist()
                        )
                        row["bayesian_evidence_increment"] = (
                            output["bayesian_evidence_increment"][index].float().cpu().tolist()
                        )
                    if "stateful_prefix_probability" in output:
                        row["stateful_prefix_probabilities"] = (
                            output["stateful_prefix_probability"][index].float().cpu().tolist()
                        )
                        row["stateful_prefix_sizes"] = output["stateful_prefix_sizes"].cpu().tolist()
                    rows.append(row)
        return rows

    def evaluate_balanced_a(
        self, split: str, threshold: float | None = None, export: bool = False,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate factor A on one balanced row per unique Query."""
        rows = self.predict(self.loader(split, "t2_a"))
        selected_threshold = (
            best_balanced_a_threshold(rows) if threshold is None else float(threshold)
        )
        metrics: dict[str, Any] = balanced_a_metrics(rows, selected_threshold)
        if export:
            metrics["writer_bootstrap_balanced_accuracy"] = writer_bootstrap(
                rows,
                lambda subset: balanced_a_metrics(
                    subset, selected_threshold,
                )["balanced_accuracy"],
                repetitions=2000,
            )
            write_jsonl(self.output / f"predictions/{split}_t2_test_a.jsonl", rows)
        return selected_threshold, metrics

    def evaluate_balanced_b(self, split: str, export: bool = False) -> dict[str, Any]:
        """Evaluate factor B on an equal Present/Absent SF episode view."""
        rows = self.predict(self.loader(split, "t2_b"))
        metrics = conditional_t2_factor_metrics(rows)["B_source_in_pool_given_sf"]
        if export:
            metrics["writer_bootstrap_accuracy"] = writer_bootstrap(
                rows,
                lambda subset: conditional_t2_factor_metrics(subset)[
                    "B_source_in_pool_given_sf"
                ]["accuracy"],
                repetitions=2000,
            )
            write_jsonl(self.output / f"predictions/{split}_t2_test_b.jsonl", rows)
        return {**metrics, "n_episodes": len(rows)}

    def evaluate_t2_factor_views(
        self, split: str, a_threshold: float, export: bool = False,
    ) -> dict[str, Any]:
        """Report balanced A/B decisions and Present-only C ranking separately."""
        _, a_metrics = self.evaluate_balanced_a(split, a_threshold, export=export)
        b_metrics = self.evaluate_balanced_b(split, export=export)
        c_rows = self.predict(self.loader(split, "t2_c"))
        c_metrics = t2_metrics(c_rows)["source_present"]
        if export:
            c_metrics["writer_bootstrap_rank_1"] = writer_bootstrap(
                c_rows,
                lambda subset: t2_metrics(subset)["source_present"]["rank_1"],
                repetitions=2000,
            )
            write_jsonl(self.output / f"predictions/{split}_t2_test_c.jsonl", c_rows)
        return {
            "test_A_balanced_unique_queries": a_metrics,
            "test_B_balanced_present_absent_given_SF": {
                **b_metrics,
            },
            "test_C_present_only_ranking": {
                **c_metrics,
                "n_episodes": len(c_rows),
            },
        }

    def evaluate_split(self, split: str, calibrator: Calibrator | None = None, export: bool = False,
                       protocols: tuple[str, ...] = ("t1_1v1", "t1_5v1", "t2")) -> dict[str, Any]:
        predictions = {}
        metrics = {}
        for protocol in protocols:
            rows = self.predict(self.loader(split, protocol), calibrator)
            predictions[protocol] = rows
            if protocol.startswith("t1"):
                references = 1 if protocol == "t1_1v1" else 5
                threshold = (calibrator.t1_1v1_threshold if references == 1 else calibrator.t1_5v1_threshold) if calibrator else 0.5
                metrics[protocol] = t1_metrics(rows, threshold)
                if export:
                    metrics[protocol]["writer_bootstrap_auc"] = writer_bootstrap(
                        rows, lambda subset: t1_metrics(subset, threshold)["overall"]["roc_auc"], repetitions=2000,
                    )
            else:
                metrics[protocol] = t2_metrics(rows)
                if self.config.model.variant in {
                    "t2_abc_v1", "unified_abc_v11", "unified_abc_v12", "unified_abc_v13",
                    "unified_abc_v14", "unified_sen_v20",
                }:
                    metrics[protocol]["conditional_factors"] = conditional_t2_factor_metrics(rows)
                if export:
                    metrics[protocol]["writer_bootstrap_joint_accuracy"] = writer_bootstrap(
                        rows, lambda subset: t2_metrics(subset)["joint_accuracy"], repetitions=2000,
                    )
            if export:
                write_jsonl(self.output / f"predictions/{split}_{protocol}.jsonl", rows)
        return metrics

    @staticmethod
    def _released_t2_rows(
        rows: list[dict[str, Any]], release_a: bool, release_b: bool, name: str,
    ) -> list[dict[str, Any]]:
        released: list[dict[str, Any]] = []
        for source in rows:
            row = dict(source)
            is_rf = row["episode_type"] == "rf_no_source"
            is_present = row["episode_type"] == "source_present"
            rf_probability = float(is_rf) if release_a else float(row["rf_probability"])
            if release_b:
                in_set_probability = float(is_present) if not is_rf else 0.0
            else:
                in_set_probability = float(row["in_set_probability"])
            rank_probability = np.asarray(row["rank_probabilities"], dtype=float)
            rank_probability /= rank_probability.sum()
            sf_probability = 1 - rf_probability
            present_probability = sf_probability * in_set_probability
            absent_probability = sf_probability * (1 - in_set_probability)
            unknown_probability = absent_probability + rf_probability
            row.update({
                "release_scenario": name,
                "rf_probability": rf_probability,
                "in_set_probability": in_set_probability,
                "exist_probability": present_probability,
                "exist_threshold": 0.5,
                "type_probabilities": [
                    present_probability, absent_probability, rf_probability,
                ],
                "joint_probabilities": [
                    *(present_probability * rank_probability).tolist(), unknown_probability,
                ],
            })
            released.append(row)
        return released

    def evaluate_t2_release(
        self, split: str, export: bool = True, a_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate E0, oracle-A, and oracle-A+B with one frozen checkpoint."""
        predicted = self.predict(self.loader(split, "t2"))
        missing = [
            row["episode_id"] for row in predicted
            if "rf_probability" not in row or "in_set_probability" not in row
        ]
        if missing:
            raise ValueError(
                "Progressive release evaluation requires explicit A/B probabilities; "
                f"missing for {missing[:3]}"
            )
        scenarios = {
            "E0_predicted_ABC": self._released_t2_rows(predicted, False, False, "E0_predicted_ABC"),
            "E1_oracle_A": self._released_t2_rows(predicted, True, False, "E1_oracle_A"),
            "E2_oracle_AB": self._released_t2_rows(predicted, True, True, "E2_oracle_AB"),
        }
        metrics: dict[str, Any] = {}
        for name, rows in scenarios.items():
            summary = t2_metrics(rows)
            summary["conditional_factors"] = conditional_t2_factor_metrics(rows)
            if export:
                summary["writer_bootstrap_joint_accuracy"] = writer_bootstrap(
                    rows, lambda subset: t2_metrics(subset)["joint_accuracy"], repetitions=2000,
                )
                write_jsonl(self.output / f"predictions/{split}_t2_{name}.jsonl", rows)
            metrics[name] = summary
        e0 = metrics["E0_predicted_ABC"]["joint_accuracy"]
        e1 = metrics["E1_oracle_A"]["joint_accuracy"]
        e2 = metrics["E2_oracle_AB"]["joint_accuracy"]
        report = {
            "split": split,
            "checkpoint_model_variant": self.config.model.variant,
            "scenario_definitions": {
                "E0_predicted_ABC": "A, B, and C are predicted",
                "E1_oracle_A": "ground-truth RF/SF is supplied; B and C are predicted",
                "E2_oracle_AB": "ground-truth RF/SF and source presence are supplied; C is predicted",
            },
            "metrics": metrics,
            "accuracy_deltas": {
                "oracle_A_gain_E1_minus_E0": e1 - e0,
                "oracle_B_gain_E2_minus_E1": e2 - e1,
                "ranking_ceiling_E2": e2,
            },
        }
        if self.config.train.t2_balanced_a_enabled:
            if a_threshold is None:
                a_threshold, _ = self.evaluate_balanced_a("val")
            report["factor_tests"] = self.evaluate_t2_factor_views(
                split, a_threshold, export=export,
            )
            report["factor_test_protocol"] = {
                "A": "one balanced row per unique Query; RF versus SF",
                "B": "balanced Present versus Absent pools, conditioned on SF",
                "C": "Present-only candidate ranking",
                "A_threshold_source": "validation balanced-accuracy optimum",
                "A_threshold": a_threshold,
            }
        if export:
            atomic_json(self.output / f"{split}_t2_release_metrics.json", report)
        return report

    def fit_calibration(self) -> Calibrator:
        calibrator = self.fit_t1_calibration()

        t2_loader = self.loader("val", "t2")
        rank_logits, exist_logits, rank_targets, exist_targets = [], [], [], []
        type_logits, type_targets = [], []
        self.model.eval()
        with torch.inference_mode():
            for batch in t2_loader:
                batch = _move(batch, self.device)
                output = self.model(batch)
                present = batch["target_index"] >= 0
                if present.any():
                    rank_logits.append(output["rank_logits"][present])
                    rank_targets.append(batch["target_index"][present])
                exist_logits.append(output["exist_logit"])
                exist_targets.append(batch["exist_label"])
                if "type_logits" in output:
                    type_logits.append(output["type_logits"])
                    type_targets.append(batch["episode_type_index"])
        calibrator.t2_rank_temperature = fit_multiclass_temperature(torch.cat(rank_logits), torch.cat(rank_targets))
        calibrator.t2_exist_temperature = fit_binary_temperature(torch.cat(exist_logits), torch.cat(exist_targets))
        if type_logits:
            joined_type_logits = torch.cat(type_logits)
            calibrator.t2_type_temperature = fit_multiclass_temperature(
                joined_type_logits, torch.cat(type_targets),
            )
            calibrated_exist = torch.softmax(
                joined_type_logits / calibrator.t2_type_temperature, dim=-1,
            )[:, 0].cpu().numpy()
        else:
            calibrated_exist = torch.sigmoid(
                torch.cat(exist_logits) / calibrator.t2_exist_temperature
            ).cpu().numpy()
        calibrator.t2_exist_threshold = eer_threshold(torch.cat(exist_targets).cpu().numpy(), calibrated_exist)
        atomic_json(self.output / "calibration.json", calibrator.__dict__)
        return calibrator

    def fit_t1_calibration(self) -> Calibrator:
        rows_1 = self.predict(self.loader("val", "t1_1v1"))
        rows_5 = self.predict(self.loader("val", "t1_5v1"))
        # Keep raw decision scores: probability saturation is irreversible.
        def arrays(rows: list[dict[str, Any]]) -> tuple[Tensor, Tensor]:
            logits = torch.tensor([row["raw_logit"] for row in rows], dtype=torch.float32, device=self.device)
            return logits, torch.tensor([row["label"] for row in rows], device=self.device)
        logits_1, labels_1 = arrays(rows_1)
        logits_5, labels_5 = arrays(rows_5)
        temperature_1 = fit_binary_temperature(logits_1, labels_1)
        temperature_5 = fit_binary_temperature(logits_5, labels_5)
        score_1 = torch.sigmoid(logits_1 / temperature_1).cpu().numpy()
        score_5 = torch.sigmoid(logits_5 / temperature_5).cpu().numpy()
        calibrator = Calibrator(
            t1_1v1_temperature=temperature_1, t1_5v1_temperature=temperature_5,
            t1_1v1_threshold=eer_threshold(labels_1.cpu().numpy(), score_1),
            t1_5v1_threshold=eer_threshold(labels_5.cpu().numpy(), score_5),
        )
        atomic_json(self.output / "t1_calibration.json", calibrator.__dict__)
        return calibrator

    @staticmethod
    def _stage_a_test_target_checks(
        test: dict[str, Any], train_config: Any,
    ) -> tuple[bool, dict[str, bool]]:
        targets = {
            "t1_1v1_accuracy": train_config.stage_a_test_target_1v1_accuracy,
            "t1_1v1_eer": train_config.stage_a_test_target_1v1_eer,
            "t1_5v1_accuracy": train_config.stage_a_test_target_5v1_accuracy,
            "t1_5v1_eer": train_config.stage_a_test_target_5v1_eer,
        }
        actual = {
            "t1_1v1_accuracy": test["t1_1v1"]["overall"]["accuracy"],
            "t1_1v1_eer": test["t1_1v1"]["overall"]["eer"],
            "t1_5v1_accuracy": test["t1_5v1"]["overall"]["accuracy"],
            "t1_5v1_eer": test["t1_5v1"]["overall"]["eer"],
        }
        checks = {
            key: (
                actual[key] <= target if key.endswith("_eer") else actual[key] >= target
            )
            for key, target in targets.items()
            if target is not None
        }
        return bool(checks) and all(checks.values()), checks

    @staticmethod
    def _stage_a_history_declined(history: list[dict[str, Any]]) -> bool:
        if len(history) < 2:
            return False
        return float(history[-1]["selection_score"]) < float(history[-2]["selection_score"])

    @staticmethod
    def _stage_a_selection_score(validation: dict[str, Any], policy: str = "t1_5v1") -> float:
        if policy == "dual_t1":
            mean_eer = 0.5 * (
                validation["t1_1v1"]["overall"]["eer"]
                + validation["t1_5v1"]["overall"]["eer"]
            )
            return 1 - mean_eer
        if policy == "t1_5v1":
            return 1 - validation["t1_5v1"]["overall"]["eer"]
        raise ValueError(f"Unsupported Stage A selection policy: {policy}")

    @staticmethod
    def _selection_score(validation: dict[str, Any], policy: str = "legacy_v2") -> float:
        if policy == "unified_sen_v20":
            t2 = validation["t2"]
            source = t2["source_present"]
            mean_t1_eer = 0.5 * (
                validation["t1_1v1"]["overall"]["eer"]
                + validation["t1_5v1"]["overall"]["eer"]
            )
            b_accuracy = validation.get("t2_b_balanced", {}).get("accuracy", 0.0)
            return (
                0.20 * (1 - mean_t1_eer)
                + 0.20 * t2["joint_accuracy"]
                + 0.35 * source["rank_1"]
                + 0.15 * source["rank_3"]
                + 0.10 * b_accuracy
            )
        if policy == "unified_abc_v14":
            t2 = validation["t2"]
            source = t2["source_present"]
            b_accuracy = validation.get("t2_b_balanced", {}).get("accuracy", 0.0)
            return (
                0.25 * t2["joint_accuracy"]
                + 0.35 * source["rank_1"]
                + 0.25 * source["rank_3"]
                + 0.10 * b_accuracy
                + 0.05 * source["mrr"]
            )
        if policy == "unified_abc_v13":
            t2 = validation["t2"]
            source = t2["source_present"]
            b_accuracy = validation.get("t2_b_balanced", {}).get("accuracy", 0.0)
            return (
                0.30 * t2["joint_accuracy"]
                + 0.30 * source["rank_1"]
                + 0.20 * source["rank_3"]
                + 0.15 * b_accuracy
                + 0.05 * source["mrr"]
            )
        if policy == "unified_abc_v12":
            t2 = validation["t2"]
            b_accuracy = validation.get("t2_b_balanced", {}).get("accuracy", 0.0)
            return (
                0.45 * t2["joint_accuracy"]
                + 0.25 * t2["source_present"]["rank_1"]
                + 0.10 * t2["source_present"]["mrr"]
                + 0.20 * b_accuracy
            )
        if policy == "unified_abc_v11":
            mean_t1_eer = 0.5 * (
                validation["t1_1v1"]["overall"]["eer"]
                + validation["t1_5v1"]["overall"]["eer"]
            )
            t2 = validation["t2"]
            t2_score = (
                0.55 * t2["joint_accuracy"]
                + 0.30 * t2["source_present"]["rank_1"]
                + 0.15 * t2["source_present"]["mrr"]
            )
            return 0.45 * (1 - mean_t1_eer) + 0.55 * t2_score
        if policy == "conditional_abc_v1":
            return float(validation["t2"]["joint_accuracy"])
        if policy in {"unified_evidence_v9", "dual_evidence_v10"}:
            t2 = validation["t2"]
            subtype = t2["hierarchical_episode_type_accuracy"]
            stateful = t2.get("stateful_progression", {})
            information_gain = max(0.0, float(stateful.get("final_information_gain_nll", 0.0)))
            return (
                0.35 * t2["hierarchical_accuracy"]
                + 0.25 * subtype.get("source_present", 0.0)
                + 0.15 * t2["source_present"]["rank_1"]
                + 0.05 * t2["source_present"]["mrr"]
                + 0.10 * subtype.get("source_absent", 0.0)
                + 0.10 * subtype.get("rf_no_source", 0.0)
                + 0.05 * min(information_gain, 1.0)
                + min(0.0, t2["collapse_margin"])
            )
        if policy == "stateful_bayesian_v8":
            t2 = validation["t2"]
            subtype = t2["hierarchical_episode_type_accuracy"]
            stateful = t2.get("stateful_progression", {})
            information_gain = max(0.0, float(stateful.get("final_information_gain_nll", 0.0)))
            return (
                0.28 * t2["hierarchical_accuracy"]
                + 0.24 * t2["source_present"]["rank_1"]
                + 0.10 * t2["source_present"]["mrr"]
                + 0.14 * subtype.get("source_absent", 0.0)
                + 0.14 * subtype.get("rf_no_source", 0.0)
                + 0.10 * min(information_gain, 1.0)
                + min(0.0, t2["collapse_margin"])
            )
        if policy == "progressive_bayesian_v7":
            t2 = validation["t2"]
            subtype = t2["hierarchical_episode_type_accuracy"]
            progression = t2.get("bayesian_progression", {})
            information_gain = max(0.0, float(progression.get("information_gain_nll", 0.0)))
            return (
                0.30 * t2["hierarchical_accuracy"]
                + 0.25 * t2["source_present"]["rank_1"]
                + 0.10 * t2["source_present"]["mrr"]
                + 0.15 * subtype.get("source_absent", 0.0)
                + 0.15 * subtype.get("rf_no_source", 0.0)
                + 0.05 * min(information_gain, 1.0)
                + min(0.0, t2["collapse_margin"])
            )
        if policy == "isolated_bayesian_v6":
            t2 = validation["t2"]
            subtype = t2["hierarchical_episode_type_accuracy"]
            return (
                0.30 * t2["hierarchical_accuracy"]
                + 0.25 * t2["source_present"]["rank_1"]
                + 0.15 * t2["source_present"]["mrr"]
                + 0.15 * subtype.get("source_absent", 0.0)
                + 0.15 * subtype.get("rf_no_source", 0.0)
                + min(0.0, t2["collapse_margin"])
            )
        if policy == "source_retrieval_v5":
            t2 = validation["t2"]
            score = (
                0.45 * t2["hierarchical_accuracy"]
                + 0.30 * t2["source_present"]["rank_1"]
                + 0.15 * t2["source_present"]["mrr"]
                + 0.10 * t2["existence"]["balanced_accuracy"]
            )
            return score + min(0.0, t2["collapse_margin"])
        if policy == "open_set_v5":
            t2 = validation["t2"]
            score = (
                0.35 * t2["hierarchical_episode_type_macro_accuracy"]
                + 0.20 * t2["hierarchical_accuracy"]
                + 0.25 * t2["source_present"]["rank_1"]
                + 0.10 * t2["source_present"]["mrr"]
                + 0.10 * t2["existence"]["balanced_accuracy"]
            )
            return score + min(0.0, t2["collapse_margin"])
        if policy in {"balanced_v3", "open_set_v4"}:
            t1_eer = 0.5 * (
                validation["t1_1v1"]["overall"]["eer"] + validation["t1_5v1"]["overall"]["eer"]
            )
            t2 = validation["t2"]
            if policy == "open_set_v4":
                t2_score = (
                    0.25 * t2["existence"]["balanced_accuracy"]
                    + 0.25 * t2["source_present"]["rank_1"]
                    + 0.15 * t2["source_present"]["mrr"]
                    + 0.35 * t2["hierarchical_accuracy"]
                )
                return 0.35 * (1 - t1_eer) + 0.65 * t2_score
            t2_score = (
                0.5 * t2["existence"]["balanced_accuracy"]
                + 0.25 * t2["source_present"]["rank_1"]
                + 0.25 * t2["source_present"]["mrr"]
            )
            return 0.5 * (1 - t1_eer) + 0.5 * t2_score
        if policy != "legacy_v2":
            raise ValueError(f"Unsupported selection policy: {policy}")
        t1_eer = validation["t1_5v1"]["overall"]["eer"]
        t2_macro_accuracy = validation["t2"]["episode_type_macro_accuracy"]
        return 0.5 * (1 - t1_eer) + 0.5 * t2_macro_accuracy

    @staticmethod
    def _v6_checkpoint_eligible(
        validation: dict[str, Any], frozen_t1_unchanged: bool, config: ExperimentConfig,
    ) -> tuple[bool, dict[str, Any]]:
        t2 = validation["t2"]
        subtype = t2["hierarchical_episode_type_accuracy"]
        checks = {
            "frozen_t1_unchanged": frozen_t1_unchanged,
            "positive_collapse_margin": t2["collapse_margin"] > 0,
            "hierarchical_accuracy": (
                t2["hierarchical_accuracy"] > config.train.checkpoint_min_hierarchical_accuracy
            ),
            "rank_1": t2["source_present"]["rank_1"] >= config.train.checkpoint_min_rank_1,
            "source_absent": (
                subtype.get("source_absent", 0.0)
                > config.train.checkpoint_min_source_absent_accuracy
            ),
            "rf_no_source": (
                subtype.get("rf_no_source", 0.0) >= config.train.checkpoint_min_rf_accuracy
            ),
        }
        return all(checks.values()), {"eligible": all(checks.values()), "checks": checks}

    @staticmethod
    def _v7_checkpoint_eligible(
        validation: dict[str, Any], frozen_t1_unchanged: bool, config: ExperimentConfig,
    ) -> tuple[bool, dict[str, Any]]:
        t2 = validation["t2"]
        subtype = t2["hierarchical_episode_type_accuracy"]
        progression = t2.get("bayesian_progression", {})
        checks = {
            "frozen_t1_unchanged": frozen_t1_unchanged,
            "positive_collapse_margin": t2["collapse_margin"] > 0,
            "hierarchical_accuracy": (
                t2["hierarchical_accuracy"] > config.train.checkpoint_min_hierarchical_accuracy
            ),
            "rank_1": t2["source_present"]["rank_1"] >= config.train.checkpoint_min_rank_1,
            "source_absent": (
                subtype.get("source_absent", 0.0)
                > config.train.checkpoint_min_source_absent_accuracy
            ),
            "rf_no_source": (
                subtype.get("rf_no_source", 0.0) >= config.train.checkpoint_min_rf_accuracy
            ),
            "bayesian_information_gain": (
                not config.train.checkpoint_require_bayesian_improvement
                or progression.get("information_gain_nll", -float("inf")) > 0
            ),
        }
        return all(checks.values()), {"eligible": all(checks.values()), "checks": checks}

    @staticmethod
    def _v8_checkpoint_eligible(
        validation: dict[str, Any], frozen_t1_unchanged: bool, config: ExperimentConfig,
    ) -> tuple[bool, dict[str, Any]]:
        eligible, report = Trainer._v7_checkpoint_eligible(validation, frozen_t1_unchanged, config)
        stateful = validation["t2"].get("stateful_progression", {})
        report["checks"]["stateful_information_gain"] = (
            stateful.get("final_information_gain_nll", -float("inf")) > 0
        )
        report["checks"]["stateful_final_matches_full_set"] = bool(
            stateful.get("final_matches_full_set", False)
        )
        report["eligible"] = eligible and all(report["checks"].values())
        return report["eligible"], report

    @staticmethod
    def _v9_checkpoint_eligible(
        validation: dict[str, Any], frozen_t1_unchanged: bool, config: ExperimentConfig,
    ) -> tuple[bool, dict[str, Any]]:
        t2 = validation["t2"]
        subtype = t2["hierarchical_episode_type_accuracy"]
        stateful = t2.get("stateful_progression", {})
        checks = {
            "frozen_t1_unchanged": frozen_t1_unchanged,
            "positive_collapse_margin": t2["collapse_margin"] > 0,
            "hierarchical_accuracy": (
                t2["hierarchical_accuracy"] > config.train.checkpoint_min_hierarchical_accuracy
            ),
            "rank_1": t2["source_present"]["rank_1"] >= config.train.checkpoint_min_rank_1,
            "source_present": subtype.get("source_present", 0.0) >= 0.25,
            "source_absent": (
                subtype.get("source_absent", 0.0) >= config.train.checkpoint_min_source_absent_accuracy
            ),
            "rf_no_source": subtype.get("rf_no_source", 0.0) >= config.train.checkpoint_min_rf_accuracy,
            "stateful_information_gain": (
                not config.train.checkpoint_require_bayesian_improvement
                or stateful.get("final_information_gain_nll", -float("inf")) > 0
            ),
            "stateful_final_matches_full_set": bool(
                stateful.get("final_matches_full_set", False)
            ),
        }
        return all(checks.values()), {"eligible": all(checks.values()), "checks": checks}

    def _record(self, stage: str, epoch: int, train: dict[str, Any], validation: dict[str, Any], score: float) -> None:
        row = {"stage": stage, "epoch": epoch, "train": train, "validation": validation,
               "selection_score": score, "timestamp": time.time()}
        self.history.append(row)
        write_jsonl(self.output / "history.jsonl", self.history)

    def _checkpoint(self, name: str, epoch: int, validation: dict[str, Any], stage: str | None = None,
                    stage_epoch: int | None = None, scheduler: LambdaLR | None = None,
                    training_state: dict[str, Any] | None = None) -> None:
        if self.config.model.variant in {
            "unified_abc_v12", "unified_abc_v13", "unified_abc_v14", "unified_sen_v20",
        }:
            bad_tensors = [
                key for key, value in self.model.state_dict().items()
                if value.is_floating_point() and not bool(torch.isfinite(value).all())
            ]
            if bad_tensors:
                raise FloatingPointError(
                    f"Refusing to write {name}; non-finite model tensors: {bad_tensors[:5]}"
                )

            def check_metrics(value: Any, path: str = "validation") -> list[str]:
                if isinstance(value, dict):
                    return [
                        item
                        for key, child in value.items()
                        for item in check_metrics(child, f"{path}.{key}")
                    ]
                if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    return [path]
                return []

            bad_metrics = check_metrics(validation)
            if bad_metrics:
                raise FloatingPointError(
                    f"Refusing to write {name}; non-finite validation metrics: {bad_metrics[:5]}"
                )
        payload = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "epoch": epoch, "global_epoch": epoch, "validation": validation,
                   "config": self.config.to_dict()}
        if stage is not None:
            payload["stage"] = stage
        if stage_epoch is not None:
            payload["stage_epoch"] = stage_epoch
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if training_state is not None:
            payload["training_state"] = training_state
        torch.save(payload, self.output / name)

    def initialize_from_checkpoint(self, path: str | Path) -> None:
        """Transfer the compatible representation and trained T1 modules into V4/V5."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved = checkpoint.get("config", {})
        saved_data, saved_model = saved.get("data", {}), saved.get("model", {})
        expected = (
            self.config.data.input_pipeline,
            self.config.model.sequence_stem,
            self.config.model.feature_dim,
            self.config.data.image_height,
            self.config.data.image_width,
        )
        actual = (
            saved_data.get("input_pipeline", "legacy_v1"),
            saved_model.get("sequence_stem", "legacy_stride8"),
            saved_model.get("feature_dim", 10),
            saved_data.get("image_height", 256),
            saved_data.get("image_width", 512),
        )
        if actual != expected:
            raise ValueError(
                "Initialization checkpoint representation does not match the target model: "
                f"checkpoint={actual}, configured={expected}"
            )
        prefixes = (
            "encoder.", "t1.global_adapter.", "t1.local_adapter.",
            "t1.relation.", "t1.qrsa.",
        )
        current = self.model.state_dict()
        transferred = {
            key: value for key, value in checkpoint["model"].items()
            if key.startswith(prefixes) and key in current and current[key].shape == value.shape
        }
        missing, unexpected = self.model.load_state_dict(transferred, strict=False)
        report = {
            "source": str(Path(path).resolve()),
            "source_epoch": checkpoint.get("epoch"),
            "source_variant": saved_model.get("variant", "v2"),
            "target_variant": self.config.model.variant,
            "transferred_tensors": len(transferred),
            "intentionally_uninitialized_tensors": len(missing),
            "unexpected_tensors": list(unexpected),
        }
        atomic_json(self.output / "initialization.json", report)
        print(
            f"initialized {self.config.model.variant.upper()} from {path}: transferred={len(transferred)} "
            f"source_epoch={checkpoint.get('epoch')}", flush=True,
        )

    def initialize_isolated_stage_b(self, path: str | Path) -> dict[str, Any]:
        """Load the complete frozen encoder/T1 anchor while leaving V6 T2 fresh."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved = checkpoint.get("config", {})
        saved_data, saved_model = saved.get("data", {}), saved.get("model", {})
        expected = (
            self.config.data.input_pipeline, self.config.model.sequence_stem,
            self.config.model.feature_dim, self.config.data.image_height, self.config.data.image_width,
        )
        actual = (
            saved_data.get("input_pipeline", "legacy_v1"),
            saved_model.get("sequence_stem", "legacy_stride8"),
            saved_model.get("feature_dim", 10),
            saved_data.get("image_height", 256), saved_data.get("image_width", 512),
        )
        if actual != expected:
            raise ValueError(f"Stage A anchor representation mismatch: checkpoint={actual}, configured={expected}")
        current = self.model.state_dict()
        anchor_keys = {key for key in current if key.startswith(("encoder.", "t1."))}
        missing = sorted(key for key in anchor_keys if key not in checkpoint["model"])
        incompatible = sorted(
            key for key in anchor_keys
            if key in checkpoint["model"] and checkpoint["model"][key].shape != current[key].shape
        )
        if missing or incompatible:
            raise ValueError(
                f"Stage A anchor is incomplete: missing={missing[:5]}, incompatible={incompatible[:5]}"
            )
        transferred = {key: checkpoint["model"][key] for key in anchor_keys}
        transferred_t2_encoder: list[str] = []
        if self.config.model.variant in {"v9", "t2_abc_v1"} and (
            self.config.model.variant != "t2_abc_v1"
            or self.config.train.t2_copy_encoder_from_t1
        ):
            for key, tensor in current.items():
                if not key.startswith("t2_encoder."):
                    continue
                source_key = "encoder." + key.removeprefix("t2_encoder.")
                if source_key in checkpoint["model"] and checkpoint["model"][source_key].shape == tensor.shape:
                    transferred[key] = checkpoint["model"][source_key]
                    transferred_t2_encoder.append(key)
        if self.config.model.variant == "v10":
            for target_prefix in ("t2_rank_encoder.", "t2_open_encoder."):
                for key, tensor in current.items():
                    if not key.startswith(target_prefix):
                        continue
                    source_key = "encoder." + key.removeprefix(target_prefix)
                    if source_key in checkpoint["model"] and checkpoint["model"][source_key].shape == tensor.shape:
                        transferred[key] = checkpoint["model"][source_key]
                        transferred_t2_encoder.append(key)
        transferred_t2: list[str] = []
        if self.config.model.variant == "v8" and saved_model.get("variant") == "v7":
            t2_prefixes = (
                "t2.rank_branch.", "t2.open_global_adapter.",
                "t2.open_local_adapter.", "t2.query_evidence.",
            )
            for key, tensor in current.items():
                if (
                    key.startswith(t2_prefixes)
                    and key in checkpoint["model"]
                    and checkpoint["model"][key].shape == tensor.shape
                ):
                    transferred[key] = checkpoint["model"][key]
                    transferred_t2.append(key)
        t2_initialization = self.config.train.t2_initialization_checkpoint
        if self.config.model.variant == "v9" and t2_initialization:
            t2_checkpoint = torch.load(t2_initialization, map_location=self.device, weights_only=False)
            t2_model = t2_checkpoint["model"]
            mappings = (
                ("t2.relation.", "t2.rank_branch."),
                ("t2.candidate_score.", "t2.incremental_rank."),
            )
            for target_prefix, source_prefix in mappings:
                for key, tensor in current.items():
                    if not key.startswith(target_prefix):
                        continue
                    source_key = source_prefix + key.removeprefix(target_prefix)
                    if source_key in t2_model and t2_model[source_key].shape == tensor.shape:
                        transferred[key] = t2_model[source_key]
                        transferred_t2.append(key)
        if self.config.model.variant == "v10" and t2_initialization:
            t2_checkpoint = torch.load(t2_initialization, map_location=self.device, weights_only=False)
            t2_model = t2_checkpoint["model"]
            mappings = (
                ("t2_rank_encoder.", "t2_encoder."),
                ("t2.rank_relation.", "t2.relation."),
                ("t2.rank_score.", "t2.candidate_score."),
                ("t2.rank_log_temperature", "t2.log_temperature"),
            )
            for target_prefix, source_prefix in mappings:
                for key, tensor in current.items():
                    if not key.startswith(target_prefix):
                        continue
                    source_key = source_prefix + key.removeprefix(target_prefix)
                    if source_key in t2_model and t2_model[source_key].shape == tensor.shape:
                        transferred[key] = t2_model[source_key]
                        transferred_t2.append(key)
        self.model.load_state_dict(transferred, strict=False)
        source_digest = hashlib.sha256()
        for key in sorted(anchor_keys):
            source_digest.update(key.encode("utf-8"))
            source_digest.update(checkpoint["model"][key].detach().contiguous().cpu().numpy().tobytes())
        source_digest_value = source_digest.hexdigest()
        loaded_digest = self._frozen_t1_digest()
        exact_anchor = source_digest_value == loaded_digest
        if not exact_anchor:
            raise RuntimeError("Loaded encoder/T1 digest does not match the Stage A source checkpoint")
        report = {
            "stage_a_checkpoint": str(Path(path).resolve()),
            "stage_a_checkpoint_epoch": checkpoint.get("epoch"),
            "source_variant": saved_model.get("variant"),
            "target_variant": self.config.model.variant,
            "transferred_tensors": len(transferred),
            "transferred_t2_tensors": len(transferred_t2),
            "transferred_t2_encoder_tensors": len(transferred_t2_encoder),
            "t2_copy_encoder_from_t1": self.config.train.t2_copy_encoder_from_t1,
            "t2_initialization_checkpoint": (
                str(Path(t2_initialization).resolve()) if t2_initialization else None
            ),
            "fresh_t2": not transferred_t2,
            "fresh_stateful_v8_modules": self.config.model.variant == "v8",
            "source_frozen_t1_digest": source_digest_value,
            "loaded_frozen_t1_digest": loaded_digest,
            "complete_v5r1_t1_anchor": (
                exact_anchor and self.config.model.t1_variant == "v5r1"
            ),
        }
        atomic_json(self.output / "stage_b_source.json", report)
        return {"checkpoint": checkpoint, "report": report}

    def initialize_sen_stage_b(self, stage_a_path: str | Path) -> dict[str, Any]:
        """Load the V2.0 Stage-A representation while leaving the T2 head fresh."""
        checkpoint = torch.load(stage_a_path, map_location="cpu", weights_only=False)
        saved_model = checkpoint.get("config", {}).get("model", {})
        expected = (
            self.config.model.variant,
            self.config.model.image_backbone,
            self.config.model.hidden_dim,
            self.config.model.sequence_stem,
        )
        actual = (
            saved_model.get("variant"),
            saved_model.get("image_backbone"),
            saved_model.get("hidden_dim"),
            saved_model.get("sequence_stem"),
        )
        if actual != expected:
            raise ValueError(
                f"Unified SEN Stage-A architecture mismatch: checkpoint={actual}, configured={expected}"
            )
        current = self.model.state_dict()
        prefixes = ("encoder.", "t1.", "shared_relation.")
        transfer = {
            key: value for key, value in checkpoint["model"].items()
            if key.startswith(prefixes) and key in current and value.shape == current[key].shape
        }
        required = {key for key in current if key.startswith(prefixes)}
        missing = sorted(required - transfer.keys())
        if missing:
            raise ValueError(f"Unified SEN Stage-A checkpoint is incomplete: {missing[:5]}")
        self.model.load_state_dict(transfer, strict=False)
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "validation": checkpoint.get("validation", {}),
            "config": checkpoint.get("config", {}),
        }
        report = {
            "initialization": "v20_stage_a_shared_encoder_t1_and_pair_matcher",
            "stage_a_checkpoint": str(Path(stage_a_path).resolve()),
            "stage_a_checkpoint_epoch": checkpoint.get("epoch"),
            "target_variant": self.config.model.variant,
            "transferred_tensors": len(transfer),
            "fresh_t2_head": True,
            "single_shared_encoder": True,
            "image_backbone": self.config.model.image_backbone,
        }
        atomic_json(self.output / "stage_b_source.json", report)
        print(
            f"initialized UNIFIED_SEN_V20 from Stage-A: transferred_tensors={len(transfer)} "
            "fresh_t2_head=true",
            flush=True,
        )
        del checkpoint
        return {"checkpoint": metadata, "report": report}

    def initialize_unified_stage_b(self, stage_a_path: str | Path) -> dict[str, Any]:
        """Compose one shared model from the frozen T1 anchor and the frozen T2 head."""
        t2_path = self.config.train.t2_initialization_checkpoint
        if not t2_path:
            raise ValueError(
                f"{self.config.model.variant} requires train.t2_initialization_checkpoint"
            )

        def representation_signature(checkpoint: dict[str, Any]) -> tuple[Any, ...]:
            saved = checkpoint.get("config", {})
            data = saved.get("data", {})
            model = saved.get("model", {})
            return (
                data.get("input_pipeline", "legacy_v1"),
                model.get("sequence_stem", "legacy_stride8"),
                model.get("feature_dim", 10),
                data.get("image_height", 256),
                data.get("image_width", 512),
            )

        expected = (
            self.config.data.input_pipeline,
            self.config.model.sequence_stem,
            self.config.model.feature_dim,
            self.config.data.image_height,
            self.config.data.image_width,
        )
        current = self.model.state_dict()

        t1_checkpoint = torch.load(stage_a_path, map_location="cpu", weights_only=False)
        if representation_signature(t1_checkpoint) != expected:
            raise ValueError(
                "Unified T1 anchor representation mismatch: "
                f"checkpoint={representation_signature(t1_checkpoint)}, configured={expected}"
            )
        t1_keys = {key for key in current if key.startswith(("encoder.", "t1."))}
        missing_t1 = sorted(key for key in t1_keys if key not in t1_checkpoint["model"])
        incompatible_t1 = sorted(
            key for key in t1_keys
            if key in t1_checkpoint["model"]
            and t1_checkpoint["model"][key].shape != current[key].shape
        )
        if missing_t1 or incompatible_t1:
            raise ValueError(
                "Unified T1 anchor is incomplete: "
                f"missing={missing_t1[:5]}, incompatible={incompatible_t1[:5]}"
            )
        self.model.load_state_dict(
            {key: t1_checkpoint["model"][key] for key in t1_keys}, strict=False,
        )
        t1_metadata = {
            "epoch": t1_checkpoint.get("epoch"),
            "validation": t1_checkpoint.get("validation", {}),
            "config": t1_checkpoint.get("config", {}),
        }
        t1_source_variant = t1_checkpoint.get("config", {}).get("model", {}).get("variant")
        del t1_checkpoint

        t2_checkpoint = torch.load(t2_path, map_location="cpu", weights_only=False)
        if representation_signature(t2_checkpoint) != expected:
            raise ValueError(
                "Unified T2 anchor representation mismatch: "
                f"checkpoint={representation_signature(t2_checkpoint)}, configured={expected}"
            )
        t2_keys = {key for key in current if key.startswith("t2.")}
        missing_t2 = sorted(key for key in t2_keys if key not in t2_checkpoint["model"])
        incompatible_t2 = sorted(
            key for key in t2_keys
            if key in t2_checkpoint["model"]
            and t2_checkpoint["model"][key].shape != current[key].shape
        )
        if missing_t2 or incompatible_t2:
            raise ValueError(
                "Unified T2 head anchor is incomplete: "
                f"missing={missing_t2[:5]}, incompatible={incompatible_t2[:5]}"
            )
        self.model.load_state_dict(
            {key: t2_checkpoint["model"][key] for key in t2_keys}, strict=False,
        )
        t2_source_variant = t2_checkpoint.get("config", {}).get("model", {}).get("variant")
        report = {
            "initialization": "t1_encoder_and_head_plus_t2_probability_head",
            "stage_a_checkpoint": str(Path(stage_a_path).resolve()),
            "stage_a_checkpoint_epoch": t1_metadata["epoch"],
            "stage_a_source_variant": t1_source_variant,
            "t2_initialization_checkpoint": str(Path(t2_path).resolve()),
            "t2_checkpoint_epoch": t2_checkpoint.get("epoch"),
            "t2_source_variant": t2_source_variant,
            "target_variant": self.config.model.variant,
            "transferred_t1_tensors": len(t1_keys),
            "transferred_t2_tensors": len(t2_keys),
            "t2_encoder_present_in_source": any(
                key.startswith("t2_encoder.") for key in t2_checkpoint["model"]
            ),
            "t2_encoder_deployed": False,
            "t2_encoder_used_as_distillation_teacher": (
                self.config.model.variant in {
                    "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                }
            ),
            "t2_adapter_bottleneck": (
                self.config.model.t2_adapter_bottleneck
                if self.config.model.variant in {
                    "unified_abc_v12", "unified_abc_v13", "unified_abc_v14",
                } else None
            ),
            "t2_adapter_kind": (
                "internal_residual" if self.config.model.variant == "unified_abc_v14"
                else "output_residual" if self.config.model.variant in {
                    "unified_abc_v12", "unified_abc_v13",
                } else None
            ),
            "shared_encoder_instances": 1,
        }
        atomic_json(self.output / "stage_b_source.json", report)
        print(
            f"initialized {self.config.model.variant.upper()} from separate T1 and T2 anchors: "
            f"t1_tensors={len(t1_keys)} t2_tensors={len(t2_keys)}",
            flush=True,
        )
        return {"checkpoint": t1_metadata, "report": report}

    def initialize_unified_scratch_stage_b(self, stage_a_path: str | Path) -> dict[str, Any]:
        """Load this run's T1 representation and leave the T2 path independently initialized."""
        if self.config.train.t2_initialization_checkpoint:
            raise ValueError("Scratch T2 initialization must not specify t2_initialization_checkpoint")
        if self.config.train.t2_adapter_distillation_steps:
            raise ValueError("Scratch T2 initialization must not use adapter distillation")

        checkpoint = torch.load(stage_a_path, map_location="cpu", weights_only=False)
        saved = checkpoint.get("config", {})
        saved_data = saved.get("data", {})
        saved_model = saved.get("model", {})
        actual = (
            saved_data.get("input_pipeline", "legacy_v1"),
            saved_model.get("sequence_stem", "legacy_stride8"),
            saved_model.get("feature_dim", 10),
            saved_data.get("image_height", 256),
            saved_data.get("image_width", 512),
        )
        expected = (
            self.config.data.input_pipeline,
            self.config.model.sequence_stem,
            self.config.model.feature_dim,
            self.config.data.image_height,
            self.config.data.image_width,
        )
        if actual != expected:
            raise ValueError(
                f"Unified scratch Stage-A representation mismatch: checkpoint={actual}, configured={expected}"
            )

        current = self.model.state_dict()
        transfer_keys = {key for key in current if key.startswith(("encoder.", "t1."))}
        missing = sorted(key for key in transfer_keys if key not in checkpoint["model"])
        incompatible = sorted(
            key for key in transfer_keys
            if key in checkpoint["model"] and checkpoint["model"][key].shape != current[key].shape
        )
        if missing or incompatible:
            raise ValueError(
                "Unified scratch Stage-A anchor is incomplete: "
                f"missing={missing[:5]}, incompatible={incompatible[:5]}"
            )
        self.model.load_state_dict(
            {key: checkpoint["model"][key] for key in transfer_keys}, strict=False,
        )
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "validation": checkpoint.get("validation", {}),
            "config": checkpoint.get("config", {}),
        }
        report = {
            "initialization": "independent_t2_from_own_stage_a",
            "stage_a_checkpoint": str(Path(stage_a_path).resolve()),
            "stage_a_checkpoint_epoch": checkpoint.get("epoch"),
            "stage_a_source_variant": saved_model.get("variant"),
            "t2_initialization_checkpoint": None,
            "external_teacher_used": False,
            "transferred_t1_tensors": len(transfer_keys),
            "fresh_t2_head": True,
            "identity_initialized_t2_adapter": True,
            "shared_encoder_instances": 1,
        }
        atomic_json(self.output / "stage_b_source.json", report)
        print(
            "initialized UNIFIED SCRATCH ABC from its own Stage-A: "
            f"t1_tensors={len(transfer_keys)} fresh_t2_head=true external_teacher=false",
            flush=True,
        )
        del checkpoint
        return {"checkpoint": metadata, "report": report}

    def load_checkpoint(self, path: str | Path, restore_optimizer: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._validate_checkpoint_config(checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        if restore_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _validate_checkpoint_config(self, checkpoint: dict[str, Any]) -> None:
        saved = checkpoint.get("config", {})
        saved_data = saved.get("data", {})
        saved_model = saved.get("model", {})
        saved_pipeline = saved_data.get("input_pipeline", "legacy_v1")
        saved_stem = saved_model.get("sequence_stem", "legacy_stride8")
        saved_feature_dim = saved_model.get("feature_dim", 10)
        saved_variant = saved_model.get("variant", "v2")
        expected = (
            self.config.data.input_pipeline,
            self.config.model.sequence_stem,
            self.config.model.feature_dim,
            self.config.model.variant,
        )
        actual = (saved_pipeline, saved_stem, saved_feature_dim, saved_variant)
        if self.config.data.input_pipeline in {"rgb_legacy_ablation", "raw_rgb_v2"}:
            expected += (
                self.config.data.image_height, self.config.data.image_width,
                self.config.data.image_margin, self.config.data.image_line_width,
                self.config.data.image_supersample,
                self.config.data.image_speed_cap_mm_s,
            )
            actual += (
                saved_data.get("image_height"), saved_data.get("image_width"),
                saved_data.get("image_margin", 12), saved_data.get("image_line_width", 2),
                saved_data.get("image_supersample", 2),
                saved_data.get("image_speed_cap_mm_s", 800.0),
            )
        if self.config.data.input_pipeline in {"gray_raw_ablation", "raw_rgb_v2"}:
            expected += (
                self.config.data.raw_time_scale_seconds, self.config.data.raw_position_scale_mm,
            )
            actual += (
                saved_data.get("raw_time_scale_seconds", 60.0),
                saved_data.get("raw_position_scale_mm", 100.0),
            )
        if actual != expected:
            raise ValueError(
                "Checkpoint input contract does not match this run: "
                f"checkpoint={actual}, configured={expected}. V1 checkpoints cannot initialize V2."
            )


def smoke_forward(config: ExperimentConfig) -> dict[str, Any]:
    trainer = Trainer(config)
    results = {}
    trainer.model.train()
    for protocol in ("t1_1v1", "t1_5v1", "t2"):
        batch = _move(next(iter(trainer.loader("train", protocol, train=True))), trainer.device)
        trainer.optimizer.zero_grad(set_to_none=True)
        output = trainer.model(batch)
        loss, parts = (trainer.t1_loss(output, batch) if protocol.startswith("t1") else trainer.t2_loss(output, batch))
        loss.backward()
        finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters())
        results[protocol] = {"loss": float(loss.detach()), "finite_gradients": bool(finite),
                             "batch_episodes": len(batch["episode_id"])}
    return results
