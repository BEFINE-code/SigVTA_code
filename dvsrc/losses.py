from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def balanced_bce(logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is not None:
        logits, targets = logits[mask], targets[mask]
    positives = targets.sum().clamp_min(1)
    negatives = (1 - targets).sum().clamp_min(1)
    weights = torch.where(targets > 0.5, 0.5 / positives, 0.5 / negatives) * len(targets)
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)


def supervised_contrastive(embedding: Tensor, labels: list[str], temperature: float = 0.1) -> Tensor:
    if len(embedding) < 2:
        return embedding.new_zeros(())
    z = F.normalize(embedding, dim=-1)
    logits = z @ z.T / temperature
    identity = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = torch.tensor([[a == b for b in labels] for a in labels], device=z.device) & ~identity
    logits = logits.masked_fill(identity, -1e4)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    valid = positive.any(dim=1)
    if not valid.any():
        return z.new_zeros(())
    return -(log_probability * positive).sum(1)[valid].div(positive.sum(1)[valid]).mean()


class T1Loss(nn.Module):
    def __init__(self, genuine_identity_only: bool = False):
        super().__init__()
        self.genuine_identity_only = genuine_identity_only

    def forward(self, output: dict[str, Tensor], batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        label = batch["label"]
        pair_target = label[:, None].expand_as(output["pair_logits"])
        pair = balanced_bce(output["pair_logits"], pair_target, batch["set_mask"])
        case = balanced_bce(output["case_logit"], label)
        writers = [row["target_writer_id"] for row in batch["metadata"]]
        identity = output["identity_embedding"]
        if self.genuine_identity_only:
            genuine = label > 0.5
            identity = identity[genuine]
            writers = [writer for writer, keep in zip(writers, genuine.tolist()) if keep]
        metric = supervised_contrastive(identity, writers)
        # Subset consistency is active when a caller supplies subset logits from reference dropout.
        subset = output.get("subset_loss", pair.new_zeros(()))
        quality = output.get("quality_rank_loss", pair.new_zeros(()))
        set_aux = (
            balanced_bce(output["set_aux_logit"], label)
            if "set_aux_logit" in output else pair.new_zeros(())
        )
        anchor_alignment = (
            F.smooth_l1_loss(
                torch.sigmoid(output["set_aux_logit"]),
                torch.sigmoid(output["evidence_anchor_logit"]).detach(),
            )
            if "set_aux_logit" in output else pair.new_zeros(())
        )
        view = 0.05 * output["view_global_loss"] + 0.02 * output["view_local_loss"]
        total = pair + case + 0.1 * metric + 0.1 * subset + 0.05 * quality
        total = total + 0.25 * set_aux + 0.1 * anchor_alignment + view
        return total, {"t1_total": total, "t1_pair": pair, "t1_set": case, "t1_metric": metric,
                       "t1_subset": subset, "t1_quality": quality, "t1_set_aux": set_aux,
                       "t1_anchor_alignment": anchor_alignment, "view": view}


class T2Loss(nn.Module):
    def __init__(self, rank_margin: float = 0.2, positive_margin: float = 0.2,
                 negative_margin: float = -0.2, mode: str = "legacy_joint",
                 pair_margin: float = 0.5, pair_weight: float = 0.5,
                 query_weight: float = 0.5, monotonic_weight: float = 0.5,
                 stateful_weight: float = 0.75, arrival_weight: float = 0.25,
                 bridge_weight: float = 0.0, official_weight: float = 0.0,
                 joint_weight: float = 1.0, factor_a_weight: float = 0.0,
                 factor_b_weight: float = 0.0, rank_weight: float = 0.0,
                 rank_margin_weight: float = 0.0, rank_distillation_weight: float = 0.0,
                 b_distillation_weight: float = 0.0, hard_negative_weight: float = 0.0,
                 distillation_temperature: float = 2.0):
        super().__init__()
        self.rank_margin = rank_margin
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.pair_margin = pair_margin
        self.pair_weight = pair_weight
        self.query_weight = query_weight
        self.monotonic_weight = monotonic_weight
        self.stateful_weight = stateful_weight
        self.arrival_weight = arrival_weight
        self.bridge_weight = bridge_weight
        self.official_weight = official_weight
        self.joint_weight = joint_weight
        self.factor_a_weight = factor_a_weight
        self.factor_b_weight = factor_b_weight
        self.rank_weight = rank_weight
        self.rank_margin_weight = rank_margin_weight
        self.rank_distillation_weight = rank_distillation_weight
        self.b_distillation_weight = b_distillation_weight
        self.hard_negative_weight = hard_negative_weight
        self.distillation_temperature = distillation_temperature
        self.training_phase = "joint"
        if mode not in {
            "legacy_joint", "factorized_v2", "hierarchical_v3",
            "open_set_v4", "open_set_v5", "bayesian_v6", "progressive_bayesian_v7",
            "stateful_bayesian_v8", "unified_evidence_v9", "dual_evidence_v10",
            "conditional_abc_v1",
        }:
            raise ValueError(f"Unsupported T2 loss: {mode}")
        self.mode = mode

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"joint", "rank_recovery", "b_calibration"}:
            raise ValueError(f"Unsupported conditional T2 training phase: {phase}")
        self.training_phase = phase

    def forward(self, output: dict[str, Tensor], batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        target = batch["target_index"]
        candidate_count = output["rank_logits"].shape[1]
        present = target >= 0
        if self.mode == "conditional_abc_v1":
            type_target = batch["episode_type_index"]
            sf = type_target != 2
            zero = output["rank_logits"].new_zeros(())
            rf_target = (type_target == 2).to(output["rf_logit"].dtype)
            rf = F.binary_cross_entropy_with_logits(output["rf_logit"], rf_target)
            in_set = (
                F.binary_cross_entropy_with_logits(
                    output["in_set_logit"][sf], present[sf].to(output["in_set_logit"].dtype),
                ) if sf.any() else zero
            )
            rank = (
                F.cross_entropy(output["rank_logits"][present], target[present])
                if present.any() else zero
            )
            if present.any():
                present_logits = output["rank_logits"][present]
                source = present_logits.gather(1, target[present, None]).squeeze(1)
                decoy = present_logits.clone()
                decoy.scatter_(1, target[present, None], -1e4)
                rank_margin = F.softplus(
                    self.rank_margin - source + decoy.max(dim=1).values
                ).mean()
            else:
                rank_margin = zero
            joint_target = target.where(present, torch.full_like(target, candidate_count))
            joint_nll = F.nll_loss(
                output["joint_probability"].clamp_min(1e-8).log(), joint_target,
            )
            official_target = torch.where(
                present,
                target,
                torch.where(
                    type_target == 2,
                    torch.full_like(target, candidate_count + 1),
                    torch.full_like(target, candidate_count),
                ),
            )
            official_nll = (
                F.nll_loss(
                    output["official_probability"].clamp_min(1e-8).log(), official_target,
                ) if "official_probability" in output else zero
            )
            bridge = (
                F.binary_cross_entropy_with_logits(
                    output["bridge_case_logit"], torch.zeros_like(output["bridge_case_logit"]),
                ) if "bridge_case_logit" in output else zero
            )
            pair_terms = []
            grouped: dict[str, dict[str, int]] = {}
            for index, row in enumerate(batch.get("metadata", [])):
                pair_id = row.get("counterfactual_pair_id")
                role = row.get("counterfactual_role")
                if pair_id and role:
                    grouped.setdefault(pair_id, {})[role] = index
            for roles in grouped.values():
                if "present" in roles and "absent" in roles:
                    pair_terms.append(F.softplus(
                        self.pair_margin
                        - output["in_set_logit"][roles["present"]]
                        + output["in_set_logit"][roles["absent"]]
                    ))
            pair = torch.stack(pair_terms).mean() if pair_terms else zero

            rank_distillation = zero
            teacher_hard_negative = zero
            if present.any() and "teacher_rank_logits" in output:
                temperature = self.distillation_temperature
                student_log_probability = F.log_softmax(
                    output["rank_logits"][present] / temperature, dim=-1,
                )
                teacher_probability = F.softmax(
                    output["teacher_rank_logits"][present].detach() / temperature, dim=-1,
                )
                rank_distillation = F.kl_div(
                    student_log_probability, teacher_probability, reduction="batchmean",
                ) * temperature ** 2
                teacher_decoy = output["teacher_rank_logits"][present].detach().clone()
                teacher_decoy.scatter_(1, target[present, None], -1e4)
                hard_negative_index = teacher_decoy.argmax(dim=1, keepdim=True)
                source_logit = output["rank_logits"][present].gather(
                    1, target[present, None],
                ).squeeze(1)
                hard_negative_logit = output["rank_logits"][present].gather(
                    1, hard_negative_index,
                ).squeeze(1)
                teacher_hard_negative = F.softplus(
                    self.rank_margin - source_logit + hard_negative_logit,
                ).mean()
            b_distillation = zero
            if sf.any() and "teacher_in_set_logit" in output:
                b_distillation = F.binary_cross_entropy_with_logits(
                    output["in_set_logit"][sf],
                    torch.sigmoid(output["teacher_in_set_logit"][sf].detach()),
                )
            joint_total = (
                self.joint_weight * joint_nll
                + self.official_weight * official_nll
                + self.bridge_weight * bridge
                + self.factor_a_weight * rf
                + self.factor_b_weight * in_set
                + self.rank_weight * rank
                + self.rank_margin_weight * rank_margin
                + self.pair_weight * pair
                + self.rank_distillation_weight * rank_distillation
                + self.b_distillation_weight * b_distillation
            )
            if self.training_phase == "rank_recovery":
                total = (
                    self.rank_weight * rank
                    + self.rank_margin_weight * rank_margin
                    + self.rank_distillation_weight * rank_distillation
                    + self.hard_negative_weight * teacher_hard_negative
                    + 0.1 * self.official_weight * official_nll
                )
            elif self.training_phase == "b_calibration":
                total = (
                    self.factor_b_weight * in_set
                    + self.pair_weight * pair
                    + self.b_distillation_weight * b_distillation
                )
            else:
                total = joint_total
            return total, {
                "t2_total": total,
                "t2_rf": rf,
                "t2_in_set": in_set,
                "t2_rank": rank,
                "t2_rank_margin": rank_margin,
                "t2_joint_nll": joint_nll,
                "t2_official_nll": official_nll,
                "t2_counterfactual_pair": pair,
                "t2_rank_distillation": rank_distillation,
                "t2_teacher_hard_negative": teacher_hard_negative,
                "t2_b_distillation": b_distillation,
                "t1_t2_bridge": bridge,
            }
        if self.mode == "dual_evidence_v10":
            type_target = batch["episode_type_index"]
            sf = type_target != 2
            zero = output["rank_logits"].new_zeros(())

            rank = (
                F.cross_entropy(output["rank_logits"][present], target[present], label_smoothing=0.05)
                if present.any() else zero
            )
            if present.any():
                present_logits = output["rank_logits"][present]
                present_target = target[present]
                source = present_logits.gather(1, present_target[:, None]).squeeze(1)
                decoy = present_logits.clone()
                decoy.scatter_(1, present_target[:, None], -1e4)
                rank_margin = F.softplus(
                    self.rank_margin - source + decoy.max(dim=1).values,
                ).mean()
                rank_candidate_target = torch.zeros_like(present_logits)
                rank_candidate_target.scatter_(1, present_target[:, None], 1)
                candidate = balanced_bce(
                    present_logits, rank_candidate_target, batch["set_mask"][present],
                )
            else:
                rank_margin = candidate = zero

            rf_target = (type_target == 2).to(output["rf_logit"].dtype)
            rf = F.binary_cross_entropy_with_logits(output["rf_logit"], rf_target)
            open_type = F.nll_loss(output["type_probability"].clamp_min(1e-8).log(), type_target)
            open_presence = (
                F.binary_cross_entropy_with_logits(
                    output["open_in_set_logit"][sf], present[sf].to(output["open_in_set_logit"].dtype),
                ) if sf.any() else zero
            )

            pair_terms, localization_terms, invariance_terms = [], [], []
            grouped: dict[str, dict[str, int]] = {}
            for index, row in enumerate(batch["metadata"]):
                pair_id = row.get("counterfactual_pair_id")
                role = row.get("counterfactual_role")
                if pair_id and role:
                    grouped.setdefault(pair_id, {})[role] = index
            for roles in grouped.values():
                if "present" not in roles or "absent" not in roles:
                    continue
                present_index, absent_index = roles["present"], roles["absent"]
                source_index = int(target[present_index])
                pair_terms.append(F.softplus(
                    self.pair_margin
                    - output["open_in_set_logit"][present_index]
                    + output["open_in_set_logit"][absent_index],
                ))
                localization_terms.append(F.softplus(
                    self.pair_margin
                    - output["open_presence_logits"][present_index, source_index]
                    + output["open_presence_logits"][absent_index, source_index],
                ))
                unchanged = batch["set_mask"][present_index].clone()
                unchanged[source_index] = False
                if unchanged.any():
                    invariance_terms.append(F.smooth_l1_loss(
                        output["open_presence_logits"][present_index, unchanged],
                        output["open_presence_logits"][absent_index, unchanged],
                    ))
            pair = torch.stack(pair_terms).mean() if pair_terms else zero
            localization = torch.stack(localization_terms).mean() if localization_terms else zero
            invariance = torch.stack(invariance_terms).mean() if invariance_terms else zero

            prefix_logits = output["stateful_prefix_logits"]
            prefix_probability = output["stateful_prefix_probability"].clamp_min(1e-8)
            prefix_sizes = output["stateful_prefix_sizes"]
            observed = present[:, None] & (target[:, None] < prefix_sizes[None])
            prefix_target = torch.where(
                (type_target == 2)[:, None], torch.full_like(observed, 2, dtype=torch.long),
                torch.where(observed, torch.zeros_like(observed, dtype=torch.long),
                            torch.ones_like(observed, dtype=torch.long)),
            )
            prefix = F.nll_loss(prefix_logits.reshape(-1, 3), prefix_target.reshape(-1))
            prefix_correct = prefix_probability.gather(
                2, prefix_target.unsqueeze(-1),
            ).squeeze(-1)
            stable = prefix_target[:, 1:] == prefix_target[:, :-1]
            prefix_nll = -prefix_correct.log()
            stateful_monotonic = (
                F.relu(prefix_nll[:, 1:] - prefix_nll[:, :-1])[stable].mean()
                if stable.any() else zero
            )
            arrival_terms = []
            prefix_present_logit = (
                prefix_probability[..., 0].log()
                - prefix_probability[..., 1:].sum(dim=-1).clamp_min(1e-8).log()
            )
            for row in torch.where(present)[0].tolist():
                observed_steps = torch.where(observed[row])[0]
                if len(observed_steps):
                    first = int(observed_steps[0])
                    before = zero if first == 0 else prefix_present_logit[row, first - 1]
                    arrival_terms.append(F.softplus(
                        self.pair_margin - prefix_present_logit[row, first] + before,
                    ))
            arrival = torch.stack(arrival_terms).mean() if arrival_terms else zero

            joint_target = target.where(present, torch.full_like(target, candidate_count))
            joint_nll = F.nll_loss(
                output["joint_probability"].clamp_min(1e-8).log(), joint_target,
            )
            one_hot = F.one_hot(joint_target, candidate_count + 1).to(output["joint_probability"].dtype)
            brier = (output["joint_probability"] - one_hot).square().sum(dim=-1).mean()
            phase = output.get("training_phase", "open")
            if phase == "rank":
                total = 1.5 * rank + 0.4 * rank_margin + 0.5 * candidate
            elif phase == "calibration":
                total = joint_nll + 0.5 * open_type + 0.25 * brier
            else:
                total = (
                    1.5 * open_type + open_presence + 0.75 * rf
                    + self.pair_weight * pair + 0.25 * localization + 0.25 * invariance
                    + self.stateful_weight * prefix
                    + self.monotonic_weight * stateful_monotonic
                    + self.arrival_weight * arrival + 0.1 * brier
                )
            return total, {
                "t2_total": total, "t2_rank": rank, "t2_rank_margin": rank_margin,
                "t2_candidate": candidate, "t2_open_type": open_type,
                "t2_open_presence": open_presence, "t2_rf": rf,
                "t2_counterfactual_pair": pair,
                "t2_counterfactual_localization": localization,
                "t2_counterfactual_invariance": invariance,
                "t2_prefix": prefix, "t2_stateful_monotonic": stateful_monotonic,
                "t2_source_arrival": arrival, "t2_joint_nll": joint_nll,
                "t2_brier": brier,
            }
        if self.mode == "unified_evidence_v9":
            type_target = batch["episode_type_index"]
            sf = type_target != 2
            rf_target = (type_target == 2).to(output["rf_logit"].dtype)
            rf = F.binary_cross_entropy_with_logits(output["rf_logit"], rf_target)

            conditional_target = target.where(present, torch.full_like(target, candidate_count))
            conditional = (
                F.cross_entropy(output["conditional_logits"][sf], conditional_target[sf])
                if sf.any() else output["rf_logit"].new_zeros(())
            )
            rank = (
                F.cross_entropy(output["rank_logits"][present], target[present], label_smoothing=0.05)
                if present.any() else output["rf_logit"].new_zeros(())
            )
            if present.any():
                present_logits = output["rank_logits"][present]
                present_target = target[present]
                source = present_logits.gather(1, present_target[:, None]).squeeze(1)
                decoy = present_logits.clone()
                decoy.scatter_(1, present_target[:, None], -1e4)
                rank_margin = F.softplus(
                    self.rank_margin - source + decoy.max(dim=1).values,
                ).mean()
            else:
                rank_margin = output["rf_logit"].new_zeros(())

            candidate_target = torch.zeros_like(output["candidate_logits"])
            if present.any():
                candidate_target[present, target[present]] = 1
            candidate = balanced_bce(
                output["candidate_logits"], candidate_target, batch["set_mask"],
            )

            pair_terms, localization_terms, invariance_terms = [], [], []
            grouped: dict[str, dict[str, int]] = {}
            for index, row in enumerate(batch["metadata"]):
                pair_id = row.get("counterfactual_pair_id")
                role = row.get("counterfactual_role")
                if pair_id and role:
                    grouped.setdefault(pair_id, {})[role] = index
            for roles in grouped.values():
                if "present" not in roles or "absent" not in roles:
                    continue
                present_index, absent_index = roles["present"], roles["absent"]
                source_index = int(target[present_index])
                pair_terms.append(F.softplus(
                    self.pair_margin
                    - output["in_set_logit"][present_index]
                    + output["in_set_logit"][absent_index],
                ))
                localization_terms.append(F.softplus(
                    self.pair_margin
                    - output["candidate_logits"][present_index, source_index]
                    + output["candidate_logits"][absent_index, source_index],
                ))
                unchanged = batch["set_mask"][present_index].clone()
                unchanged[source_index] = False
                if unchanged.any():
                    invariance_terms.append(F.smooth_l1_loss(
                        output["candidate_logits"][present_index, unchanged],
                        output["candidate_logits"][absent_index, unchanged],
                    ))
            zero = output["rf_logit"].new_zeros(())
            pair = torch.stack(pair_terms).mean() if pair_terms else zero
            localization = torch.stack(localization_terms).mean() if localization_terms else zero
            invariance = torch.stack(invariance_terms).mean() if invariance_terms else zero

            prefix_logits = output["prefix_conditional_logits"]
            prefix_sizes = output["stateful_prefix_sizes"]
            observed = present[:, None] & (target[:, None] < prefix_sizes[None])
            prefix_target = torch.where(
                observed, target[:, None].expand(-1, len(prefix_sizes)),
                torch.full_like(observed, candidate_count, dtype=torch.long),
            )
            prefix = (
                F.cross_entropy(
                    prefix_logits[sf].reshape(-1, candidate_count + 1),
                    prefix_target[sf].reshape(-1),
                ) if sf.any() else zero
            )
            prefix_probability = prefix_logits.softmax(dim=-1)
            prefix_correct = prefix_probability.gather(
                2, prefix_target.unsqueeze(-1),
            ).squeeze(-1).clamp_min(1e-8)
            stable = prefix_target[:, 1:] == prefix_target[:, :-1]
            prefix_nll = -prefix_correct.log()
            stateful_monotonic = (
                F.relu(prefix_nll[:, 1:] - prefix_nll[:, :-1])[stable].mean()
                if stable.any() else zero
            )

            arrival_terms = []
            prefix_in_set = torch.logsumexp(prefix_logits[..., :-1], dim=-1) - prefix_logits[..., -1]
            for row in torch.where(present)[0].tolist():
                observed_steps = torch.where(observed[row])[0]
                if len(observed_steps):
                    first = int(observed_steps[0])
                    before = zero if first == 0 else prefix_in_set[row, first - 1]
                    arrival_terms.append(F.softplus(
                        self.pair_margin - prefix_in_set[row, first] + before,
                    ))
            arrival = torch.stack(arrival_terms).mean() if arrival_terms else zero

            joint_target = target.where(present, torch.full_like(target, candidate_count))
            one_hot = F.one_hot(joint_target, candidate_count + 1).to(output["joint_probability"].dtype)
            brier = (output["joint_probability"] - one_hot).square().sum(dim=-1).mean()
            phase = output.get("training_phase", "unified")
            if phase == "representation":
                total = 1.5 * rank + 0.4 * rank_margin + 0.5 * candidate
            elif phase == "calibration":
                total = conditional + 0.75 * rf + 0.25 * prefix + 0.25 * brier
            else:
                total = (
                    1.5 * conditional + 0.75 * rf + 0.8 * rank + 0.2 * rank_margin
                    + 0.25 * candidate + self.pair_weight * (pair + localization)
                    + 0.25 * invariance + self.stateful_weight * prefix
                    + self.monotonic_weight * stateful_monotonic
                    + self.arrival_weight * arrival + 0.1 * brier
                )
            return total, {
                "t2_total": total, "t2_conditional": conditional, "t2_rf": rf,
                "t2_rank": rank, "t2_rank_margin": rank_margin,
                "t2_candidate": candidate, "t2_counterfactual_pair": pair,
                "t2_counterfactual_localization": localization,
                "t2_counterfactual_invariance": invariance,
                "t2_prefix": prefix, "t2_stateful_monotonic": stateful_monotonic,
                "t2_source_arrival": arrival, "t2_brier": brier,
            }
        if self.mode in {"progressive_bayesian_v7", "stateful_bayesian_v8"}:
            type_target = batch["episode_type_index"]
            stage_logits = output["bayesian_stage_logits"]
            stage_probability = output["bayesian_stage_probability"].clamp_min(1e-8)
            rank = (
                F.cross_entropy(output["rank_logits"][present], target[present], label_smoothing=0.05)
                if present.any() else stage_logits.new_zeros(())
            )
            if present.any():
                present_logits = output["rank_logits"][present]
                present_target = target[present]
                source = present_logits.gather(1, present_target[:, None]).squeeze(1)
                decoy = present_logits.clone()
                decoy.scatter_(1, present_target[:, None], -1e4)
                rank_margin = F.softplus(self.rank_margin - source + decoy.max(1).values).mean()
            else:
                rank_margin = stage_logits.new_zeros(())

            rf_target = (type_target == 2).to(stage_logits.dtype)
            query_rf_logit = stage_logits[:, 1, 2] - torch.logsumexp(stage_logits[:, 1, :2], dim=1)
            query = F.binary_cross_entropy_with_logits(query_rf_logit, rf_target)
            type_weight = stage_logits.new_tensor([1.0, 2.5, 5.0 / 3.0])
            global_type = F.cross_entropy(stage_logits[:, 2], type_target, weight=type_weight)
            final_type = F.cross_entropy(stage_logits[:, 3], type_target, weight=type_weight)

            correct_probability = stage_probability.gather(
                2, type_target[:, None, None].expand(-1, stage_probability.shape[1], 1),
            ).squeeze(-1)
            full_nll = -correct_probability.log()
            rf_probability = stage_probability[:, :, 2]
            binary_correct = torch.where(
                rf_target[:, None].bool(), rf_probability, 1 - rf_probability,
            ).clamp_min(1e-8)
            binary_nll = -binary_correct.log()
            if self.mode == "stateful_bayesian_v8":
                query_target = torch.where(type_target == 2, type_target, torch.ones_like(type_target))
                query_nll = -stage_probability[:, 1].gather(1, query_target[:, None]).squeeze(1).log()
                global_nll = full_nll[:, 2]
                stable_query_to_global = type_target != 0
                query_to_global = (
                    F.relu(global_nll - query_nll)[stable_query_to_global].mean()
                    if stable_query_to_global.any() else stage_logits.new_zeros(())
                )
                monotonic = (
                    F.relu(binary_nll[:, 1] - binary_nll[:, 0]).mean()
                    + query_to_global
                    + F.relu(full_nll[:, 3] - full_nll[:, 2]).mean()
                ) / 3
            else:
                monotonic = (
                    F.relu(binary_nll[:, 1] - binary_nll[:, 0]).mean()
                    + F.relu(full_nll[:, 2] - full_nll[:, 1]).mean()
                    + F.relu(full_nll[:, 3] - full_nll[:, 2]).mean()
                ) / 3

            pair_terms = []
            grouped: dict[str, dict[str, int]] = {}
            for index, row in enumerate(batch["metadata"]):
                pair_id = row.get("counterfactual_pair_id")
                role = row.get("counterfactual_role")
                if pair_id and role:
                    grouped.setdefault(pair_id, {})[role] = index
            for roles in grouped.values():
                if "present" in roles and "absent" in roles:
                    pair_terms.append(F.softplus(
                        self.pair_margin
                        - output["in_set_logit"][roles["present"]]
                        + output["in_set_logit"][roles["absent"]]
                    ))
            pair = torch.stack(pair_terms).mean() if pair_terms else stage_logits.new_zeros(())
            total = (
                1.5 * rank + 0.25 * rank_margin
                + self.query_weight * query + 0.8 * global_type + 1.2 * final_type
                + self.monotonic_weight * monotonic + self.pair_weight * pair
            )
            stateful = stage_logits.new_zeros(())
            stateful_monotonic = stage_logits.new_zeros(())
            arrival = stage_logits.new_zeros(())
            if self.mode == "stateful_bayesian_v8":
                prefix_logits = output["stateful_prefix_logits"]
                prefix_probability = output["stateful_prefix_probability"].clamp_min(1e-8)
                prefix_sizes = output["stateful_prefix_sizes"]
                observed = (target[:, None] >= 0) & (target[:, None] < prefix_sizes[None])
                dynamic_target = torch.where(
                    type_target[:, None] == 2,
                    torch.full_like(observed, 2, dtype=torch.long),
                    torch.where(observed, torch.zeros_like(observed, dtype=torch.long),
                                torch.ones_like(observed, dtype=torch.long)),
                )
                stateful = F.cross_entropy(
                    prefix_logits.reshape(-1, 3), dynamic_target.reshape(-1), weight=type_weight,
                )
                prefix_correct = prefix_probability.gather(
                    2, dynamic_target.unsqueeze(-1),
                ).squeeze(-1)
                prefix_nll = -prefix_correct.log()
                stable = dynamic_target[:, 1:] == dynamic_target[:, :-1]
                if stable.any():
                    stateful_monotonic = F.relu(
                        prefix_nll[:, 1:] - prefix_nll[:, :-1],
                    )[stable].mean()

                present_log_odds = prefix_logits[..., 0] - torch.logsumexp(prefix_logits[..., 1:], dim=-1)
                query_log_odds = stage_logits[:, 1, 0] - torch.logsumexp(stage_logits[:, 1, 1:], dim=-1)
                arrival_terms = []
                for row in torch.where(type_target == 0)[0].tolist():
                    observed_steps = torch.where(observed[row])[0]
                    if len(observed_steps):
                        first = int(observed_steps[0])
                        before = query_log_odds[row] if first == 0 else present_log_odds[row, first - 1]
                        arrival_terms.append(F.softplus(
                            self.pair_margin - present_log_odds[row, first] + before,
                        ))
                if arrival_terms:
                    arrival = torch.stack(arrival_terms).mean()
                total = total + self.stateful_weight * (
                    stateful + self.monotonic_weight * stateful_monotonic
                ) + self.arrival_weight * arrival
            return total, {
                "t2_total": total, "t2_rank": rank, "t2_rank_margin": rank_margin,
                "t2_query_rf": query, "t2_global_type": global_type,
                "t2_final_type": final_type, "t2_monotonic": monotonic,
                "t2_counterfactual_pair": pair,
                "t2_stateful": stateful, "t2_stateful_monotonic": stateful_monotonic,
                "t2_source_arrival": arrival,
            }
        if self.mode == "bayesian_v6":
            rf_target = (batch["episode_type_index"] == 2).to(output["rf_logit"].dtype)
            rf = F.binary_cross_entropy_with_logits(output["rf_logit"], rf_target)
            sf = batch["episode_type_index"] != 2
            in_set_target = present.to(output["in_set_logit"].dtype)
            in_set = F.binary_cross_entropy_with_logits(
                output["in_set_logit"][sf], in_set_target[sf],
            ) if sf.any() else output["rf_logit"].new_zeros(())
            rank = (
                F.cross_entropy(output["rank_logits"][present], target[present], label_smoothing=0.05)
                if present.any() else output["rf_logit"].new_zeros(())
            )
            candidate_target = torch.zeros_like(output["match_logits"])
            if present.any():
                candidate_target[present, target[present]] = 1
            candidate = balanced_bce(output["match_logits"], candidate_target, batch["set_mask"])
            joint_target = target.where(target >= 0, torch.full_like(target, candidate_count))
            joint_probability = output["training_joint_probability"].clamp_min(1e-8)
            joint = -joint_probability.gather(1, joint_target[:, None]).log().mean()

            pair_terms = []
            grouped: dict[str, dict[str, int]] = {}
            for index, row in enumerate(batch["metadata"]):
                pair_id = row.get("counterfactual_pair_id")
                role = row.get("counterfactual_role")
                if pair_id and role:
                    grouped.setdefault(pair_id, {})[role] = index
            for roles in grouped.values():
                if "present" in roles and "absent" in roles:
                    pair_terms.append(F.softplus(
                        self.pair_margin
                        - output["in_set_logit"][roles["present"]]
                        + output["in_set_logit"][roles["absent"]]
                    ))
            pair = torch.stack(pair_terms).mean() if pair_terms else output["rf_logit"].new_zeros(())
            total = 1.0 * rf + 1.0 * in_set + 1.25 * rank + 0.35 * candidate + 0.5 * joint
            total = total + self.pair_weight * pair
            return total, {
                "t2_total": total, "t2_rf": rf, "t2_in_set": in_set,
                "t2_rank": rank, "t2_candidate": candidate, "t2_joint": joint,
                "t2_counterfactual_pair": pair,
            }
        if self.mode == "open_set_v5":
            joint_target = target.where(target >= 0, torch.full_like(target, candidate_count))
            joint = F.cross_entropy(output["open_set_logits"], joint_target)
            exist = F.binary_cross_entropy_with_logits(output["exist_logit"], batch["exist_label"])
            type_weight = output["type_logits"].new_tensor([1.0, 2.5, 5.0 / 3.0])
            subtype = F.cross_entropy(
                output["type_logits"], batch["episode_type_index"], weight=type_weight,
            )
            correct_type = output["type_logits"].gather(
                1, batch["episode_type_index"][:, None],
            ).squeeze(1)
            competing_type = output["type_logits"].clone()
            competing_type.scatter_(1, batch["episode_type_index"][:, None], -1e4)
            subtype_margin = F.softplus(
                self.rank_margin - correct_type + competing_type.max(dim=1).values
            ).mean()

            candidate_target = torch.zeros_like(output["match_logits"])
            if present.any():
                candidate_target[present, target[present]] = 1
            candidate = balanced_bce(output["match_logits"], candidate_target, batch["set_mask"])

            if present.any():
                present_rank = output["rank_logits"][present]
                present_target = target[present]
                rank = F.cross_entropy(present_rank, present_target, label_smoothing=0.05)
                source = present_rank.gather(1, present_target[:, None]).squeeze(1)
                decoy = present_rank.clone()
                decoy.scatter_(1, present_target[:, None], -1e4)
                hardest = decoy.max(dim=1).values
                hard_margin = F.softplus(self.rank_margin - source + hardest).mean()
                negative_evidence = torch.logsumexp(
                    decoy - source[:, None] + self.rank_margin, dim=1,
                )
                # log(1 + sum(exp(margin + decoy - source))) is non-negative
                # and approaches zero once every decoy is safely below the source.
                all_negative = F.softplus(negative_evidence).mean()
                source_open = output["open_set_logits"][present].gather(
                    1, present_target[:, None],
                ).squeeze(1)
                present_open = F.softplus(
                    self.rank_margin - source_open + output["unknown_logit"][present]
                ).mean()
            else:
                rank = joint.new_zeros(())
                hard_margin = joint.new_zeros(())
                all_negative = joint.new_zeros(())
                present_open = joint.new_zeros(())
            unknown = ~present
            if unknown.any():
                strongest_candidate = output["open_set_logits"][unknown, :candidate_count].max(dim=1).values
                unknown_open = F.softplus(
                    self.rank_margin - output["unknown_logit"][unknown] + strongest_candidate
                ).mean()
            else:
                unknown_open = joint.new_zeros(())
            open_margin = (present_open + unknown_open) / (int(present.any()) + int(unknown.any()))
            view = 0.05 * output["view_global_loss"] + 0.02 * output["view_local_loss"]
            total = (
                1.25 * rank + 0.75 * subtype + 0.5 * exist + 0.5 * joint
                + 0.3 * candidate + 0.25 * hard_margin + 0.1 * all_negative
                + 0.2 * subtype_margin + 0.2 * open_margin + view
            )
            return total, {
                "t2_total": total,
                "t2_rank": rank,
                "t2_subtype": subtype,
                "t2_exist": exist,
                "t2_joint": joint,
                "t2_candidate": candidate,
                "t2_hard_margin": hard_margin,
                "t2_all_negative": all_negative,
                "t2_subtype_margin": subtype_margin,
                "t2_open_margin": open_margin,
                "view": view,
            }
        if self.mode == "open_set_v4":
            joint_target = target.where(target >= 0, torch.full_like(target, candidate_count))
            joint = F.cross_entropy(output["open_set_logits"], joint_target)
            exist = F.binary_cross_entropy_with_logits(output["exist_logit"], batch["exist_label"])
            candidate_target = torch.zeros_like(output["match_logits"])
            if present.any():
                candidate_target[present, target[present]] = 1
            candidate = balanced_bce(output["match_logits"], candidate_target, batch["set_mask"])
            if present.any():
                present_rank = output["rank_logits"][present]
                present_target = target[present]
                rank = F.cross_entropy(present_rank, present_target)
                source = present_rank.gather(1, present_target[:, None]).squeeze(1)
                decoy = present_rank.clone()
                decoy.scatter_(1, present_target[:, None], -1e4)
                margin = F.relu(self.rank_margin - source + decoy.max(1).values).mean()
                source_open = output["open_set_logits"][present].gather(
                    1, present_target[:, None],
                ).squeeze(1)
                present_open = F.relu(
                    self.rank_margin - source_open + output["unknown_logit"][present]
                ).mean()
            else:
                rank = joint.new_zeros(())
                margin = joint.new_zeros(())
                present_open = joint.new_zeros(())
            unknown = ~present
            if unknown.any():
                strongest_candidate = output["open_set_logits"][unknown, :candidate_count].max(1).values
                unknown_open = F.relu(
                    self.rank_margin - output["unknown_logit"][unknown] + strongest_candidate
                ).mean()
            else:
                unknown_open = joint.new_zeros(())
            open_margin = (present_open + unknown_open) / (int(present.any()) + int(unknown.any()))
            view = 0.05 * output["view_global_loss"] + 0.02 * output["view_local_loss"]
            total = (
                joint + 0.5 * exist + 0.5 * rank + 0.25 * candidate
                + 0.1 * margin + 0.2 * open_margin + view
            )
            return total, {
                "t2_total": total, "t2_joint": joint, "t2_exist": exist,
                "t2_rank": rank, "t2_candidate": candidate, "t2_margin": margin,
                "t2_open_margin": open_margin, "view": view,
            }
        if self.mode == "hierarchical_v3":
            exist = F.binary_cross_entropy_with_logits(output["exist_logit"], batch["exist_label"])
            candidate_target = torch.zeros_like(output["match_logits"])
            if present.any():
                candidate_target[present, target[present]] = 1
            candidate = balanced_bce(output["match_logits"], candidate_target, batch["set_mask"])
            absolute_parts = []
            if present.any():
                present_rank = output["rank_logits"][present]
                present_match = output["match_logits"][present]
                present_target = target[present]
                rank = F.cross_entropy(present_rank, present_target)
                source = present_rank.gather(1, present_target[:, None]).squeeze(1)
                decoy = present_rank.clone()
                decoy.scatter_(1, present_target[:, None], -1e4)
                margin = F.relu(self.rank_margin - source + decoy.max(1).values).mean()
                source_match = present_match.gather(1, present_target[:, None]).squeeze(1)
                absolute_parts.append(F.softplus(self.positive_margin - source_match).mean())
            else:
                rank = exist.new_zeros(())
                margin = exist.new_zeros(())
            unknown = ~present
            if unknown.any():
                unknown_match = output["match_logits"][unknown].masked_fill(~batch["set_mask"][unknown], -1e4)
                absolute_parts.append(F.softplus(unknown_match.max(1).values - self.negative_margin).mean())
            absolute = torch.stack(absolute_parts).mean() if absolute_parts else exist.new_zeros(())
            view = 0.05 * output["view_global_loss"] + 0.02 * output["view_local_loss"]
            total = exist + rank + 0.5 * candidate + 0.2 * margin + 0.2 * absolute + view
            return total, {
                "t2_total": total, "t2_exist": exist, "t2_rank": rank,
                "t2_candidate": candidate, "t2_margin": margin,
                "t2_absolute": absolute, "view": view,
            }
        if self.mode == "factorized_v2":
            # The V2 sampler balances existence labels over the stream. Per-batch
            # class weights are unstable when a memory-limited micro-batch has size one.
            exist = F.binary_cross_entropy_with_logits(output["exist_logit"], batch["exist_label"])
            if present.any():
                present_logits = output["rank_logits"][present]
                present_target = target[present]
                rank = F.cross_entropy(present_logits, present_target)
                source = present_logits.gather(1, present_target[:, None]).squeeze(1)
                decoy = present_logits.clone()
                decoy.scatter_(1, present_target[:, None], -1e4)
                margin = F.relu(self.rank_margin - source + decoy.max(1).values).mean()
            else:
                rank = exist.new_zeros(())
                margin = exist.new_zeros(())
            view = 0.05 * output["view_global_loss"] + 0.02 * output["view_local_loss"]
            factorized = exist + rank
            total = factorized + 0.2 * margin + view
            return total, {
                "t2_total": total, "t2_factorized": factorized, "t2_exist": exist,
                "t2_rank": rank, "t2_margin": margin, "view": view,
            }

        joint_target = target.where(target >= 0, torch.full_like(target, candidate_count))
        joint = F.nll_loss(output["joint_probability"].clamp_min(1e-8).log(), joint_target)
        exist = F.binary_cross_entropy_with_logits(output["exist_logit"], batch["exist_label"])
        if present.any():
            source = output["rank_logits"][present].gather(1, target[present, None]).squeeze(1)
            decoy = output["rank_logits"][present].clone()
            decoy.scatter_(1, target[present, None], -1e4)
            rank = F.relu(self.rank_margin - source + decoy.max(1).values).mean()
            open_present = F.relu(self.positive_margin - source).mean()
        else:
            rank = joint.new_zeros(())
            open_present = joint.new_zeros(())
        unknown = ~present
        open_unknown = (F.relu(output["rank_logits"][unknown].max(1).values - self.negative_margin).mean()
                        if unknown.any() else joint.new_zeros(()))
        open_loss = (open_present + open_unknown) / (int(present.any()) + int(unknown.any()))
        view = 0.05 * output["view_global_loss"] + 0.02 * output["view_local_loss"]
        total = joint + 0.5 * exist + 0.2 * rank + 0.1 * open_loss + view
        return total, {"t2_total": total, "t2_joint": joint, "t2_exist": exist,
                       "t2_rank": rank, "t2_open": open_loss, "view": view}


def project_conflicting(gradient_a: list[Tensor | None], gradient_b: list[Tensor | None]) -> tuple[list[Tensor | None], Tensor]:
    pairs = [(a, b) for a, b in zip(gradient_a, gradient_b) if a is not None and b is not None]
    if not pairs:
        return [None for _ in gradient_a], torch.zeros(())

    gradients = [gradient for pair in pairs for gradient in pair]
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("PCGrad received non-finite task gradients")

    # Global dot products can overflow float32 even when every gradient element
    # is finite. A shared scale cancels from all projection coefficients.
    scale = torch.stack([
        gradient.detach().abs().amax().float() for gradient in gradients
    ]).amax().clamp_min(1.0)
    normalized = [(a.float() / scale, b.float() / scale) for a, b in pairs]
    dot = sum((a * b).sum() for a, b in normalized)
    norm_a = sum((a * a).sum() for a, _ in normalized).clamp_min(1e-12)
    norm_b = sum((b * b).sum() for _, b in normalized).clamp_min(1e-12)
    if not bool(torch.isfinite(torch.stack([dot, norm_a, norm_b])).all()):
        raise FloatingPointError("PCGrad projection statistics are non-finite")
    cosine = (dot / (norm_a.sqrt() * norm_b.sqrt())).detach()
    conflict = bool((dot < 0).item())
    output: list[Tensor | None] = []
    for a, b in zip(gradient_a, gradient_b):
        if a is None and b is None:
            output.append(None)
        elif a is None:
            output.append(b)
        elif b is None:
            output.append(a)
        elif conflict:
            output.append(a - dot / norm_b * b + b - dot / norm_a * a)
        else:
            output.append(a + b)
    return output, cosine
