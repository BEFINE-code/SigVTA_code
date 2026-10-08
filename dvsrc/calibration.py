from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

def fit_binary_temperature(logits: Tensor, labels: Tensor) -> float:
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits.detach() / log_temperature.exp(), labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20).detach())


def fit_multiclass_temperature(logits: Tensor, targets: Tensor) -> float:
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits.detach() / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20).detach())


def eer_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    label_values = [int(value) for value in labels.tolist()]
    score_values = [float(value) for value in scores.tolist()]
    if len(label_values) != len(score_values):
        raise ValueError("EER labels and scores must have the same length")
    positive_scores = [
        score for label, score in zip(label_values, score_values) if label == 1
    ]
    negative_scores = [
        score for label, score in zip(label_values, score_values) if label == 0
    ]
    thresholds = sorted(set(score_values))
    best = (float("inf"), 0.0)
    for threshold in thresholds:
        far = (
            sum(score >= threshold for score in negative_scores) / len(negative_scores)
            if negative_scores else 0.0
        )
        frr = (
            sum(score < threshold for score in positive_scores) / len(positive_scores)
            if positive_scores else 0.0
        )
        if abs(far - frr) < best[0]:
            best = abs(far - frr), float(threshold)
    return best[1]


@dataclass
class Calibrator:
    t1_1v1_temperature: float = 1.0
    t1_5v1_temperature: float = 1.0
    t1_1v1_threshold: float = 0.5
    t1_5v1_threshold: float = 0.5
    t2_rank_temperature: float = 1.0
    t2_exist_temperature: float = 1.0
    t2_type_temperature: float = 1.0
    t2_exist_threshold: float = 0.5

    def calibrate_t1(self, logits: Tensor, references: int) -> Tensor:
        temperature = self.t1_1v1_temperature if references == 1 else self.t1_5v1_temperature
        return torch.sigmoid(logits.float() / temperature)

    def calibrate_t2(self, rank_logits: Tensor, exist_logits: Tensor,
                     type_logits: Tensor | None = None) -> Tensor:
        rank = torch.softmax(rank_logits / self.t2_rank_temperature, dim=-1)
        if type_logits is not None:
            exist = torch.softmax(type_logits / self.t2_type_temperature, dim=-1)[:, 0]
        else:
            exist = torch.sigmoid(exist_logits / self.t2_exist_temperature)
        return torch.cat([exist[:, None] * rank, (1 - exist)[:, None]], dim=1)
