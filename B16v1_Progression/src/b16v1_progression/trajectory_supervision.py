"""Existing B16v1 supervision terms with fixed trajectory-priority weights.

This module does not change ProgressionCondition or add a new loss type.
Inputs and targets must be historical pre-target representations.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor
from torch.nn import functional as F

from .progression_condition import ProgressionCondition

COMPONENTS = ("teacher_mse", "teacher_cos", "delta_mse", "delta_cos", "compatibility")
SUPERVISION_ARMS = {
    "S0": dict(zip(COMPONENTS, (1.00, 0.20, 0.50, 0.10, 0.10))),
    "S1": dict(zip(COMPONENTS, (0.50, 0.10, 1.00, 0.20, 0.10))),
    "S2": dict(zip(COMPONENTS, (0.25, 0.05, 1.20, 0.30, 0.10))),
    "S3": dict(zip(COMPONENTS, (0.50, 0.10, 1.10, 0.20, 0.00))),
}
RESEARCH_CANDIDATE_ARM = "S2"


def existing_supervision_terms(
    condition16: Tensor,
    progression16: Tensor,
    teacher_condition16: Tensor,
    historical_next_delta16: Tensor,
    compatibility_condition16: Tensor,
) -> dict[str, Tensor]:
    """Original batch-reduced MSE/cosine terms, without target redefinition."""
    return {
        "teacher_mse": F.mse_loss(condition16, teacher_condition16),
        "teacher_cos": 1 - F.cosine_similarity(condition16, teacher_condition16, dim=-1).mean(),
        "delta_mse": F.mse_loss(progression16, historical_next_delta16),
        "delta_cos": 1 - F.cosine_similarity(progression16, historical_next_delta16, dim=-1).mean(),
        "compatibility": F.mse_loss(condition16, compatibility_condition16),
    }


def weighted_supervision_loss(terms: dict[str, Tensor], arm: str = "S2") -> Tensor:
    """Apply one of the four fixed coefficient budgets (sum 1.90)."""
    weights = SUPERVISION_ARMS[arm]
    return (
        weights["teacher_mse"] * terms["teacher_mse"]
        + weights["teacher_cos"] * terms["teacher_cos"]
        + weights["delta_mse"] * terms["delta_mse"]
        + weights["delta_cos"] * terms["delta_cos"]
        + weights["compatibility"] * terms["compatibility"]
    )


def load_trajectory_supervision_candidate(
    path: str | Path, device: str | torch.device = "cpu"
) -> ProgressionCondition:
    """Load a tensor-only candidate into the unchanged original architecture."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()
    ):
        raise ValueError("Expected a nonempty tensor-only ProgressionCondition state_dict")
    model = ProgressionCondition()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
