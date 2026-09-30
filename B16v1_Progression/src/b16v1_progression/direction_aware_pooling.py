"""Parameter-free direction-aware consensus pooling (DACP), version 1."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor

EPSILON = 1e-8
EPSILON_WEIGHT = 1e-6


def direction_aware_consensus_pooling(
    encoded: Tensor, mask: Tensor, *, diagnostics: bool = False
):
    """Aggregate [batch, pairs, features] using only valid encoded directions.

    u_i = E_i / (||E_i|| + 1e-8)
    s_i = mean_{j != i}(u_i dot u_j)
    a_i = (1 + s_i)/2 + 1e-6; w_i = a_i / sum(a)
    Empty histories produce zero, single transitions are returned unchanged.
    No gradient detachment: training differentiates through the direction scores.
    """
    valid = mask.to(encoded.dtype)
    count = valid.sum(dim=1, keepdim=True)
    norms = encoded.norm(dim=-1)
    unit = encoded / (norms.unsqueeze(-1) + EPSILON)
    similarity = torch.bmm(unit, unit.transpose(1, 2))
    pairs = valid.unsqueeze(2) * valid.unsqueeze(1)
    diagonal = torch.eye(encoded.shape[1], device=encoded.device, dtype=encoded.dtype).unsqueeze(0)
    scores = (similarity * pairs * (1 - diagonal)).sum(-1) / (count - 1).clamp_min(1)
    unnormalized = ((1 + scores) * 0.5 + EPSILON_WEIGHT) * valid
    weights = unnormalized / unnormalized.sum(dim=1, keepdim=True).clamp_min(EPSILON_WEIGHT)
    pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
    if diagnostics:
        return pooled, scores, weights, norms
    return pooled


def make_direction_aware_progression_class(original_class):
    """Integrate DACP by inheriting unchanged encoder/fusion/adapter modules."""
    class DirectionAwareProgressionCondition(original_class):
        def forward(self, before, after, temporal, deltas, delta_mask):
            encoded = self.delta_encoder(deltas)
            progression = direction_aware_consensus_pooling(encoded, delta_mask)
            fused64 = self.fusion(torch.cat([before, after, temporal, progression], dim=-1))
            condition16 = torch.tanh(self.cpfm_adapter(fused64))
            return condition16, progression, fused64
    return DirectionAwareProgressionCondition


def load_direction_aware_progression_state(path: str | Path, device='cpu'):
    """Explicit experimental DACP loader; never overrides default mean inference.

    Tensor shapes match the original model, so architecture choice must be
    explicit. A mean-pooling checkpoint does not become a validated DACP model
    merely because it can load with strict=True.
    """
    from .progression_condition import ProgressionCondition
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items()
    ):
        raise ValueError('Expected a tensor-only ProgressionCondition state_dict')
    model = make_direction_aware_progression_class(ProgressionCondition)()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
