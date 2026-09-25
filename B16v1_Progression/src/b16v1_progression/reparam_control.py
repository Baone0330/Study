"""B16v1-Reparam-Control: model-only condition reparameterization control.

This module has no clinical-data loading, training launcher, or diffusion runtime.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ReparamOnly(nn.Module):
    """Frozen-shape 64→128→64→16 condition mapper with progression input zeroed."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        self.cpfm_adapter = nn.Linear(64, 16)

    def forward(self, before: torch.Tensor, after: torch.Tensor,
                temporal: torch.Tensor) -> torch.Tensor:
        if before.shape[-1] != 16 or after.shape[-1] != 16 or temporal.shape[-1] != 16:
            raise ValueError('before, after, and temporal must each end in 16 features')
        zero = torch.zeros_like(before)
        fused = self.fusion(torch.cat((before, after, temporal, zero), dim=-1))
        return torch.tanh(self.cpfm_adapter(fused))


def load_reparam_control(weight_path: str | Path,
                         *, map_location: str | torch.device = 'cpu') -> ReparamOnly:
    """Load a release-safe, state-dict-only checkpoint with strict key validation."""
    state = torch.load(Path(weight_path), map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('expected a state-dict-only tensor checkpoint')
    model = ReparamOnly()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
