"""Minimal, independently switchable B16v1 progression transmission adapters.

The frozen B14CPFM computation and all U-Net hook positions remain unchanged.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from b16v1_progression.b14_cpfm import B14CPFM


def _gate_logit(initial: float = 0.10) -> float:
    if not 0.0 < initial < 1.0:
        raise ValueError('gate initialization must lie strictly within (0, 1)')
    return math.log(initial / (1.0 - initial))


class ProgressionResidualAdapter(nn.Module):
    """LayerNorm(16) → Linear(16,16) → tanh, with bounded scalar gate."""

    def __init__(self, initial_gate: float = 0.10) -> None:
        super().__init__()
        self.mapping = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16), nn.Tanh())
        self.a_c = nn.Parameter(torch.tensor(_gate_logit(initial_gate), dtype=torch.float32))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.a_c)

    def forward(self, c_base: torch.Tensor, progression16: torch.Tensor) -> torch.Tensor:
        if c_base.shape[-1] != 16 or progression16.shape[-1] != 16:
            raise ValueError('condition and progression must end in 16 features')
        residual = self.mapping(progression16)
        return torch.tanh(c_base + self.gate * residual)


class B14CPFMProgressionResidual(B14CPFM):
    """Inject progression at the existing patient-direction latent z only.

    The original B14CPFM effective(), normalization, magnitude heads, layer
    gates, gamma/beta dimensions, and external B14Hooks are inherited intact.
    """

    def __init__(self, initial_gate: float = 0.10) -> None:
        super().__init__()
        self.progression_projection = nn.Sequential(
            nn.LayerNorm(16), nn.Linear(16, 16), nn.SiLU()
        )
        self.a_p = nn.Parameter(torch.tensor(_gate_logit(initial_gate), dtype=torch.float32))
        self._active_progression: torch.Tensor | None = None

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.a_p)

    def set_progression(self, progression16: torch.Tensor | None) -> None:
        if progression16 is not None and progression16.shape[-1] != 16:
            raise ValueError('progression must end in 16 features')
        self._active_progression = progression16

    def patient_representation(self, condition: torch.Tensor,
                               history_summary: torch.Tensor) -> torch.Tensor:
        z = super().patient_representation(condition, history_summary)
        p = self._active_progression
        if p is None:
            return z
        p = p.to(device=z.device, dtype=z.dtype)
        if p.ndim == 1:
            p = p[None]
        if p.shape[0] == 1 and z.shape[0] != 1:
            p = p.expand(z.shape[0], -1)
        elif p.shape[0] != z.shape[0]:
            raise ValueError('progression batch does not match CPFM batch')
        # effective() subtracts a zero-condition reference branch. Keep that
        # reference progression-free so a genuine P residual does not cancel.
        c = condition if condition.ndim == 2 else condition[None]
        active = (c.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=z.dtype)
        return z + active * self.gate * self.progression_projection(p)
