"""Minimal zero-preserving progression residuals; no base-model changes."""
from __future__ import annotations

import math

import torch
from torch import nn

from b16v1_progression.b14_cpfm import B14CPFM


def gate_logit(initial: float = .10) -> float:
    if not 0 < initial < 1:
        raise ValueError('gate must start strictly between 0 and 1')
    return math.log(initial / (1 - initial))


class ZeroProjector(nn.Module):
    """Linear16x16(no bias) -> SiLU -> Linear16x16(no bias); r(0)=0."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(16, 16, bias=False), nn.SiLU(),
                                    nn.Linear(16, 16, bias=False))

    def forward(self, progression: torch.Tensor) -> torch.Tensor:
        if progression.shape[-1] != 16:
            raise ValueError('progression last dimension must be 16')
        return self.layers(progression)


class ZeroPreservingConditionResidual(nn.Module):
    """c_ZC = c_R + sigmoid(a_c) r_c(P), exactly c_R when P is zero."""

    def __init__(self, initial_gate: float = .10) -> None:
        super().__init__()
        self.projector = ZeroProjector()
        self.a_c = nn.Parameter(torch.tensor(gate_logit(initial_gate), dtype=torch.float32))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.a_c)

    def forward(self, c_base: torch.Tensor, progression: torch.Tensor) -> torch.Tensor:
        if c_base.shape[-1] != 16 or progression.shape[-1] != 16:
            raise ValueError('condition/progression last dimension must be 16')
        return c_base + self.gate * self.projector(progression)


class B14CPFMZeroPreservingResidual(B14CPFM):
    """Only add a zero-preserving residual to the original patient latent."""

    def __init__(self, initial_gate: float = .10) -> None:
        super().__init__()
        self.progression_projector = ZeroProjector()
        self.a_p = nn.Parameter(torch.tensor(gate_logit(initial_gate), dtype=torch.float32))
        self._active_progression: torch.Tensor | None = None

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.a_p)

    def set_progression(self, progression: torch.Tensor | None) -> None:
        if progression is not None and progression.shape[-1] != 16:
            raise ValueError('progression last dimension must be 16')
        self._active_progression = progression

    def patient_representation(self, condition: torch.Tensor,
                               history_summary: torch.Tensor) -> torch.Tensor:
        base = super().patient_representation(condition, history_summary)
        progression = self._active_progression
        if progression is None:
            return base
        progression = progression.to(device=base.device, dtype=base.dtype)
        if progression.ndim == 1:
            progression = progression[None]
        if progression.shape[0] == 1 and base.shape[0] != 1:
            progression = progression.expand(base.shape[0], -1)
        elif progression.shape[0] != base.shape[0]:
            raise ValueError('progression batch mismatch')
        # Keep the existing zero-condition reference path progression-free.
        c = condition if condition.ndim == 2 else condition[None]
        active = (c.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=base.dtype)
        return base + active * self.gate * self.progression_projector(progression)
