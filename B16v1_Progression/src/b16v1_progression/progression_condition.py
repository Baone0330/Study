from __future__ import annotations
import torch
from torch import nn

class ProgressionCondition(nn.Module):
    """Document-faithful 16-D progression + 64->128->64 fusion.

    The final 64->16 adapter is required only because the frozen B14 CPFM
    runtime contract accepts 16-D condition vectors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.delta_encoder = nn.Sequential(
            nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 16)
        )
        self.fusion = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 64)
        )
        self.cpfm_adapter = nn.Linear(64, 16)

    def forward(
        self, before: torch.Tensor, after: torch.Tensor, temporal: torch.Tensor,
        deltas: torch.Tensor, delta_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.delta_encoder(deltas)
        denom = delta_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        progression = (encoded * delta_mask.unsqueeze(-1)).sum(dim=1) / denom
        fused64 = self.fusion(torch.cat([before, after, temporal, progression], dim=-1))
        condition16 = torch.tanh(self.cpfm_adapter(fused64))
        return condition16, progression, fused64
