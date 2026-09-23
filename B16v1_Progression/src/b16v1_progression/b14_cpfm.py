from __future__ import annotations
import math
import torch
from torch import nn

LAYER_CHANNELS = [320, 320, 640, 1280]
EPS = 1e-6

class B14CPFM(nn.Module):
    """B13 controller plus a zero-initialized patient-state residual branch."""
    def __init__(self, condition_dim=16, hidden=128, layer_dim=8, time_dim=8):
        super().__init__()
        # Keep exact B13 namespace for warm-start compatibility.
        self.condition_dim, self.hidden, self.layer_dim, self.time_dim = condition_dim, hidden, layer_dim, time_dim
        self.layer_embedding = nn.Embedding(4, layer_dim)
        in_dim = condition_dim + layer_dim + time_dim
        self.trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.gamma_heads = nn.ModuleList([nn.Linear(hidden, c) for c in LAYER_CHANNELS])
        self.beta_heads = nn.ModuleList([nn.Linear(hidden, c) for c in LAYER_CHANNELS])
        self.layer_scale_logits = nn.Parameter(torch.full((4,), -2.0))
        self.gamma_magnitude_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in LAYER_CHANNELS])
        self.beta_magnitude_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in LAYER_CHANNELS])
        self.beta_safety_logit = nn.Parameter(torch.full((4,), -2.0))
        for heads in (self.gamma_magnitude_heads, self.beta_magnitude_heads):
            for head in heads:
                nn.init.zeros_(head.weight); nn.init.constant_(head.bias, -2.0)
        self.history_projection = nn.Linear(16, 16)
        self.current_projection = nn.Linear(16, 16)
        direction_in = condition_dim + 16 + 16 + time_dim + layer_dim
        self.direction_trunk = nn.Sequential(nn.LayerNorm(direction_in), nn.Linear(direction_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.gamma_direction_heads = nn.ModuleList([nn.Linear(hidden, c) for c in LAYER_CHANNELS])
        self.beta_direction_heads = nn.ModuleList([nn.Linear(hidden, c) for c in LAYER_CHANNELS])
        for heads in (self.gamma_direction_heads, self.beta_direction_heads):
            for head in heads:
                nn.init.xavier_uniform_(head.weight, gain=.05); nn.init.zeros_(head.bias)
        # New patient branch.  Zero initialization preserves B13 output before training.
        self.patient_projection = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 16))
        self.patient_direction_trunk = nn.Sequential(nn.LayerNorm(condition_dim + 16 + 16 + time_dim + layer_dim), nn.Linear(condition_dim + 16 + 16 + time_dim + layer_dim, 64), nn.SiLU())
        self.patient_gamma_heads = nn.ModuleList([nn.Linear(64, c) for c in LAYER_CHANNELS])
        self.patient_beta_heads = nn.ModuleList([nn.Linear(64, c) for c in LAYER_CHANNELS])
        for heads in (self.patient_gamma_heads, self.patient_beta_heads):
            for head in heads:
                nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def time_embedding(self, t, device, dtype):
        t = t.reshape(-1, 1).to(device=device, dtype=dtype)
        return torch.cat([t, t*t, torch.sin(math.pi*t), torch.cos(math.pi*t), torch.sin(2*math.pi*t), torch.cos(2*math.pi*t), torch.sin(4*math.pi*t), torch.cos(4*math.pi*t)], dim=1)

    def _expand(self, x, n, device):
        x = x.to(device=device, dtype=torch.float32)
        if x.ndim == 1: x = x[None]
        return x.expand(n, -1) if x.shape[0] == 1 and n != 1 else x[:n]

    def b12_raw_forward(self, condition, t_norm, layer_index):
        p = next(self.parameters()); c = self._expand(condition, condition.shape[0] if condition.ndim > 1 else 1, p.device)
        n = c.shape[0]; tt = t_norm if torch.is_tensor(t_norm) else torch.tensor([t_norm], device=p.device); tt = tt.reshape(-1).to(p.device)
        te = self.time_embedding(tt, p.device, c.dtype); te = te.expand(n, -1) if te.shape[0] == 1 and n != 1 else te
        li = torch.full((n,), int(layer_index), device=p.device, dtype=torch.long); h = self.trunk(torch.cat([c, te, self.layer_embedding(li)], 1))
        return self.gamma_heads[layer_index](h), self.beta_heads[layer_index](h), torch.sigmoid(self.gamma_magnitude_heads[layer_index](h)), torch.sigmoid(self.beta_magnitude_heads[layer_index](h))

    def _summaries(self, condition, t_norm, layer_index, history_summary, current_summary):
        p = next(self.parameters()); c = self._expand(condition, condition.shape[0] if condition.ndim > 1 else 1, p.device); n = c.shape[0]
        hs = self._expand(history_summary, n, p.device); cs = self._expand(current_summary, n, p.device)
        tt = t_norm if torch.is_tensor(t_norm) else torch.tensor([t_norm], device=p.device); tt = tt.reshape(-1).to(p.device); te = self.time_embedding(tt, p.device, c.dtype); te = te.expand(n, -1) if te.shape[0] == 1 and n != 1 else te
        li = torch.full((n,), int(layer_index), device=p.device, dtype=torch.long)
        return c, hs, cs, te, self.layer_embedding(li)

    def direction_raw_forward(self, condition, t_norm, layer_index, history_summary, current_summary):
        c, hs, cs, te, le = self._summaries(condition, t_norm, layer_index, history_summary, current_summary)
        h = self.direction_trunk(torch.cat([c, self.history_projection(hs), self.current_projection(cs), te, le], 1))
        return self.gamma_direction_heads[layer_index](h), self.beta_direction_heads[layer_index](h)

    def patient_representation(self, condition, history_summary):
        c = condition if condition.ndim > 1 else condition[None]; hs = history_summary if history_summary.ndim > 1 else history_summary[None]
        c = c.to(next(self.parameters()).device, dtype=torch.float32); hs = hs.to(c.device, dtype=torch.float32)
        if hs.shape[0] == 1 and c.shape[0] != 1: hs = hs.expand(c.shape[0], -1)
        return self.patient_projection(torch.cat([c, hs], 1))

    def patient_raw_forward(self, condition, t_norm, layer_index, history_summary, current_summary):
        c, hs, cs, te, le = self._summaries(condition, t_norm, layer_index, history_summary, current_summary)
        z = self.patient_representation(c, hs)
        h = self.patient_direction_trunk(torch.cat([c, z, self.current_projection(cs), te, le], 1))
        return self.patient_gamma_heads[layer_index](h), self.patient_beta_heads[layer_index](h), z

    def effective(self, condition, t_norm, layer_index, history_summary, current_summary, patient_scale=.25):
        _, _, ag, ab = self.b12_raw_forward(condition, t_norm, layer_index)
        zero_c = torch.zeros_like(condition)
        dg_c, db_c = self.direction_raw_forward(condition, t_norm, layer_index, history_summary, current_summary)
        dg_z, db_z = self.direction_raw_forward(zero_c, t_norm, layer_index, history_summary, current_summary)
        pg_c, pb_c, zc = self.patient_raw_forward(condition, t_norm, layer_index, history_summary, current_summary)
        pg_z, pb_z, _ = self.patient_raw_forward(zero_c, t_norm, layer_index, history_summary, current_summary)
        dg_raw = (dg_c - dg_z) + patient_scale * (pg_c - pg_z); db_raw = (db_c - db_z) + patient_scale * (pb_c - pb_z)
        dg_rms = torch.sqrt(dg_raw.pow(2).mean(1, keepdim=True) + EPS); db_rms = torch.sqrt(db_raw.pow(2).mean(1, keepdim=True) + EPS)
        dg = torch.where(dg_raw.abs().sum(1, keepdim=True) == 0, torch.zeros_like(dg_raw), dg_raw / dg_rms); db = torch.where(db_raw.abs().sum(1, keepdim=True) == 0, torch.zeros_like(db_raw), db_raw / db_rms)
        zero_mask = condition.abs().sum(1, keepdim=True) == 0; ag_eff = torch.where(zero_mask, torch.zeros_like(ag), ag); ab_eff = torch.where(zero_mask, torch.zeros_like(ab), ab)
        ab_final = ab_eff * torch.sigmoid(self.beta_safety_logit[layer_index]).expand_as(ab_eff)
        return ag_eff * dg, ab_final * db, dg_rms, db_rms, ag_eff, ab_final, dg, db, zc
