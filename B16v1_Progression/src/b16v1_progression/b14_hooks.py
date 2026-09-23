from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

LAYER_NAMES = ['L1', 'L2', 'L3', 'L4']

def phase(step: int):
    return 'early' if int(step) < 10 else 'middle' if int(step) < 20 else 'late'


class B14Hooks:
    def __init__(self, unet, cpfm, holder):
        self.cpfm, self.holder = cpfm, holder
        self.handles = [unet.register_forward_pre_hook(self._unet_pre, with_kwargs=True)]
        for i in range(4): self.handles.append(unet.down_blocks[i].resnets[0].register_forward_pre_hook(self._make_hook(i)))

    def _unet_pre(self, _module, inputs, kwargs):
        t = kwargs.get("timestep") if isinstance(kwargs, dict) else None
        if t is None and len(inputs) > 1: t = inputs[1]
        if torch.is_tensor(t):
            tv = float(t.detach().float().reshape(-1)[0].cpu()); self.holder["t_norm"] = tv / 999.0
            sts = self.holder.get("scheduler_timesteps", [])
            if sts: self.holder["step_index"] = int(np.argmin(np.abs(np.asarray(sts) - tv)))

    def _active(self, i):
        if not self.holder.get("enabled", False): return False
        active = self.holder.get("active_layers")
        if active is not None and i not in active: return False
        wanted = self.holder.get("phase_only")
        return wanted is None or phase(self.holder.get("step_index", 0)) == wanted

    @staticmethod
    def _current_summary(x):
        xf = x.float(); vals = [xf.mean((1,2,3)), xf.std((1,2,3)), xf.abs().mean((1,2,3)), xf.abs().amax((1,2,3))]
        return torch.stack(vals, 1).repeat(1, 4)

    def _make_hook(self, layer_index):
        def hook(_module, inputs):
            if not self._active(layer_index) or not inputs or not torch.is_tensor(inputs[0]): return inputs
            x = inputs[0]; c = self.holder.get("condition")
            if c is None: return inputs
            c = c if torch.is_tensor(c) else torch.tensor(c, dtype=torch.float32); c = c[None] if c.ndim == 1 else c
            c = c.to(x.device); c = c.expand(x.shape[0], -1) if c.shape[0] == 1 and x.shape[0] != 1 else c[:1].expand(x.shape[0], -1) if c.shape[0] != x.shape[0] else c
            t = torch.full((x.shape[0],), float(self.holder.get("t_norm", .5)), device=x.device)
            hs = self.holder.get("history_summary", torch.zeros((1,16), device=x.device)); cs = self._current_summary(x)
            out = self.cpfm.effective(c, t, layer_index, hs, cs); gf,bf,grms,brms,ag,ab,dg,db,z = out
            mode = self.holder.get("component_mode", "full")
            if mode == "gamma_only": bf = torch.zeros_like(bf)
            elif mode == "beta_only": gf = torch.zeros_like(gf)
            ge = gf.to(x.dtype).view(x.shape[0], x.shape[1], 1, 1); be = bf.to(x.dtype).view(x.shape[0], x.shape[1], 1, 1)
            scale = torch.sigmoid(self.cpfm.layer_scale_logits[layer_index]).to(x.dtype).view(1,1,1,1); xf = x.float(); mean = xf.mean((2,3), keepdim=True); std = xf.std((2,3), keepdim=True).clamp_min(1e-5); u = (xf-mean)/std; mod = (1+ge.float())*u+be.float(); y = xf + scale.float()*(mod-u)
            td = self.holder.get("target_dirs", {}).get(layer_index); cg=cb=cc=np.nan
            if td is not None:
                target = torch.as_tensor(td, device=x.device, dtype=torch.float32).reshape(1,-1); cg=float(F.cosine_similarity(dg.float(), target, dim=1).mean().detach().cpu()); cb=float(F.cosine_similarity(db.float(), target, dim=1).mean().detach().cpu()); cc=float(F.cosine_similarity((gf+bf).float(), target, dim=1).mean().detach().cpu())
            self.holder.setdefault("capture", []).append({"step_index": int(self.holder.get("step_index",0)), "layer": LAYER_NAMES[layer_index], "layer_index": layer_index, "gamma_raw_rms": float(grms.mean().detach().cpu()), "beta_raw_rms": float(brms.mean().detach().cpu()), "a_gamma": float(ag.mean().detach().cpu()), "a_beta": float(ab.mean().detach().cpu()), "beta_safety": float(torch.sigmoid(self.cpfm.beta_safety_logit[layer_index]).detach().cpu()), "gamma_final_energy": float(gf.pow(2).mean().detach().cpu()), "beta_final_energy": float(bf.pow(2).mean().detach().cpu()), "total_mod_energy": float((gf.pow(2).mean()+bf.pow(2).mean()).detach().cpu()), "gamma_direction_cosine_target": cg, "beta_direction_cosine_target": cb, "combined_direction_cosine_target": cc, "component_mode": mode, "patient_repr_norm": float(z.norm(dim=1).mean().detach().cpu())})
            self.holder.setdefault("loss_terms", []).append((gf.pow(2).mean(), bf.pow(2).mean(), ag.mean(), ab.mean()))
            self.holder.setdefault("direction_terms", []).append({"layer": layer_index, "gamma": dg, "beta": db, "combined": gf+bf})
            self.holder["patient_repr"] = z
            return (y.to(x.dtype),) + tuple(inputs[1:])
        return hook

    def close(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles = []
