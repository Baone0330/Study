"""Synthetic-only load and forward checks; no clinical input."""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from b16v1_progression import LongitudinalConditionModel, ProgressionCondition, B14CPFM, B14Hooks

def state(path):
    return torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--longitudinal_ckpt", required=True)
    p.add_argument("--cpfm_ckpt", required=True)
    p.add_argument("--progression_ckpt", required=True)
    a=p.parse_args()
    longitudinal=LongitudinalConditionModel().eval()
    progression=ProgressionCondition().eval()
    cpfm=B14CPFM().eval()
    for model,path in ((longitudinal,a.longitudinal_ckpt),(progression,a.progression_ckpt),(cpfm,a.cpfm_ckpt)):
        model.load_state_dict(state(path), strict=True)
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        path=Path(tmp)/"synthetic.png"
        Image.fromarray(np.full((32,32), 127, dtype=np.uint8)).save(path)
        b,_=longitudinal.before_encoder([str(path)], [])
        a_repr,_=longitudinal.after_encoder([str(path)], [])
        t,_=longitudinal.query_encoder({"query_date":"2026-01-01", "anchor_date":"2024-01-01", "before_exam_date":"2024-01-01", "after_exam_date":"2025-01-01", "annual_index":2})
        c,prog,fused=progression(b[None],a_repr[None],t[None],torch.zeros((1,1,16)),torch.zeros((1,1)))
        out=cpfm.effective(c, torch.tensor([.5]), 0, torch.zeros((1,16)), torch.zeros((1,16)))
        assert (b.shape,a_repr.shape,t.shape,c.shape,prog.shape,fused.shape,out[0].shape)==((16,),(16,),(16,),(1,16),(1,16),(1,64),(1,320))
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.resnets=torch.nn.ModuleList([torch.nn.Identity()])
        class UNet(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.down_blocks=torch.nn.ModuleList([Block() for _ in range(4)])
            def forward(self, sample, timestep=None):
                return [self.down_blocks[i].resnets[0](torch.zeros((1,channels,4,4))) for i,channels in enumerate((320,320,640,1280))]
        unet=UNet()
        holder={"enabled":True,"condition":c[0],"component_mode":"longitudinal_only",
                "history_summary":torch.zeros((1,16)),"scheduler_timesteps":[500],"capture":[],
                "loss_terms":[],"direction_terms":[],"target_dirs":{}}
        hooks=B14Hooks(unet,cpfm,holder)
        try:
            unet(torch.zeros((1,4,4,4)),timestep=torch.tensor(500))
            assert len(holder["capture"])==4
        finally:
            hooks.close()
    print("PASS: strict checkpoint loads and synthetic condition/CPFM/hook forward")

if __name__ == "__main__":
    main()
