"""Portable wrapper of the audited Progression generation path.

The condition construction and diffusion settings follow the formal runtime.
Callers provide their own authorized history and external SD1.5/medical LoRA.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import numpy as np
import torch
from diffusers import PNDMScheduler, StableDiffusionPipeline

from .longitudinal_condition import LongitudinalConditionModel
from .progression_condition import ProgressionCondition
from .b14_cpfm import B14CPFM
from .b14_hooks import B14Hooks

STEPS = 30
GUIDANCE = 7.0
WIDTH = HEIGHT = 512
BASE_PROMPT = "a grayscale thyroid ultrasound image, clinical ultrasound"
NEGATIVE_PROMPT = "color, drawing, text, watermark, measurement calipers, annotation marks, overexposed, oversharpened, artificial texture, repeated pattern, low quality, artifacts"


def _load_state(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("model_state_dict", payload.get("model", payload))


def load_runtime_stack(base_model, lora, longitudinal_ckpt, cpfm_ckpt, progression_ckpt, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16, local_files_only=True, safety_checker=None, requires_safety_checker=False)
    pipe.load_lora_weights(lora)
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    if str(pipe.scheduler.config.prediction_type) != "epsilon":
        raise RuntimeError("SD scheduler prediction_type must be epsilon")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    longitudinal = LongitudinalConditionModel().cpu().eval()
    longitudinal.load_state_dict(_load_state(longitudinal_ckpt), strict=True)
    progression = ProgressionCondition().cpu().eval()
    progression.load_state_dict(_load_state(progression_ckpt), strict=True)
    cpfm = B14CPFM().to(device).eval()
    cpfm.load_state_dict(_load_state(cpfm_ckpt), strict=True)
    for model in (longitudinal, progression, cpfm, pipe.unet, pipe.vae, pipe.text_encoder):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    pipe.scheduler.set_timesteps(STEPS, device=device)
    timesteps = [int(x) for x in pipe.scheduler.timesteps.detach().cpu().tolist()]
    holder = {"cpfm": cpfm, "enabled": False, "scheduler_timesteps": timesteps,
              "component_mode": "longitudinal_only", "capture": [], "loss_terms": [],
              "direction_terms": [], "target_dirs": {}, "history_summary": torch.zeros((1, 16))}
    hooks = B14Hooks(pipe.unet, cpfm, holder)
    return {"pipe": pipe, "condition_model": longitudinal, "progression_model": progression,
            "cpfm": cpfm, "holder": holder, "hooks": hooks}


def make_condition(stack, history, query_date, annual_index):
    qdate = date.fromisoformat(str(query_date)[:10])
    history = [exam for exam in history if date.fromisoformat(str(exam["exam_date"])[:10]) < qdate]
    if not history:
        raise ValueError("no strict pre-query real history")
    before = [path for exam in history[:-1] for path in exam["image_paths"]]
    after = list(history[-1]["image_paths"])
    task = {"before_exam_date": history[-2]["exam_date"] if len(history)>1 else history[-1]["exam_date"],
            "after_exam_date": history[-1]["exam_date"], "query_date": query_date,
            "anchor_date": history[0]["exam_date"], "annual_index": int(annual_index)}
    frozen = stack["condition_model"]
    with torch.no_grad():
        before_repr, _ = frozen.before_encoder(before, [])
        after_repr, _ = frozen.after_encoder(after, [])
        temporal, _ = frozen.query_encoder(task)
        exam_features = [frozen.after_encoder(exam["image_paths"], [])[0] for exam in history]
        if len(exam_features) > 1:
            deltas = torch.stack([exam_features[i] - exam_features[i-1] for i in range(1,len(exam_features))])[None]
            mask = torch.ones((1,deltas.shape[1]))
        else:
            deltas = torch.zeros((1,1,16)); mask = torch.zeros((1,1))
        vector, _, _ = stack["progression_model"](before_repr[None], after_repr[None], temporal[None], deltas, mask)
        old = frozen.bridge(before_repr[None], after_repr[None], temporal[None])
    condition_scale = float(np.clip(0.55 + float(old["scale_delta"][0]), 0.50, 0.60))
    return {"condition_vector": vector[0].detach().cpu().float(), "condition_scale": condition_scale}


def generate_one(stack, condition, output_png, seed=42):
    pipe, holder = stack["pipe"], stack["holder"]
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    holder.update({"enabled": True, "condition": condition["condition_vector"],
                   "component_mode": "longitudinal_only", "history_summary": torch.zeros((1,16)),
                   "capture": [], "loss_terms": [], "direction_terms": [], "target_dirs": {}})
    try:
        with torch.inference_mode():
            result = pipe(prompt=BASE_PROMPT, negative_prompt=NEGATIVE_PROMPT,
                          num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                          height=HEIGHT, width=WIDTH, generator=generator)
        output = Path(output_png)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].convert("RGB").save(output)
        return output
    finally:
        holder["enabled"] = False
