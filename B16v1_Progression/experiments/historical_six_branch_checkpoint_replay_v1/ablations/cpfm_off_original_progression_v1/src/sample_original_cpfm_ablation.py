"""Full, original-batch paired pilot; runtime remains the original loader/source."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cpfm_common import atomic_json, load_module, now, protected_postflight, read_json, write_csv
from cpfm_toggle_adapter import hook_count, temporary_cpfm_enabled


def difference(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    assert np.isfinite(d).all()
    return {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d * d).mean())),
            "max_abs": float(np.abs(d).max()), "bitexact": bool(np.array_equal(a, b))}


def original_runtime(cfg):
    mod = load_module("cpfm_ablation_original_generator", cfg["paths"]["original_generator"])
    stack = mod.load_runtime_stack()
    original_hooks = type(stack["hooks"])
    stack["hooks"].close()
    assert hook_count(stack["pipe"].unet) == 0
    stack["original_hooks_class"] = original_hooks
    stack["original_module"] = mod
    assert all(not p.requires_grad for m in (stack["condition_model"], stack["progression_model"], stack["cpfm"], stack["pipe"].unet, stack["pipe"].vae, stack["pipe"].text_encoder) for p in m.parameters())
    return stack


def capture_condition(stack, record, node):
    captured = {}
    handles = []
    names = ("before_encoder", "after_encoder", "query_encoder")
    for name in names:
        module = getattr(stack["condition_model"], name)
        def observe(_m, _i, out, key=name):
            captured.setdefault(key, []).append(out[0].detach().cpu().clone())
        handles.append(module.register_forward_hook(observe))
    def observe_progression(_m, _i, out):
        captured["condition16"] = out[0].detach().cpu().clone()
        captured["progression16"] = out[1].detach().cpu().clone()
        captured["fusion64"] = out[2].detach().cpu().clone()
    handles.append(stack["progression_model"].register_forward_hook(observe_progression))
    try:
        condition, provenance = stack["original_module"].make_condition(stack, record, node["query_date"], node["annual_slot_index"])
    finally:
        for handle in handles:
            handle.remove()
    assert all(torch.isfinite(t).all() for values in captured.values() for t in (values if isinstance(values, list) else [values]))
    return condition, provenance, captured


def equal_condition(a, b):
    assert set(a) == set(b)
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, list):
            assert len(x) == len(y) and all(torch.equal(i, j) for i, j in zip(x, y)), key
        else:
            assert torch.equal(x, y), key
    return True


def sample_batch(stack, condition, node, enabled, outputs, detailed=True, preserve_existing=()):
    """All diffusion kwargs match original generate_batch. No supplied latents or gain."""
    base = stack["original_module"].base
    pipe, holder = stack["pipe"], stack["holder"]
    holder.update({"condition": condition["condition_vector"], "component_mode": "longitudinal_only", "history_summary": torch.zeros(1, 16),
                   "capture": [], "loss_terms": [], "direction_terms": [], "target_dirs": {}})
    traces = {"unet_noise": [], "unet_input": [], "prompt_embedding": [], "final_latent": None}
    handles = []
    initial = {}
    original_prepare = pipe.prepare_latents
    def prepare(*args, **kwargs):
        latents = original_prepare(*args, **kwargs)
        initial["latents"] = latents.detach().cpu().clone()
        return latents
    pipe.prepare_latents = prepare
    if detailed:
        def text_observer(_module, _inputs, output):
            traces["prompt_embedding"].append(output[0].detach().cpu().clone())
        handles.append(pipe.text_encoder.register_forward_hook(text_observer))
        def pre_observer(_module, inputs, kwargs):
            x = kwargs.get("sample", inputs[0] if inputs else None)
            traces["unet_input"].append(x.detach().cpu().clone())
        handles.append(pipe.unet.register_forward_pre_hook(pre_observer, with_kwargs=True))
        def out_observer(_module, _inputs, output):
            noise = output[0] if isinstance(output, tuple) else output.sample
            traces["unet_noise"].append(noise.detach().cpu().clone())
        handles.append(pipe.unet.register_forward_hook(out_observer))
    def callback(_pipeline, _step, _timestep, callback_kwargs):
        traces["final_latent"] = callback_kwargs["latents"].detach().cpu().clone()
        return callback_kwargs
    generators = [torch.Generator(device="cuda").manual_seed(int(s)) for s in node["seeds"]]
    audit = {}
    started = time.perf_counter()
    try:
        with temporary_cpfm_enabled(pipe.unet, stack["cpfm"], holder, stack["original_hooks_class"], enabled, audit, capture_residuals=detailed):
            with torch.inference_mode():
                result = pipe(prompt=[base.BASE_PROMPT] * len(outputs), negative_prompt=[base.NEGATIVE_PROMPT] * len(outputs),
                              num_inference_steps=base.STEPS, guidance_scale=base.GUIDANCE, height=base.HEIGHT, width=base.WIDTH,
                              num_images_per_prompt=1, generator=generators, callback_on_step_end=callback)
        assert len(result.images) == len(outputs)
        pixels = []
        for image, output in zip(result.images, outputs):
            rgb = image.convert("RGB")
            array = np.asarray(rgb).copy()
            assert rgb.size == (512, 512) and array.shape == (512, 512, 3) and array.size > 0
            if output in preserve_existing:
                with Image.open(output) as previous:
                    assert np.array_equal(np.asarray(previous.convert("RGB")), array), "Completed output replay differs; do not overwrite"
                pixels.append(array)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_name(output.name + ".tmp")
            rgb.save(tmp, format="PNG")
            os.replace(tmp, output)
            pixels.append(array)
    finally:
        pipe.prepare_latents = original_prepare
        for handle in handles:
            handle.remove()
        holder["enabled"] = False
        assert hook_count(pipe.unet) == 0
    traces["initial_latent"] = initial["latents"]
    assert traces["final_latent"] is not None and torch.isfinite(traces["final_latent"]).all()
    assert audit["physical_prehook_calls_by_layer"] == [len(stack["timesteps"])] * 4
    assert audit["effective_modulations"] == (4 * len(stack["timesteps"]) if enabled else 0)
    if detailed:
        assert len(traces["unet_noise"]) == len(stack["timesteps"])
        assert audit["physical_prehook_calls_by_layer"] == [len(stack["timesteps"])] * 4
        assert audit["effective_modulations"] == (4 * len(stack["timesteps"]) if enabled else 0)
        audit["effective_gamma_beta_outputs"] = [{k: v for k, v in r.items() if k in ("step_index", "layer_index", "gamma_final_energy", "beta_final_energy", "a_gamma", "a_beta")} for r in holder["capture"]]
    audit.update({"elapsed_seconds": time.perf_counter() - started, "unet_calls": len(stack["timesteps"]),
                  "own_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "own_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "P16_conditions_computed": True, "P16_condition_reaches_UNet_without_CPFM": False,
                  "independent_hooks_calls": 0, "medical_review": "NOT_RUN", "batch": len(outputs)})
    return traces, np.stack(pixels), audit


def pilot(root, repeat_only=False):
    root = Path(root)
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    nodes = read_json(root / "manifests/FROZEN_NODES_V1.json")
    records = read_json(root / "manifests/FROZEN_REAL_HISTORY_RECORDS_V1.json")
    selected = sorted(nodes, key=lambda n: n["node_id"])[:6]
    assert [n["node_id"] for n in selected] == cfg["pilot_node_ids"]
    if repeat_only:
        selected = selected[:1]
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    if free < 16 * 1024**3:
        raise RuntimeError("Pilot requires at least 16 GiB currently free; preserve other jobs")
    torch.cuda.reset_peak_memory_stats()
    stack = original_runtime(cfg)
    init_qa = {"all_source_modules_frozen": True, "unet_calls_per_sampling": len(stack["timesteps"]),
               "batch": 6, "scheduler_class": type(stack["pipe"].scheduler).__name__, "prediction_type": stack["pipe"].scheduler.config.prediction_type,
               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
               "scheduler_timesteps": stack["timesteps"], "free_memory_at_start_bytes": free, "total_memory_bytes": total,
               "historical_runtime_exact_revision": "UNKNOWN", "repeat_new_process": repeat_only, "started_utc": now()}
    atomic_json(root / f"pilot/qa/{'REPEAT' if repeat_only else 'PILOT'}_RUNTIME_LOAD_V1.json", init_qa)
    paired = []
    repeat_qa = []
    try:
        for ix, node in enumerate(selected, 1):
            captures = {}
            trace_store = {}
            for enabled, name in ((True, "ON"), (False, "OFF")):
                condition, provenance, capture = capture_condition(stack, records[node["patient_id"]], node)
                captures[name] = capture
                if name == "OFF":
                    equal_condition(captures["ON"], capture)
                phase = "repeat" if repeat_only else "fixed_36_pairs"
                outputs = [root / "pilot" / phase / f"node_{ix:02d}" / name / f"gen_{i:02d}.png" for i in range(6)]
                trace, pixels, audit = sample_batch(stack, condition, node, enabled, outputs)
                trace["condition_components"] = capture
                trace["rgb_uint8"] = torch.from_numpy(pixels)
                trace_path = root / "pilot/traces" / f"{'repeat_' if repeat_only else ''}node_{ix:02d}_{name}.pt"
                torch.save(trace, trace_path)
                atomic_json(trace_path.with_suffix(".json"), {"node_key_private": node["node_id"], "branch": name, "seeds": node["seeds"], **audit})
                trace_store[name] = (trace, pixels, audit)
                if repeat_only:
                    old = torch.load(root / "pilot/traces" / f"node_{ix:02d}_{name}.pt", map_location="cpu", weights_only=True)
                    equal_condition(old["condition_components"], capture)
                    assert torch.equal(old["initial_latent"], trace["initial_latent"])
                    for i in range(6):
                        repeat_qa.append({"branch": name, "image_index": i, **{f"repeat_latent_{k}": v for k, v in difference(old["final_latent"][i].numpy(), trace["final_latent"][i].numpy()).items()},
                                          **{f"repeat_pixel_{k}": v for k, v in difference(old["rgb_uint8"][i].numpy(), pixels[i]).items()}})
            on, on_pixels, on_audit = trace_store["ON"]
            off, off_pixels, off_audit = trace_store["OFF"]
            assert torch.equal(on["initial_latent"], off["initial_latent"])
            assert len(on["prompt_embedding"]) == len(off["prompt_embedding"])
            assert all(torch.equal(x, y) for x, y in zip(on["prompt_embedding"], off["prompt_embedding"]))
            for i in range(6):
                with Image.open(node["reference_on_paths"][i]) as im:
                    reference = np.asarray(im.convert("RGB"))
                historical = difference(on_pixels[i], reference)
                rgb = difference(on_pixels[i], off_pixels[i])
                latent = difference(on["final_latent"][i].numpy(), off["final_latent"][i].numpy())
                # Separate unconditional and text-conditional halves of first full UNet call.
                unet_u = difference(on["unet_noise"][0][i].numpy(), off["unet_noise"][0][i].numpy())
                unet_c = difference(on["unet_noise"][0][i + 6].numpy(), off["unet_noise"][0][i + 6].numpy())
                on_feat = stack["original_module"].base.make_condition  # retain original source object; no target image input
                from longitudinal_condition import image_feature
                a_feat = image_feature(root / "pilot" / ("repeat" if repeat_only else "fixed_36_pairs") / f"node_{ix:02d}/ON" / f"gen_{i:02d}.png")
                b_feat = image_feature(root / "pilot" / ("repeat" if repeat_only else "fixed_36_pairs") / f"node_{ix:02d}/OFF" / f"gen_{i:02d}.png")
                features = difference(a_feat, b_feat)
                paired.append({"node_key_private": node["node_id"], "patient_id_private": node["patient_id"], "image_index": i, "master_seed": node["seeds"][i],
                               "condition16_bitexact": True, "all_condition_components_bitexact": True, "prompt_bitexact": True, "initial_latent_bitexact": True,
                               "on_effective_modulations": on_audit["effective_modulations"], "off_effective_modulations": off_audit["effective_modulations"],
                               **{f"historical_on_pixel_{k}": v for k, v in historical.items()}, **{f"on_off_pixel_{k}": v for k, v in rgb.items()},
                               **{f"on_off_latent_{k}": v for k, v in latent.items()}, **{f"early_unet_uncond_{k}": v for k, v in unet_u.items()},
                               **{f"early_unet_cond_{k}": v for k, v in unet_c.items()}, **{f"generated_image_gray8_{k}": v for k, v in features.items()},
                               "zero_pixel_response": rgb["bitexact"], "actual_batch": 6})
            print(json.dumps({"stage": "repeat" if repeat_only else "pilot", "nodes_completed": ix, "nodes_total": len(selected), "paired_images": 6 * ix,
                              "historical_on_pixel_bitexact_count": sum(r["historical_on_pixel_bitexact"] for r in paired),
                              "condition_parity": "PASS", "off_effective_modulations": 0, "own_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            write_csv(root / "pilot/metrics" / ("REPEAT_PAIRED_METRICS_V1.csv" if repeat_only else "FIXED_36_PAIR_METRICS_V1.csv"), paired)
    finally:
        stack["original_module"].base.close_runtime(stack)
    protected_postflight(root)
    if repeat_only:
        assert all(r["repeat_latent_bitexact"] and r["repeat_pixel_bitexact"] for r in repeat_qa), "New-process exact replay failed"
        write_csv(root / "pilot/qa/NEW_PROCESS_REPEAT_QA_V1.csv", repeat_qa)
        atomic_json(root / "pilot/qa/NEW_PROCESS_REPEAT_COMPLETED_V1.json", {"status": "PASS", "images_replayed": len(repeat_qa), "independent_process": True, "completed_utc": now()})
    else:
        qa = {"status": "ENGINEERING_FUNCTIONAL_ABLATION_VERIFIED", "nodes": len(selected), "pairs": len(paired),
              "all_condition_initial_prompt_bitexact": True, "historical_on_exact_pixels": sum(r["historical_on_pixel_bitexact"] for r in paired),
              "historical_on_max_abs_pixel_error": max(r["historical_on_pixel_max_abs"] for r in paired),
              "historical_on_max_pixel_rmse": max(r["historical_on_pixel_rmse"] for r in paired),
              "on_off_zero_response_images": sum(r["zero_pixel_response"] for r in paired), "off_effective_modulations": 0,
              "actual_unet_calls": len(stack["timesteps"]), "medical_review": "NOT_RUN", "test_access": 0, "completed_utc": now()}
        atomic_json(root / "pilot/qa/FIXED_36_PILOT_COMPLETED_V1.json", qa)
        print(json.dumps(qa), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--repeat-only", action="store_true")
    a = p.parse_args()
    try:
        pilot(a.root, a.repeat_only)
    except Exception as exc:
        atomic_json(Path(a.root) / "release/ENGINEERING_INCOMPLETE.json", {"status": "ENGINEERING_INVALID_STOPPED", "stage": "repeat" if a.repeat_only else "pilot", "error": repr(exc), "updated_utc": now(), "test_access": 0})
        raise
