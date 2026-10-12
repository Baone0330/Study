"""Frozen batch6 OFF generation with transactional per-image recovery; no filtering."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cpfm_common import atomic_json, now, protected_postflight, read_json, write_csv
from sample_original_cpfm_ablation import capture_condition, difference, equal_condition, original_runtime, sample_batch


def output_paths(root, node):
    return [root / "generation/CPFM_OFF" / f"patient_{node['patient_id']}" / f"year_{node['annual_slot_index']:03d}_GENERATED/images" / f"gen_{i:02d}.png" for i in range(6)]


def validate_png(path):
    with Image.open(path) as im:
        assert im.mode == "RGB" and im.size == (512, 512)
        im.verify()


def main(root, max_nodes=None):
    root = Path(root)
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    assert cfg["comparison_route"] == "H" and cfg["formal_target_images"] == 9270
    assert read_json(root / "pilot/qa/NEW_PROCESS_REPEAT_COMPLETED_V1.json")["status"] == "PASS"
    nodes = read_json(root / "manifests/FROZEN_NODES_V1.json")
    records = read_json(root / "manifests/FROZEN_REAL_HISTORY_RECORDS_V1.json")
    db = sqlite3.connect(root / "generation/ledger.sqlite", timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS contract (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS images (node_key TEXT, image_index INTEGER, master_seed INTEGER NOT NULL, branch TEXT NOT NULL, path TEXT NOT NULL, state TEXT NOT NULL, metrics TEXT, error_history TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(node_key,image_index))")
    db.execute("CREATE TABLE IF NOT EXISTS events (time TEXT, node_key TEXT, event TEXT, detail TEXT)")
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    contract = db.execute("SELECT payload FROM contract WHERE id=1").fetchone()
    if contract:
        assert contract[0] == payload, "Resume config/source contract differs"
    else:
        db.execute("INSERT INTO contract VALUES (1,?)", (payload,))
    for n in nodes:
        for i, (seed, path) in enumerate(zip(n["seeds"], output_paths(root, n))):
            db.execute("INSERT OR IGNORE INTO images(node_key,image_index,master_seed,branch,path,state) VALUES(?,?,?,?,?,?)", (n["node_id"], i, seed, "HIST_ORIGINAL_P16_CPFM_OFF", str(path), "PENDING"))
            saved = db.execute("SELECT master_seed,branch,path FROM images WHERE node_key=? AND image_index=?", (n["node_id"], i)).fetchone()
            assert saved == (seed, "HIST_ORIGINAL_P16_CPFM_OFF", str(path))
    db.commit()
    assert db.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 9270
    free, _ = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise RuntimeError("Shared GPU has less than 12 GiB free; do not alter batch or other processes")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    stack = original_runtime(cfg)
    newly_generated_nodes = 0
    try:
        for index, node in enumerate(nodes, 1):
            states = db.execute("SELECT image_index,state,path FROM images WHERE node_key=? ORDER BY image_index", (node["node_id"],)).fetchall()
            preserved = {Path(p) for _, state, p in states if state == "COMPLETE"}
            for path in preserved:
                validate_png(path)
            if len(preserved) == 6:
                continue
            if max_nodes is not None and newly_generated_nodes >= max_nodes:
                break
            db.execute("INSERT INTO events VALUES(?,?,?,?)", (now(), node["node_id"], "NODE_START", json.dumps({"preserved_complete": len(preserved), "batch": 6})))
            db.execute("UPDATE images SET state='RUNNING' WHERE node_key=? AND state<>'COMPLETE'", (node["node_id"],))
            db.commit()
            node_started = time.perf_counter()
            try:
                condition, _, on_condition = capture_condition(stack, records[node["patient_id"]], node)
                condition_off, _, off_condition = capture_condition(stack, records[node["patient_id"]], node)
                equal_condition(on_condition, off_condition)
                outputs = output_paths(root, node)
                trace, pixels, audit = sample_batch(stack, condition_off, node, False, outputs, detailed=False, preserve_existing=preserved)
                assert audit["effective_modulations"] == 0 and not audit["hook_leak"]
                audit["actual_residuals"] = []
                for i, output in enumerate(outputs):
                    validate_png(output)
                    with Image.open(node["reference_on_paths"][i]) as im:
                        ref = np.asarray(im.convert("RGB"))
                    pixel_diff = difference(ref, pixels[i])
                    with torch.no_grad():
                        on_f16, _ = stack["condition_model"].after_encoder([node["reference_on_paths"][i]], [])
                        off_f16, _ = stack["condition_model"].after_encoder([str(output)], [])
                    f16_diff = difference(on_f16.numpy(), off_f16.numpy())
                    measurement = {"node_key_private": node["node_id"], "patient_id_private": node["patient_id"], "image_index": i, "master_seed": node["seeds"][i],
                                   "condition16_bitexact": True, "all_condition_components_bitexact": True, "physical_hook_calls": audit["physical_prehook_calls_by_layer"],
                                   "effective_modulations": 0, "unet_calls": audit["unet_calls"], "P16_conditions_computed": True,
                                   "P16_condition_reaches_UNet_without_CPFM": False, "independent_hooks_calls": 0,
                                   **{f"on_off_pixel_{k}": v for k, v in pixel_diff.items()},
                                   **{f"on_off_generated_f16_{k}": v for k, v in f16_diff.items()}, "rgb_std": float(pixels[i].std()),
                                   "rgb_min": int(pixels[i].min()), "rgb_max": int(pixels[i].max()), "low_texture_flag_std_lt_5": bool(pixels[i].std() < 5),
                                   "finite": True, "complete": True, "pixel_qa": "RGB512_PASS", "rgb_png_path": str(output),
                                   "initial_noise_status": "EXACT_FROZEN_SEED_BATCH6; actual pair equality verified fixed36 pilot",
                                   "final_latent_status": "FINITE; matched oldON latent unavailable outside fixed36 pilot",
                                   "upstream_sources": cfg["paths"], "completed_utc": now()}
                    # A preserved result is only verified; its original completion evidence stays untouched.
                    if output not in preserved:
                        db.execute("UPDATE images SET state='COMPLETE',metrics=? WHERE node_key=? AND image_index=?", (json.dumps(measurement), node["node_id"], i))
                        db.commit()
                newly_generated_nodes += 1
                db.execute("INSERT INTO events VALUES(?,?,?,?)", (now(), node["node_id"], "NODE_COMPLETE", json.dumps({"elapsed_seconds": time.perf_counter() - node_started, "batch": 6})))
                db.commit()
                counts = dict(db.execute("SELECT state,COUNT(*) FROM images GROUP BY state").fetchall())
                complete_nodes = db.execute("SELECT COUNT(*) FROM (SELECT node_key FROM images WHERE state='COMPLETE' GROUP BY node_key HAVING COUNT(*)=6)").fetchone()[0]
                progress = {"stage": "FORMAL_OFF_GENERATION", "node_index": index, "completed_images": counts.get("COMPLETE", 0), "target_images": 9270,
                            "completed_nodes": complete_nodes, "target_nodes": 1545, "newly_generated_nodes_this_process": newly_generated_nodes,
                            "node_elapsed_seconds": time.perf_counter() - node_started, "process_elapsed_seconds": time.perf_counter() - started,
                            "own_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3, "own_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                            "status_counts": counts, "batch": 6, "updated_utc": now(), "test_access": 0}
                atomic_json(root / "release/GENERATION_PROGRESS_V1.json", progress)
                atomic_json(root / "release/EXECUTION_STATE_V1.json", {"status": "FORMAL_OFF_GENERATING", "phase": 4, **progress})
                print(json.dumps(progress), flush=True)
            except Exception as exc:
                unfinished = db.execute("SELECT image_index,error_history FROM images WHERE node_key=? AND state<>'COMPLETE'", (node["node_id"],)).fetchall()
                for i, previous in unfinished:
                    errors = json.loads(previous) + [{"time": now(), "error": repr(exc)}]
                    db.execute("UPDATE images SET state='FAILED',error_history=? WHERE node_key=? AND image_index=?", (json.dumps(errors), node["node_id"], i))
                db.execute("INSERT INTO events VALUES(?,?,?,?)", (now(), node["node_id"], "NODE_FAILED", repr(exc)))
                db.commit()
                raise
    finally:
        stack["original_module"].base.close_runtime(stack)
    complete = db.execute("SELECT COUNT(*) FROM images WHERE state='COMPLETE'").fetchone()[0]
    if complete == 9270:
        measurements = [json.loads(m) for (m,) in db.execute("SELECT metrics FROM images ORDER BY node_key,image_index").fetchall()]
        for row in measurements:
            validate_png(Path(row["rgb_png_path"]))
        write_csv(root / "generation/FORMAL_PER_IMAGE_ENGINEERING_METRICS_PRIVATE_V1.csv", [{k: v for k, v in row.items() if k != "upstream_sources"} for row in measurements])
        protected = protected_postflight(root)
        qa = {"status": "FORMAL_OFF_GENERATION_COMPLETED", "complete_nodes": 1545, "complete_images": complete, "failed_images": 0,
              "patients": 860, "all_png_rgb512_decodable": True, "all_condition_components_bitexact": True, "effective_modulations": 0,
              "zero_pixel_response_images": sum(row["on_off_pixel_bitexact"] for row in measurements),
              "low_texture_flag_images_retained": sum(row["low_texture_flag_std_lt_5"] for row in measurements),
              "pixel_rmse_image_mean": float(np.mean([r["on_off_pixel_rmse"] for r in measurements])),
              "protected_source_status": protected["status"], "medical_review": "NOT_RUN", "test_access": 0, "completed_utc": now()}
        atomic_json(root / "release/FORMAL_OFF_GENERATION_COMPLETED_V1.json", qa)
        print(json.dumps(qa), flush=True)
    db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--max-nodes", type=int, help="Engineering first-node check only; not a new queue")
    args = p.parse_args()
    try:
        main(args.root, args.max_nodes)
    except Exception as exc:
        atomic_json(Path(args.root) / "release/ENGINEERING_INCOMPLETE.json", {"status": "PARTIAL_WITH_LIMITATIONS", "stage": "formal_generation", "error": repr(exc), "test_access": 0, "updated_utc": now()})
        raise
