#!/usr/bin/env python3
"""Train-only self-supervised Progression Condition for frozen B16v1.

The official prediction target is never read.  Each sample is built only from
successive REAL exams inside the official Train history: a prefix predicts the
next historical exam's frozen B16 condition-space representation.  Patient-level
inner train/dev splitting prevents repeated-exam leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from b16v1_progression.progression_condition import ProgressionCondition

OUT: Path
G1: Path
COND_CKPT: Path
SEED = 20260921
EPOCHS = 120
PATIENCE = 18
BATCH_SIZE = 128


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if fields:
            w.writeheader(); w.writerows(rows)



def parse_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def load_records() -> list[dict[str, Any]]:
    records = []
    with G1.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            history = sorted(row["history"], key=lambda x: str(x["exam_date"])[:10])
            if any(str(x.get("source_type")) != "REAL" for x in history):
                raise RuntimeError(f"non-REAL history: {row['patient_id']}")
            if any(int(str(x["exam_date"])[:4]) >= int(row["target_year"]) for x in history):
                raise RuntimeError(f"official target-year history read: {row['patient_id']}")
            records.append({"patient_id": str(row["patient_id"]), "target_year": int(row["target_year"]), "history": history})
    return records


def build_frozen_model():
    from b16v1_progression.longitudinal_condition import LongitudinalConditionModel
    model = LongitudinalConditionModel().cpu().eval()
    raw = torch.load(COND_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(raw["model"] if isinstance(raw, dict) and "model" in raw else raw, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def weighted_mean(items: list[tuple[torch.Tensor, int]], empty: torch.Tensor) -> torch.Tensor:
    if not items:
        return empty
    total = sum(max(1, n) for _, n in items)
    return sum(x * max(1, n) for x, n in items) / float(total)


def extract_samples(records: list[dict[str, Any]], frozen) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = sorted(r["patient_id"] for r in records if len(r["history"]) >= 2)
    rng = random.Random(SEED); rng.shuffle(eligible)
    n_dev = max(1, round(0.20 * len(eligible)))
    dev_ids = set(eligible[:n_dev]); train_ids = set(eligible[n_dev:])
    if train_ids & dev_ids:
        raise RuntimeError("patient overlap")

    samples: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    empty_before, _ = frozen.before_encoder([], [])
    with torch.no_grad():
        for pidx, record in enumerate(records, 1):
            history = record["history"]
            if len(history) < 2:
                continue
            before_cache=[]; after_cache=[]
            for exam in history:
                paths=[str(x) for x in exam.get("image_paths", [])]
                if not paths or any(not Path(x).is_file() for x in paths):
                    raise RuntimeError(f"missing REAL images: {record['patient_id']} {exam['exam_date']}")
                b,_=frozen.before_encoder(paths, []); a,_=frozen.after_encoder(paths, [])
                before_cache.append((b.detach().float(),len(paths))); after_cache.append(a.detach().float())
            for target_idx in range(1, len(history)):
                prefix=history[:target_idx]; target=history[target_idx]
                before=weighted_mean(before_cache[:max(0,target_idx-1)], empty_before)
                after=after_cache[target_idx-1]
                task={
                    "query_date":str(target["exam_date"])[:10],
                    "anchor_date":str(history[0]["exam_date"])[:10],
                    "before_exam_date":str(prefix[-2]["exam_date"] if len(prefix)>1 else prefix[-1]["exam_date"])[:10],
                    "after_exam_date":str(prefix[-1]["exam_date"])[:10],
                    "annual_index":target_idx,
                }
                temporal,_=frozen.query_encoder(task)
                old=frozen.bridge(before[None],after[None],temporal[None])["condition_vector"][0]
                target_after=after_cache[target_idx]
                teacher=frozen.bridge(before[None],target_after[None],temporal[None])["condition_vector"][0]
                feats=after_cache[:target_idx]
                deltas=torch.stack([feats[k]-feats[k-1] for k in range(1,len(feats))]) if len(feats)>1 else torch.zeros((1,16))
                split="inner_dev" if record["patient_id"] in dev_ids else "inner_train"
                sample={"before":before,"after":after,"temporal":temporal.detach().float(),"deltas":deltas.float(),"target_delta":(target_after-after).float(),"old":old.detach().float(),"teacher":teacher.detach().float(),"split":split}
                samples.append(sample)
                manifest.append({
                    "patient_id":record["patient_id"],"inner_role":split,"prefix_exam_count":target_idx,
                    "source_exam_date":str(prefix[-1]["exam_date"])[:10],"self_supervision_exam_date":str(target["exam_date"])[:10],
                    "clinical_target_year":record["target_year"],"self_supervision_precedes_clinical_target":int(int(str(target["exam_date"])[:4])<record["target_year"]),
                    "target_label_used":0,"official_val_used":0,"official_test_used":0,
                })
            if pidx % 100 == 0:
                print(json.dumps({"stage":"feature_extract","patients":pidx,"samples":len(samples)}),flush=True)
    if not samples or any(x["self_supervision_precedes_clinical_target"] != 1 for x in manifest):
        raise RuntimeError("self-supervision leakage audit failed")
    write_csv(OUT/"manifests/PROGRESSION_SELF_SUPERVISION_PAIRS_V1.csv",manifest)
    write_json(OUT/"audits/PATIENT_SPLIT_AUDIT_V1.json",{
        "seed":SEED,"eligible_patients":len(eligible),"inner_train_patients":len(train_ids),"inner_dev_patients":len(dev_ids),
        "patient_overlap":0,"official_val_access":0,"official_test_access":0,"official_target_exam_access":0,
    })
    return samples, manifest


def batchify(rows: list[dict[str, Any]], indices: list[int]):
    chosen=[rows[i] for i in indices]; max_len=max(x["deltas"].shape[0] for x in chosen)
    deltas=torch.zeros((len(chosen),max_len,16)); mask=torch.zeros((len(chosen),max_len))
    for i,x in enumerate(chosen):
        n=x["deltas"].shape[0]; deltas[i,:n]=x["deltas"]; mask[i,:n]=1.0 if n>1 or torch.any(x["deltas"]!=0) else 0.0
    stack=lambda k:torch.stack([x[k] for x in chosen])
    return stack("before"),stack("after"),stack("temporal"),deltas,mask,stack("target_delta"),stack("old"),stack("teacher")


def losses(model: ProgressionCondition, batch) -> tuple[torch.Tensor,dict[str,float]]:
    before,after,temporal,deltas,mask,target_delta,old,teacher=batch
    cond,progression,_=model(before,after,temporal,deltas,mask)
    teacher_mse=F.mse_loss(cond,teacher)
    teacher_cos=(1-F.cosine_similarity(cond,teacher,dim=-1).mean())
    progression_mse=F.mse_loss(progression,target_delta)
    progression_cos=(1-F.cosine_similarity(progression,target_delta,dim=-1).mean())
    compatibility=F.mse_loss(cond,old)
    total=teacher_mse+0.20*teacher_cos+0.50*progression_mse+0.10*progression_cos+0.10*compatibility
    return total,{"loss":float(total.detach()),"teacher_mse":float(teacher_mse.detach()),"teacher_cos":float(teacher_cos.detach()),"progression_mse":float(progression_mse.detach()),"progression_cos":float(progression_cos.detach()),"compatibility_mse":float(compatibility.detach())}


def evaluate(model,rows,indices):
    model.eval(); acc=[]
    with torch.no_grad():
        for start in range(0,len(indices),BATCH_SIZE):
            _,m=losses(model,batchify(rows,indices[start:start+BATCH_SIZE])); acc.append(m)
    return {k:float(np.mean([x[k] for x in acc])) for k in acc[0]}


def main() -> None:
    for rel in ("configs","manifests","audits","training","checkpoints","release","logs"):
        (OUT/rel).mkdir(parents=True,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    frozen=build_frozen_model(); rows,_=extract_samples(load_records(),frozen)
    train_idx=[i for i,x in enumerate(rows) if x["split"]=="inner_train"]
    dev_idx=[i for i,x in enumerate(rows) if x["split"]=="inner_dev"]
    model=ProgressionCondition().cpu(); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    log=[]; best=float("inf"); bad=0
    for epoch in range(1,EPOCHS+1):
        model.train(); order=train_idx.copy(); random.Random(SEED+epoch).shuffle(order); tr=[]
        for start in range(0,len(order),BATCH_SIZE):
            opt.zero_grad(set_to_none=True); loss,m=losses(model,batchify(rows,order[start:start+BATCH_SIZE])); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tr.append(m)
        dev=evaluate(model,rows,dev_idx); row={"epoch":epoch,"train_loss":float(np.mean([x["loss"] for x in tr])),**{f"dev_{k}":v for k,v in dev.items()}}; log.append(row); write_csv(OUT/"training/TRAIN_LOG_V1.csv",log)
        print(json.dumps(row),flush=True)
        if dev["loss"] < best-1e-6:
            best=dev["loss"]; bad=0
            torch.save({"model":model.state_dict(),"epoch":epoch,"dev_metrics":dev,"architecture":{"progression_dim":16,"fusion":"64->128->LayerNorm->GELU->64","cpfm_adapter":"64->16"},"test_access":0},OUT/"checkpoints/best_progression_condition_v1.pt")
        else:
            bad+=1
            if bad>=PATIENCE: break
    ckpt=torch.load(OUT/"checkpoints/best_progression_condition_v1.pt",map_location="cpu",weights_only=True); model.load_state_dict(ckpt["model"]); final=evaluate(model,rows,dev_idx)
    config={"architecture":{"before":16,"after":16,"temporal":16,"progression":16,"concat":64,"fusion":["Linear(64,128)","LayerNorm(128)","GELU","Linear(128,64)"],"frozen_cpfm_adapter":"Linear(64,16)+tanh"},"training":{"seed":SEED,"optimizer":"AdamW","lr":0.002,"batch_size":BATCH_SIZE,"max_epochs":EPOCHS,"patience":PATIENCE,"best_epoch":ckpt["epoch"]},"supervision":"next REAL historical exam within official Train only","official_val_access":0,"official_test_access":0,"official_target_exam_access":0,"synthetic_input":0}
    write_json(OUT/"configs/PROGRESSION_ENCODER_CONFIG_V1.json",config)
    write_json(OUT/"release/PROGRESSION_TRAINING_RELEASE_V1.json",{"status":"B16V1_PROGRESSION_CONDITION_TRAINING_COMPLETED","samples":len(rows),"train_samples":len(train_idx),"dev_samples":len(dev_idx),"best_epoch":ckpt["epoch"],"inner_dev_metrics":final,"checkpoint":str(OUT/"checkpoints/best_progression_condition_v1.pt"),"test_access":0,"completed_utc":now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on an authorized Train-only history JSONL")
    parser.add_argument("--train_history_jsonl", type=Path, required=True)
    parser.add_argument("--longitudinal_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("output_dir must be outside this public release")
    G1 = args.train_history_jsonl
    COND_CKPT = args.longitudinal_ckpt
    OUT = args.output_dir
    main()
