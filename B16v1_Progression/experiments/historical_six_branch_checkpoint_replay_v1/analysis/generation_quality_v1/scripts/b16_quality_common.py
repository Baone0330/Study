"""Isolated quality-study IO; never writes source images or clinical reports."""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def now():
    return datetime.now(timezone.utc).isoformat()


def setup(root):
    root = Path(root).resolve()
    assert root.name == "b16v1_historical_original_progression_generation_quality_v1"
    for name in ("protocol", "source_audit", "manifests", "metrics", "image_features", "figures/nonmedical", "figures/private_medical_panels",
                 "figures/source_data", "figures/captions", "figures/export_qa", "writing", "scripts", "tests", "logs", "release", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache/matplotlib"))
    sys.path.insert(0, str(root / "vendor"))
    return root


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(tmp, index=False, encoding="utf-8")
    else:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    os.replace(tmp, path)


def stat_record(path):
    path = Path(path)
    st = path.stat()
    return {"path": str(path), "resolved_path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


def protect(root, paths):
    dest = root / "source_audit/PROTECTED_SOURCE_METADATA_V1.json"
    if dest.exists():
        raise RuntimeError("Protection baseline already exists; do not reset it")
    atomic_json(dest, [stat_record(p) for p in sorted(set(map(str, paths))) if Path(p).is_file()])


def postflight(root):
    changed = []
    records = read_json(root / "source_audit/PROTECTED_SOURCE_METADATA_V1.json")
    auxiliary = root / 'source_audit/AUXILIARY_PROTECTED_SOURCE_METADATA_V1.json'
    if auxiliary.exists(): records += read_json(auxiliary)
    encoder = root / 'image_features/encoder_contract.json'
    if encoder.exists():
        c = read_json(encoder)
        records += [c[k] for k in ['inception_weight_metadata','alexnet_weight_metadata','lpips_calibration_metadata']]
    for r in records:
        try:
            if stat_record(r["path"]) != r: changed.append(r["path"])
        except OSError:
            changed.append(r["path"])
    result = {"status": "FAIL" if changed else "PASS", "files": len(records), "changed_paths": changed,
              "comparison": "path,resolved_path,size,mtime_ns,inode", "checked_utc": now()}
    atomic_json(root / "source_audit/PROTECTED_SOURCE_POSTFLIGHT_V1.json", result)
    if changed: raise RuntimeError("Referenced original asset changed; see private postflight")
    return result


def state(root, phase, **extra):
    atomic_json(root / "release/EXECUTION_STATE_V1.json", {"phase": phase, "updated_utc": now(), **extra})
