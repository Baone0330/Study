"""Private execution utilities; unique name avoids native CLIMATv2 common package."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stat_record(path):
    path = Path(path)
    st = path.stat()
    return {"path": str(path), "resolved_path": str(path.resolve()), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


def protected_postflight(root):
    rows = read_json(Path(root) / "audit/PROTECTED_SOURCE_METADATA_V1.json")
    changed = []
    for row in rows:
        try:
            if stat_record(row["path"]) != row:
                changed.append(row["path"])
        except OSError:
            changed.append(row["path"])
    result = {"status": "PASS" if not changed else "FAIL", "files": len(rows),
              "changed_paths": changed, "comparison": "path,resolved_path,size,mtime_ns,inode", "checked_utc": now()}
    atomic_json(Path(root) / "audit/PROTECTED_SOURCE_POSTFLIGHT_V1.json", result)
    if changed:
        raise RuntimeError("Protected source metadata changed; see private audit")
    return result


def safe_root(root):
    root = Path(root).resolve()
    if root.name != "b16v1_original_progression_cpfm_only_ablation_historical_v1":
        raise ValueError("Unexpected isolated output root")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root
