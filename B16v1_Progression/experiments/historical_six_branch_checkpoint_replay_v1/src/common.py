from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

IDENTITY = "HISTORICAL_FIXED_CLIMATV2_INTERNAL_VAL_SIX_BRANCH_REPLAY"
BRANCHES = (
    "REAL_ONLY", "FULL_B16V1", "REMOVE_CPFM_PLAIN_SD", "HISTORY_ONLY",
    "TEMPORAL_ONLY", "B16V1_PLUS_ORIGINAL_PROGRESSION",
)
DISPLAY = ("REAL-only", "Full B16v1", "Remove CPFM / Plain-SD", "history_only",
           "temporal_only", "B16v1+progression")
FORBIDDEN_SOURCE_MARKERS = ("strict_pretarget", "clean_f16", "clean_original_progression",
                            "s2_direction", "s2_directionaware")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    return write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    return path


def write_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.17g")
    temporary.chmod(0o600)
    temporary.replace(path)
    return path


def metadata(path):
    path = Path(path)
    st = path.stat()
    return dict(path=str(path), resolved=str(path.resolve()), size=st.st_size,
                mtime_ns=st.st_mtime_ns, inode=st.st_ino)


class SourceGuard:
    """Metadata-only source protection; intentionally no file digests."""

    def __init__(self):
        self.records = {}

    def track(self, path):
        item = metadata(path)
        previous = self.records.get(item["path"])
        if previous is not None and previous != item:
            raise RuntimeError("Protected source metadata changed")
        self.records[item["path"]] = item
        return Path(path)

    def check(self):
        changed = [key for key, before in self.records.items() if metadata(key) != before]
        return dict(status="PASS" if not changed else "FAIL", protected_files=len(self.records),
                    changed=changed, method="path/resolved/size/mtime_ns/inode", checksum_workflow=0)

    def save(self, directory):
        directory = Path(directory)
        result = self.check()
        write_json(directory / "PROTECTED_SOURCE_METADATA_V1.json", list(self.records.values()))
        write_json(directory / "PROTECTED_SOURCE_POSTFLIGHT_V1.json", result)
        if result["status"] != "PASS":
            raise RuntimeError("Protected sources changed; publication blocked")
        return result


def load_config(path):
    import yaml
    with Path(path).open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("identity") != IDENTITY:
        raise ValueError("This tool accepts only the explicit historical replay identity")
    groups = config.get("branches", [])
    if [row.get("branch_id") for row in groups] != list(BRANCHES):
        raise ValueError("All six historical branch definitions must be explicit and ordered")
    for row in groups:
        if len(row.get("seeds", [])) != 3 or len(set(row["seeds"])) != 3:
            raise ValueError("Each historical branch needs exactly three distinct source seeds")
        for key in ("root", "val_manifest", "cohort_manifest", "eval_script", "upstream_config"):
            value = row.get(key)
            if value and any(marker in str(value).lower() for marker in FORBIDDEN_SOURCE_MARKERS):
                raise ValueError("Later clean/strict/S2 assets cannot serve as historical sources")
    if config.get("batch_size") != 4 or config.get("dtype") != "fp32":
        raise ValueError("The validated historical contract is FP32 patient batch four")
    if config.get("expected_val") != {"N": 189, "HT0": 131, "HT1": 58}:
        raise ValueError("Historical Val support must be declared, not silently replaced")
    for item in (config["model_runner"], config["model_root"]):
        if any(marker in str(item).lower() for marker in FORBIDDEN_SOURCE_MARKERS):
            raise ValueError("Wrong model source identity")
    config["config_source"] = str(Path(path).resolve())
    return config


def load_original_module(path, name, guard):
    path = guard.track(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze_replay_config(config, out):
    """Prevent a resumed output root from mixing changed sources/flags/seeds."""
    frozen = {key: value for key, value in config.items() if key != "config_source"}
    path = Path(out) / "configs_private/FROZEN_REPLAY_CONFIG_V1.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError("Replay source/config identity changed; use a new output root")
    else:
        write_json(path, frozen)


def configure_inference_environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
