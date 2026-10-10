from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .common import (SourceGuard, configure_inference_environment, freeze_replay_config,
                     load_config, read_json, utc_now, write_json)
from .checkpoint_discovery import discover


def parser():
    p = argparse.ArgumentParser(description="Historical selected-checkpoint inference only; no training/generation")
    p.add_argument("--config", required=True, help="Server-private YAML paths and original source configuration")
    p.add_argument("--out", required=True, help="New private replay root, never an original experiment root")
    p.add_argument("--dry-run", action="store_true", help="CPU path/config/metadata validation only")
    p.add_argument("--replay", action="store_true", help="Independent process per original branch/seed")
    p.add_argument("--aggregate", action="store_true", help="Aggregate newly generated replay probabilities")
    p.add_argument("--worker-branch", help=argparse.SUPPRESS)
    p.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    configure_inference_environment()
    config, out = load_config(args.config), Path(args.out).resolve()
    if args.worker_branch:
        from .val_forward import replay_one
        return replay_one(args.config, out, args.worker_branch, args.worker_seed)
    if not (args.dry_run or args.replay or args.aggregate):
        parser().error("Choose --dry-run, --replay and/or --aggregate")
    for branch in config["branches"]:
        source = Path(branch["root"]).resolve()
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Output and historical source roots must be disjoint")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    import fcntl
    lock_path = out / "release/REPLAY_ACTIVE.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        baseline = out / "audit/PROTECTED_SOURCE_METADATA_V1.json"
        if baseline.exists():
            guard = SourceGuard()
            for item in read_json(baseline):
                guard.records[item["path"]] = item
            if guard.check()["status"] != "PASS":
                raise RuntimeError("Sources changed before dry-run/resume; do not reuse prior outputs")
        freeze_replay_config(config, out)
        index_path = out / "configs_private/HISTORICAL_RUN_INDEX_V1.json"
        if args.dry_run or not index_path.exists():
            rows, _ = discover(config, out)
        else:
            rows = read_json(index_path)
        print(f"HISTORICAL_DRY_RUN records={len(rows)} eligible={sum(r['eligibility']=='SELECTED_BEST_PROVEN' for r in rows)} GPU_not_used=True", flush=True)
        if args.replay:
            cli = Path(__file__).resolve().parents[1] / "scripts/replay_six_branches.py"
            done, failed = 0, 0
            for row in rows:
                run = out / "predictions_private" / row["branch"] / f"seed_{row['seed']}"
                marker = run / "status.json"
                if marker.exists() and read_json(marker).get("status") == "REPLAY_COMPLETED":
                    qa = read_json(run / "reload_qa.json")
                    if qa["checkpoint_metadata"]["path"] != row["checkpoint_path"]:
                        raise ValueError("Resume identity changed")
                    done += 1
                    continue
                log = out / "logs" / f"{row['branch']}_seed_{row['seed']}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-B", str(cli), "--config", args.config, "--out", str(out),
                           "--worker-branch", row["branch"], "--worker-seed", str(row["seed"])]
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                with log.open("a", encoding="utf-8") as handle:
                    log.chmod(0o600)
                    rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment).returncode
                done += rc == 0
                failed += rc != 0
                if rc == 0:
                    write_json(run / "stdout_source.json", dict(log=str(log)))
                progress = dict(stage="CHECKPOINT_REPLAY", completed=int(done), failed=int(failed),
                                expected=len(rows), branch=row["branch"], seed=row["seed"], utc=utc_now())
                write_json(out / "release/REPLAY_PROGRESS_V1.json", progress)
                print(f"REPLAY_PROGRESS completed={done} failed={failed} expected={len(rows)} branch={row['branch']} seed={row['seed']}", flush=True)
                if rc and marker.exists() and read_json(marker).get("category") == "CUDA_DEVICE_FAILURE":
                    print("CUDA_DEVICE_FAILURE: further GPU attempts stopped; no automatic restart", flush=True)
                    break
        if args.aggregate:
            from .aggregate import aggregate
            aggregate(args.config, out)
    return 0
