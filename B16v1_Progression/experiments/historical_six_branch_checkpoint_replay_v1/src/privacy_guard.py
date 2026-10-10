from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

from .common import BRANCHES, SourceGuard, read_json

PACKAGE = "B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1"
ALLOWED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".csv", ".txt"}
PRIVATE_PATH = re.compile(r"(?:/(?:data|home)/[\w.-]+|[A-Za-z]:[\\/]Users[\\/][^\s]+)")
SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")
IDENTIFIER = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
PRIVATE_DATA_KEYS = {"patient_id", "patient_ids", "image_paths", "image_paths_json", "y_true", "labels",
                     "prob_ht1", "prob_positive", "checkpoint_path", "source_run_dir", "target_label"}


def scan_package(directory):
    root = Path(directory).resolve()
    problems, inventory = [], []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            problems.append(dict(file=path.relative_to(root).as_posix(), reason="Symlink not allowed"))
            continue
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        if path.is_symlink() or (path.suffix.lower() not in ALLOWED_SUFFIXES and path.name != ".gitignore"):
            problems.append(dict(file=name, reason="File type/symlink not allowed"))
            continue
        size = path.stat().st_size
        if size > 2_000_000:
            problems.append(dict(file=name, reason="Release file exceeds two MB"))
            continue
        text = path.read_text(encoding="utf-8-sig")
        for rule, reason in ((PRIVATE_PATH, "Private absolute path"), (SECRET, "Credential-like value"),
                             (IDENTIFIER, "Personal identification-like number")):
            if rule.search(text):
                problems.append(dict(file=name, reason=reason))
        if path.suffix == ".csv":
            records = list(csv.DictReader(text.splitlines()))
            if len(records) != 6 or {r.get("branch_id") for r in records} != set(BRANCHES):
                problems.append(dict(file=name, reason="Public CSV must contain exactly six aggregate branch rows"))
            if records and set(records[0]) & PRIVATE_DATA_KEYS:
                problems.append(dict(file=name, reason="Patient/checkpoint-level CSV columns"))
        if path.suffix == ".json":
            obj = json.loads(text)
            def visit(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in PRIVATE_DATA_KEYS:
                            problems.append(dict(file=name, reason="Private data key in public JSON: " + key))
                        visit(item)
                elif isinstance(value, list):
                    for item in value:
                        visit(item)
            visit(obj)
        inventory.append(dict(relative_path=name, size=size))
    return dict(status="PASS" if inventory and not problems else "FAIL", files=inventory, problems=problems,
                checksum_workflow=0, raw_weights_allowed=False, clinical_images_allowed=False,
                patient_tables_allowed=False, logs_allowed=False)


def validate_release(repo, work, allow_partial=False, require_staged=False):
    repo, work = Path(repo).resolve(), Path(work).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    origin, branch = git("remote", "get-url", "origin"), git("branch", "--show-current")
    if origin not in ("git@github-baone:Baone0330/Study.git", "https://github.com/Baone0330/Study.git") or branch != "main":
        raise ValueError("Wrong repository or branch; publication blocked")
    result = scan_package(repo / PACKAGE)
    staged = git("diff", "--cached", "--name-only").splitlines()
    if (require_staged and not staged) or any(not name.startswith(PACKAGE + "/") for name in staged):
        raise ValueError("Stage scope is not exclusively the new replay package")
    if git("diff", "--name-only"):
        raise ValueError("Unstaged tracked changes remain; reviewed candidate is not the staged version")
    for name in staged:
        staged_bytes = subprocess.check_output(["git", "-C", str(repo), "show", ":" + name])
        if staged_bytes != (repo / name).read_bytes():
            raise ValueError("Staged bytes differ from privacy-reviewed working file")
    guard = SourceGuard()
    for item in read_json(work / "audit/PROTECTED_SOURCE_METADATA_V1.json"):
        guard.records[item["path"]] = item
    if guard.check()["status"] != "PASS":
        raise ValueError("Source metadata changed since replay; publication blocked")
    aggregate = json.loads((work / "release/REPLAY_AGGREGATE_STATUS_V1.json").read_text())
    if not allow_partial and (aggregate["completed_checkpoint_forwards"] != 18 or aggregate["complete_ensembles"] != 6):
        raise ValueError("Full replay coverage not actually completed")
    if aggregate["source_postflight"]["status"] != "PASS":
        raise ValueError("Protected source postflight failed")
    if any(aggregate[key] for key in ("new_training_steps", "new_generation_images", "optimizer_steps", "backward_calls", "checksum_workflow")):
        raise ValueError("Prohibited replay activity")
    from .common import write_json
    result.update(repository="Baone0330/Study", branch=branch, package=PACKAGE, staged_files=staged,
                  completed_checkpoint_forwards=aggregate["completed_checkpoint_forwards"],
                  complete_ensembles=aggregate["complete_ensembles"], actual_source_postflight="PASS")
    write_json(work / "release/GITHUB_RELEASE_PRIVACY_QA_V1.json", result)
    if result["status"] != "PASS":
        raise ValueError("Public file privacy guard failed; inspect private QA report")
    print(f"RELEASE_VALIDATED files={len(result['files'])} replay={aggregate['completed_checkpoint_forwards']}/18 scope={PACKAGE}")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate six-branch replay publication without private data")
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--require-staged", action="store_true")
    args = p.parse_args(argv)
    validate_release(args.repo_dir, args.work_dir, args.allow_partial, args.require_staged)
    return 0
