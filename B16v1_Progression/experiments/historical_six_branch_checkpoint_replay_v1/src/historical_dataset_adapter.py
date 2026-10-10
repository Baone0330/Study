from __future__ import annotations

import json
from collections import Counter


def original_entries(config, guard):
    """Lossless old flat-manifest reconstruction; no new eligibility filtering."""
    import pandas as pd
    first = config["branches"][0]
    canonical = pd.read_csv(guard.track(first["val_manifest"]), dtype={"patient_id": str},
                            float_precision="round_trip")
    cohort = pd.read_csv(guard.track(first["cohort_manifest"]), dtype={"patient_id": str},
                         float_precision="round_trip")
    if len(canonical) != 1134 or len(cohort) != 189 or cohort.patient_id.nunique() != 189:
        raise ValueError("Wrong historical Val support")
    if Counter(cohort.target_label.astype(int)) != Counter({0: 131, 1: 58}):
        raise ValueError("Historical Val labels differ")
    for branch in config["branches"]:
        other = pd.read_csv(guard.track(branch["val_manifest"]), dtype={"patient_id": str},
                            float_precision="round_trip")
        pd.testing.assert_frame_equal(canonical.reset_index(drop=True),
                                      other[canonical.columns].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        other_cohort = pd.read_csv(guard.track(branch["cohort_manifest"]), dtype={"patient_id": str},
                                   float_precision="round_trip")
        a = cohort.set_index("patient_id").target_label.sort_index()
        b = other_cohort.set_index("patient_id").target_label.sort_index()
        pd.testing.assert_series_equal(a, b, check_dtype=False)
    patients = cohort.set_index("patient_id").to_dict("index")
    entries = []
    for patient, slots in canonical.groupby("patient_id", sort=False):
        if len(slots) != 6 or not slots.slot_index.is_monotonic_increasing:
            raise ValueError("Historical six-slot order changed")
        cr = patients[patient]
        nodes = []
        for row in slots.to_dict("records"):
            if row["split"] != "val" or row["source_type"] not in ("REAL", "LOCF"):
                raise ValueError("Replay may instantiate historical Val REAL/LOCF only")
            paths = json.loads(row["image_paths_json"])
            if not paths:
                raise ValueError("Empty original examination")
            for path in paths:
                guard.track(path)
            nodes.append(dict(slot_index=int(row["slot_index"]), slot_date=row["slot_date"],
                              source_type=row["source_type"], exam_date=row["exam_date"],
                              image_paths=paths, causal_source_k=row.get("causal_source_k")))
        entries.append(dict(patient_id=patient, split="val", target_year=int(cr["target_year"]),
                            target_label=int(cr["target_label"]), history_count=int(cr["history_count"]),
                            nodes=nodes))
    if set(patients) != {x["patient_id"] for x in entries}:
        raise ValueError("Manifest/cohort identities disagree")
    return entries


def create_dataset(base, entries, guard):
    allowed = {path for row in entries for slot in row["nodes"] for path in slot["image_paths"]}

    class WhitelistedOriginalDataset(base.FixedDataset):
        def _load(self, path, target_year):
            if path not in allowed:
                raise RuntimeError("Image outside original Val manifest")
            guard.track(path)
            return super()._load(path, target_year)

    return WhitelistedOriginalDataset(entries, "val", "HISTORICAL_CHECKPOINT_REPLAY")
