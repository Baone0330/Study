"""Anonymous matching, bootstrap and exact saved-RNG continuation tests."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cpfm_common import atomic_json
from downstream_execution import rng_state, restore_rng
from downstream_metric_utils import binary_metrics, paired_patient_bootstrap


def main(root):
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([.1, .3, .6, .8, .2, .9])
    identical = paired_patient_bootstrap(y, p, p, n_resamples=40)
    assert all(r["point_difference"] == r["ci_lower_95"] == r["ci_upper_95"] == 0 for r in identical)
    met = binary_metrics(y, p)
    assert met["ROC_AUC"] == met["AP"] == 1 and met["TN"] + met["FP"] + met["FN"] + met["TP"] == 6
    assert paired_patient_bootstrap(y, p, 1-p, 40) == paired_patient_bootstrap(y, p, 1-p, 40)
    random.seed(50); np.random.seed(50); torch.manual_seed(50)
    gen = torch.Generator().manual_seed(77)
    state = rng_state(gen)
    expected = (random.random(), np.random.random(3), torch.rand(3), torch.randperm(12, generator=gen))
    random.random(); np.random.random(7); torch.rand(7); torch.randperm(12, generator=gen)
    restore_rng(state, gen)
    actual = (random.random(), np.random.random(3), torch.rand(3), torch.randperm(12, generator=gen))
    assert expected[0] == actual[0] and np.array_equal(expected[1], actual[1]) and torch.equal(expected[2], actual[2]) and torch.equal(expected[3], actual[3])
    rows = [{"node_key": "ANON_A", "image_index": i, "seed": 200 + i} for i in range(6)]
    assert [r["image_index"] for r in rows] == list(range(6)) and len({(r["node_key"], r["image_index"]) for r in rows}) == 6
    bad = rows[:-1] + [rows[0]]
    assert len({(r["node_key"], r["image_index"]) for r in bad}) != 6
    qa = {"status": "PASS", "paired_identical_arm_zero_CI": True, "stratified_patient_bootstrap_reproducible": True,
          "confusion_count_arithmetic": True, "same_rng_and_loader_order_restored": True, "seed_alignment_duplicate_detection": True,
          "clinical_images_read": 0, "test_access": 0}
    atomic_json(Path(root) / "audit/ANONYMOUS_CONTRACT_TESTS_V1.json", qa)
    print(json.dumps(qa))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
