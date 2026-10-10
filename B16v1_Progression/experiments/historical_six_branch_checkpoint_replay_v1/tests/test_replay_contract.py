import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import BRANCHES, SourceGuard, freeze_replay_config, load_config
from src.original_metrics import independent_metrics
from src.privacy_guard import scan_package
from src.seed_ensemble import align_and_average


class MetricsTest(unittest.TestCase):
    def test_ties_and_constant_scores(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        for p in (np.array([.1, .4, .4, .4, .9, .1]), np.full(6, .5)):
            result = independent_metrics(y, p, .4)
            self.assertAlmostEqual(result["ROC_AUC"], roc_auc_score(y, p), places=12)
            self.assertAlmostEqual(result["PR_AUC"], average_precision_score(y, p), places=12)

    def test_many_anonymous_toy_vectors(self):
        rng = np.random.default_rng(103)
        for _ in range(50):
            y = np.r_[np.zeros(31, dtype=int), np.ones(17, dtype=int)]
            p = rng.integers(0, 11, size=48) / 10
            m = independent_metrics(y, p, .5)
            self.assertAlmostEqual(m["ROC_AUC"], roc_auc_score(y, p), places=12)
            self.assertAlmostEqual(m["PR_AUC"], average_precision_score(y, p), places=12)


class AlignmentTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"patient_id": ["toy-a", "toy-b"], "label": [0, 1], "prob_ht1": [.2, .8]})

    def test_key_alignment_not_positions(self):
        frames = {1: self.frame(), 2: self.frame().iloc[::-1], 3: self.frame()}
        ids, y, p = align_and_average(frames, [1, 2, 3])
        self.assertEqual(ids, ["toy-a", "toy-b"])
        np.testing.assert_allclose(p, [.2, .8])

    def test_missing_duplicate_or_wrong_label_fails(self):
        with self.assertRaises(ValueError):
            align_and_average({1: self.frame(), 2: self.frame()}, [1, 2, 3])
        frames = {1: self.frame(), 2: self.frame(), 3: self.frame()}
        frames[2].loc[1, "label"] = 0
        with self.assertRaises(ValueError):
            align_and_average(frames, [1, 2, 3])
        frames[2] = self.frame()
        frames[2].loc[1, "patient_id"] = "toy-a"
        with self.assertRaises(ValueError):
            align_and_average(frames, [1, 2, 3])


class ProtectionTest(unittest.TestCase):
    def test_metadata_mutation_is_detected_without_digest(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("REPLAY_TEST_TMPDIR")) as temporary:
            path = Path(temporary) / "toy.txt"
            path.write_text("anonymous")
            guard = SourceGuard()
            guard.track(path)
            self.assertEqual(guard.check()["status"], "PASS")
            path.write_text("anonymous changed")
            self.assertEqual(guard.check()["status"], "FAIL")

    def test_weight_extension_and_private_json_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("REPLAY_TEST_TMPDIR")) as temporary:
            root = Path(temporary)
            (root / "toy.pt").write_bytes(b"anonymous fixture")
            (root / "toy.json").write_text(json.dumps({"patient_id": "toy-a"}))
            result = scan_package(root)
            self.assertEqual(result["status"], "FAIL")

    def test_safe_six_row_aggregate(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("REPLAY_TEST_TMPDIR")) as temporary:
            path = Path(temporary) / "aggregate.csv"
            pd.DataFrame({"branch_id": BRANCHES, "ROC_AUC": [.5] * 6}).to_csv(path, index=False)
            self.assertEqual(scan_package(temporary)["status"], "PASS")


class ConfigTest(unittest.TestCase):
    def test_resume_config_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("REPLAY_TEST_TMPDIR")) as temporary:
            config = {"identity": "anonymous-toy", "seeds": [1, 2, 3]}
            freeze_replay_config(config, temporary)
            freeze_replay_config(dict(config), temporary)
            with self.assertRaises(ValueError):
                freeze_replay_config({**config, "seeds": [1, 2, 4]}, temporary)

    def test_public_template_cannot_silently_run_as_private_config(self):
        template = Path(__file__).resolve().parents[1] / "configs/example_local_paths.yaml"
        with self.assertRaises((ValueError, KeyError)):
            load_config(template)


if __name__ == "__main__":
    unittest.main()
