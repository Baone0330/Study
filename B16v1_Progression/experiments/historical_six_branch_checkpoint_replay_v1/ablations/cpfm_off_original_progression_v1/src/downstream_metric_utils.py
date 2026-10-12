"""Patient-level metrics; historical ensemble Youden uses smallest-threshold ties."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(y.tolist()) != {0, 1}:
        raise ValueError("Two-class aligned patient vectors required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    return y, p


def binary_metrics(y, p, threshold=None):
    y, p = validate(y, p)
    if threshold is None:
        best = None
        for candidate in sorted(set(float(value) for value in p)):
            pred = p >= candidate
            sensitivity = float(pred[y == 1].mean())
            specificity = float((~pred[y == 0]).mean())
            key = (sensitivity + specificity - 1, -candidate, candidate)
            if best is None or key > best:
                best = key
        threshold = best[2]
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {"ROC_AUC": float(roc_auc_score(y, p)), "AP": float(average_precision_score(y, p)),
            "Accuracy": (tp + tn) / len(y), "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "threshold": float(threshold), "Val_N": len(y),
            "HT0": int((y == 0).sum()), "HT1": int((y == 1).sum())}


def paired_patient_bootstrap(y, on, off, n_resamples=2000, seed=20261012):
    y, on = validate(y, on)
    _, off = validate(y, off)
    rng = np.random.default_rng(seed)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    samples = {"ROC_AUC": [], "AP": []}
    for _ in range(n_resamples):
        index = np.concatenate((rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)))
        samples["ROC_AUC"].append(float(roc_auc_score(y[index], on[index]) - roc_auc_score(y[index], off[index])))
        samples["AP"].append(float(average_precision_score(y[index], on[index]) - average_precision_score(y[index], off[index])))
    functions = {"ROC_AUC": roc_auc_score, "AP": average_precision_score}
    rows = []
    for metric, values in samples.items():
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append({"metric": metric, "comparison": "ON_minus_OFF", "point_difference": float(functions[metric](y, on) - functions[metric](y, off)),
                     "ci_lower_95": float(lower), "ci_upper_95": float(upper), "ci_includes_zero": bool(lower <= 0 <= upper),
                     "patient_n": len(y), "HT0": len(negative), "HT1": len(positive), "n_resamples": n_resamples,
                     "bootstrap_seed": seed, "unit": "patient; three seeds averaged before resampling",
                     "interval": "two-sided stratified paired percentile 95% CI",
                     "independent_clinical_interval": False})
    return rows
