from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np


def validate_binary(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Invalid patient probability vectors")
    if set(y.tolist()) != {0, 1} or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels/probabilities")
    return y, p


def independent_metrics(y, p, threshold):
    """Tie-aware rank ROC and grouped-score AP independent of sklearn."""
    y, p = validate_binary(y, p)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=float)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and p[order[end]] == p[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(y) - 1]
    tps = np.cumsum(ys)[ends]
    ap = np.sum(np.diff(np.r_[0, tps]) / n1 * tps / (ends + 1))
    yes = p >= float(threshold)
    tp, tn = int((yes & (y == 1)).sum()), int((~yes & (y == 0)).sum())
    fp, fn = n0 - tn, n1 - tp
    return dict(ROC_AUC=float(auc), PR_AUC=float(ap), Accuracy=(tp+tn)/len(y),
                Sensitivity=tp/n1, Specificity=tn/n0, Precision=tp/(tp+fp) if tp+fp else 0.,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., Val_N=len(y), HT0=n0, HT1=n1,
                TN=tn, FP=fp, FN=fn, TP=tp, threshold=float(threshold))


def load_native_metric(path, name, guard):
    """Execute one trusted historical function, not its training main module."""
    import pandas as pd
    source = guard.track(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    fragment = ast.get_source_segment(source, node)
    scope = dict(np=np, Path=Path, Any=Any,
                 read_csv=lambda q: pd.read_csv(q, dtype={"patient_id": str},
                                               float_precision="round_trip").to_dict("records"))
    exec(compile(fragment, "historical_native_metric:" + name, "exec"), scope)
    return scope[name], fragment


def historical_ensemble_metrics(branch, frame, private_csv, guard):
    from sklearn.metrics import average_precision_score, roc_auc_score
    function, source = load_native_metric(branch["eval_script"], branch["metric_function"], guard)
    y, p = validate_binary(frame.label.to_numpy(int), frame.prob_ht1.to_numpy(float))
    if branch["metric_call"] == "csv":
        native = function(Path(private_csv))
    elif branch["metric_call"] == "arrays":
        native = function(y, p)
    else:
        raise ValueError("Unknown original metric function signature")
    result = independent_metrics(y, p, float(native["threshold"]))
    checks = dict(ROC_AUC=float(roc_auc_score(y, p)), PR_AUC=float(average_precision_score(y, p)),
                  **{key: float(native[key]) for key in ("Accuracy", "Sensitivity", "Specificity", "F1")})
    errors = {key: abs(checks[key] - result[key]) for key in checks}
    if max(errors.values()) >= 1e-12:
        raise RuntimeError("Independent metric arithmetic disagrees with original function")
    return result, errors, source


def historical_seed_metrics(y, p):
    from sklearn.metrics import roc_curve
    y, p = validate_binary(y, p)
    fpr, tpr, thresholds = roc_curve(y, p)
    threshold = float(thresholds[int(np.nanargmax(tpr-fpr))])
    threshold = min(max(threshold, 0.), 1.) if math.isfinite(threshold) else .5
    return independent_metrics(y, p, threshold)
