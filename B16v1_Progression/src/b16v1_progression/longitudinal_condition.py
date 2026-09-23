"""Lightweight, trainable longitudinal condition modules for AATUG v15.4.

The recovered SD1.5 generator remains frozen.  This module only encodes the
available before/after exam sets and the query-time information, then exposes
a small condition bridge that the original generator wrapper can consume.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn


ATTR_KEYS = (
    "ht_positive",
    "diffuse_change",
    "hypoechoic",
    "heterogeneous_echo",
    "reticular_change",
    "coarse_texture",
    "gland_enlargement",
    "unclear_boundary",
)
DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def _date_value(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
        if isinstance(decoded, list):
            return [str(item) for item in decoded if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [text]


def _path_date(path: str) -> str:
    match = DATE_RE.search(str(path))
    return match.group(1) if match else ""


def split_exam_sets(task: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return unordered before/after image sets without using target images."""

    before = _paths(task.get("before_exam_set")) or _paths(task.get("before_image_paths_json"))
    after = _paths(task.get("after_exam_set")) or _paths(task.get("after_image_paths_json"))
    history = _paths(task.get("history_image_paths"))
    before_date = str(task.get("before_exam_date", ""))[:10]
    after_date = str(task.get("after_exam_date", ""))[:10]
    if not before and not after and history:
        for path in history:
            pdate = _path_date(path)
            if before_date and pdate == before_date:
                before.append(path)
            elif after_date and pdate == after_date:
                after.append(path)
            elif not before:
                before.append(path)
            else:
                after.append(path)
    return before, after


def split_exam_reports(task: dict[str, Any]) -> tuple[list[str], list[str]]:
    reports = task.get("history_reports", [])
    if isinstance(reports, str):
        try:
            reports = json.loads(reports)
        except json.JSONDecodeError:
            reports = [reports] if reports else []
    reports = [str(item or "") for item in (reports or [])]
    before = task.get("before_report", "")
    after = task.get("after_report", "")
    if before or after:
        return ([str(before)] if before else []), ([str(after)] if after else [])
    if len(reports) >= 2:
        return [reports[0]], [reports[1]]
    return reports, []


def _report_features(text: str) -> np.ndarray:
    text = str(text or "")
    # Small language-agnostic/Chinese-aware counts; the target report is never
    # passed to this function by the inference entry point.
    terms = (
        ("heterogeneous", "不均匀", "弥漫"),
        ("nodule", "结节", "低回声"),
        ("enlarged", "增大", "饱满"),
        ("boundary", "边界", "欠清"),
        ("flow", "血流", "CDFI"),
        ("hashimoto", "桥本", "甲状腺炎"),
    )
    values = [float(sum(text.lower().count(term.lower()) for term in group)) for group in terms]
    return np.asarray([min(v, 8.0) / 8.0 for v in values], dtype=np.float32)


def image_feature(path: str | Path, size: int = 32) -> np.ndarray:
    """Return stable grayscale statistics used by the tiny trainable encoder."""

    with Image.open(path) as source:
        image = source.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    gx = np.diff(array, axis=1).mean() if array.shape[1] > 1 else 0.0
    gy = np.diff(array, axis=0).mean() if array.shape[0] > 1 else 0.0
    q10, q50, q90 = np.quantile(array, [0.10, 0.50, 0.90])
    return np.asarray(
        [
            float(array.mean()),
            float(array.std()),
            float(q10),
            float(q50),
            float(q90),
            float(np.mean(np.abs(gx))),
            float(np.mean(np.abs(gy))),
            float(array.max() - array.min()),
        ],
        dtype=np.float32,
    )


def target_proxy_features(path: str | Path) -> torch.Tensor:
    """Supervision proxy for the pilot, not an inference condition."""

    feat = image_feature(path)
    return torch.from_numpy(feat[:4].copy())


class ExamSetEncoder(nn.Module):
    """Permutation-invariant variable-cardinality exam-set encoder."""

    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.Linear(14, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, self.output_dim),
        )

    def forward(self, image_paths: Iterable[str], reports: Iterable[str]) -> tuple[torch.Tensor, dict[str, Any]]:
        paths = [str(path) for path in image_paths if str(path).strip()]
        report_list = [str(report or "") for report in reports]
        rows: list[np.ndarray] = []
        for index, path in enumerate(paths):
            try:
                image_row = image_feature(path)
            except Exception:
                image_row = np.zeros(8, dtype=np.float32)
            report_row = _report_features(report_list[min(index, len(report_list) - 1)]) if report_list else np.zeros(6, dtype=np.float32)
            rows.append(np.concatenate([image_row, report_row], axis=0))
        if not rows:
            rows = [np.zeros(14, dtype=np.float32)]
        raw = torch.from_numpy(np.stack(rows, axis=0)).to(next(self.parameters()).device)
        embedding = self.projection(raw).mean(dim=0)
        # Mean pooling is deliberately order invariant; this is audited in the
        # inference script by encoding a reversed set and comparing the result.
        return embedding, {
            "image_count": len(paths),
            "report_count": len(report_list),
            "raw_feature_mean": raw.detach().mean(dim=0).cpu().tolist(),
        }


class LongitudinalQueryEncoder(nn.Module):
    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(nn.Linear(9, 24), nn.LayerNorm(24), nn.GELU(), nn.Linear(24, self.output_dim))

    def _numeric(self, task: dict[str, Any]) -> torch.Tensor:
        query = _date_value(task.get("query_date"))
        anchor = _date_value(task.get("anchor_date"))
        before = _date_value(task.get("before_exam_date"))
        after = _date_value(task.get("after_exam_date"))
        annual = float(task.get("annual_index") or task.get("relative_year") or 0.0)
        rel_days = float((query - anchor).days) if query and anchor else annual * 365.25
        before_gap = float((query - before).days) if query and before else 0.0
        after_gap = float((after - query).days) if query and after else 0.0
        month = float(query.month) if query else 1.0
        values = [
            annual / 5.0,
            rel_days / 365.25,
            before_gap / 730.5,
            after_gap / 730.5,
            float(bool(before)),
            float(bool(after)),
            np.sin(2.0 * np.pi * month / 12.0),
            np.cos(2.0 * np.pi * month / 12.0),
            min(max(len(str(task.get("query_date", ""))), 0), 10) / 10.0,
        ]
        return torch.tensor(values, dtype=torch.float32, device=next(self.parameters()).device)

    def forward(self, task: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        numeric = self._numeric(task)
        return self.projection(numeric), {"query_numeric": numeric.detach().cpu().tolist()}


class ConditionBridge(nn.Module):
    """Map the three representations into small, safe original-stack controls."""

    def __init__(self, representation_dim: int = 16):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(representation_dim * 3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.condition_projection = nn.Linear(32, 16)
        self.scale_projection = nn.Linear(32, 1)
        self.attribute_projection = nn.Linear(32, len(ATTR_KEYS))
        self.proxy_projection = nn.Linear(32, 4)
        # Keep the first enabled branch observably nonzero while bounded.
        nn.init.constant_(self.scale_projection.bias, 0.20)

    def forward(self, before: torch.Tensor, after: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self.fusion(torch.cat([before, after, query], dim=-1))
        condition_vector = torch.tanh(self.condition_projection(fused))
        scale_delta = 0.035 * torch.tanh(self.scale_projection(fused)).squeeze(-1)
        attribute_delta = 0.025 * torch.tanh(self.attribute_projection(fused))
        proxy_prediction = torch.sigmoid(self.proxy_projection(fused))
        return {
            "fused": fused,
            "condition_vector": condition_vector,
            "scale_delta": scale_delta,
            "attribute_delta": attribute_delta,
            "proxy_prediction": proxy_prediction,
        }


class LongitudinalConditionModel(nn.Module):
    def __init__(self, representation_dim: int = 16):
        super().__init__()
        self.before_encoder = ExamSetEncoder(representation_dim)
        self.after_encoder = ExamSetEncoder(representation_dim)
        self.query_encoder = LongitudinalQueryEncoder(representation_dim)
        self.bridge = ConditionBridge(representation_dim)

    def encode_task(self, task: dict[str, Any], enabled: bool = True) -> dict[str, Any]:
        before_paths, after_paths = split_exam_sets(task)
        before_reports, after_reports = split_exam_reports(task)
        before_repr, before_meta = self.before_encoder(before_paths, before_reports)
        after_repr, after_meta = self.after_encoder(after_paths, after_reports)
        query_repr, query_meta = self.query_encoder(task)
        bridge = self.bridge(before_repr[None], after_repr[None], query_repr[None])
        attr_delta = bridge["attribute_delta"][0]
        condition_vector = bridge["condition_vector"][0]
        scale_delta = float(bridge["scale_delta"][0].detach().cpu())
        condition_scale = float(np.clip(0.55 + scale_delta, 0.50, 0.60))
        attrs = {key: float(attr_delta[index].detach().cpu()) for index, key in enumerate(ATTR_KEYS)}
        return {
            "enabled": bool(enabled),
            "before_repr": before_repr.detach(),
            "after_repr": after_repr.detach(),
            "query_repr": query_repr.detach(),
            "condition_vector": condition_vector.detach(),
            "condition_scale": condition_scale if enabled else 0.55,
            "pathology_attribute_delta": attrs if enabled else {key: 0.0 for key in ATTR_KEYS},
            "bridge_mode": "v15_4_exam_set_query_condition_bridge" if enabled else "static_baseline_compatible",
            "condition_fingerprint": "not_generated",
            "before_exam_count_used": int(before_meta["image_count"]),
            "after_exam_count_used": int(after_meta["image_count"]),
            "query_date_used": str(task.get("query_date", "")),
            "annual_index_used": str(task.get("annual_index", task.get("relative_year", ""))),
            "relative_time_delta_days": query_meta["query_numeric"][2] * 730.5,
            "encoder_audit": {
                "before": before_meta,
                "after": after_meta,
                "query": query_meta,
            },
        }


def condition_for_generator(model: LongitudinalConditionModel, task: dict[str, Any], enabled: bool = True) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        return model.encode_task(task, enabled=enabled)


__all__ = [
    "ATTR_KEYS",
    "ConditionBridge",
    "ExamSetEncoder",
    "LongitudinalConditionModel",
    "LongitudinalQueryEncoder",
    "condition_for_generator",
    "image_feature",
    "split_exam_reports",
    "split_exam_sets",
    "target_proxy_features",
]
