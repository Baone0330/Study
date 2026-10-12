"""Python-only source-data-driven nonmedical publication figures and explicit QA."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

from cpfm_common import atomic_json, now, read_json, write_csv

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .7, "legend.frameon": False})
ON = "#4679A8"
OFF = "#C88743"
GRAY = "#81909C"


def displayed_texts(fig):
    """Exclude locator-created ticks that Matplotlib itself does not draw.

    Axis.draw uses _update_ticks to filter tick locations against the view
    interval. get_visible alone is insufficient for these inactive labels.
    No data, plot limits, tick locations or rendered appearance is changed.
    """
    all_tick_labels, rendered_tick_labels = set(), set()
    for axis in fig.axes:
        for coord in (axis.xaxis, axis.yaxis):
            all_tick_labels.add(coord.get_offset_text())
            for tick in coord.get_major_ticks() + coord.get_minor_ticks():
                all_tick_labels.update((tick.label1, tick.label2))
            if axis.get_visible() and axis.axison and coord.get_visible():
                rendered_tick_labels.add(coord.get_offset_text())
                for tick in coord._update_ticks():
                    rendered_tick_labels.update((tick.label1, tick.label2))
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text(): continue
        if text.axes is not None and not text.axes.get_visible(): continue
        if text in all_tick_labels and text not in rendered_tick_labels: continue
        yield text


def check_text_boundaries(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    texts = list(displayed_texts(fig))
    bad = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            bad.append(text.get_text())
    if bad: raise RuntimeError("Figure text outside canvas: " + repr(bad))
    return {"rendered_text_count": len(texts),
            "minimum_rendered_font_points": min((t.get_fontsize() for t in texts), default=7),
            "tick_filter": "exact installed Matplotlib Axis.draw tick list; inactive labels excluded"}


def export(root, fig, stem, caption_en, caption_zh):
    for folder in ("png", "svg", "qa", "captions", "source_csv"):
        (root / "figures" / folder).mkdir(parents=True, exist_ok=True)
    text_qa = check_text_boundaries(fig)
    assert text_qa["minimum_rendered_font_points"] >= 7
    png = root / "figures/png" / f"{stem}.png"
    svg = root / "figures/svg" / f"{stem}.svg"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(svg, facecolor="white")
    with Image.open(png) as image:
        assert image.mode in ("RGB", "RGBA")
        if image.mode != "RGB":
            image.convert("RGB").save(png, dpi=(600, 600))
        size = image.size
    assert "<text" in svg.read_text(encoding="utf-8")
    (root / "figures/captions" / f"{stem}_CN_EN.md").write_text(caption_en + "\n\n" + caption_zh + "\n", encoding="utf-8")
    plt.close(fig)
    return {"figure": stem, "dpi": 600, "pixel_dimensions": size, "svg_editable_text": True, **text_qa,
            "text_boundary_check": "PASS", "agent_visual_review": "NOT_RUN", "medical_review": "NOT_RUN", "contains_medical_pixels": False}


def mechanism(root):
    fig, ax = plt.subplots(figsize=(183/25.4, 136/25.4))
    fig.subplots_adjust(left=.025, right=.985, top=.92, bottom=.065)
    ax.set(xlim=(0, 100), ylim=(0, 72))
    ax.axis("off")
    fig.suptitle("Original Progression: isolated CPFM injection bypass", fontsize=10, y=.977)
    def box(x, y, w, h, text, color="#E6EDF4", fontsize=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1", facecolor=color, edgecolor="#758A9E", lw=.7))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=max(7, fontsize))
    def arrow(x1, y1, x2, y2, color="#566979", style="-"):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "lw": .85, "color": color, "linestyle": style})
    for x, title in ((16, "a  Original condition computation"), (51, "b  Frozen original B14 controller"), (86, "c  Original UNet / bypass")):
        ax.text(x, 69, title, ha="center", va="top", fontsize=8, weight="bold")
    box(1, 56, 29, 7, "REAL historical images\nGray8 statistics + zero report6")
    box(1, 44, 13.5, 9, "BEFORE14\nEarlier exams\nLinear32, LN\nGELU, Linear16\nImage mean", "#DAE8ED", 6.6)
    box(16.5, 44, 13.5, 9, "AFTER14\nLatest / per-Exam\nLinear32, LN\nGELU, Linear16\nImage mean", "#DAE8ED", 6.6)
    arrow(8, 56, 8, 53); arrow(23, 56, 23, 53)
    box(1, 33, 29, 8, "Adjacent exam deltas16\nLinear32 → LN → GELU →16\nValid masked mean → P16", "#E9E0F0", 7)
    arrow(23, 44, 23, 41)
    box(1, 23, 29, 7, "Query9 → Linear24 → LN\nGELU → time16", "#F4E9D7", 7)
    box(1, 8, 29, 11, "Concat(B16, A16, time16, P16)=64\nLinear128 → LN → GELU →64\nAdapter64→16, tanh\nSame condition16 in ON and OFF", "#E9E0F0", 6.8)
    arrow(8, 44, 0, 42); arrow(0, 42, 0, 14); arrow(0, 14, 1, 14)
    arrow(23, 33, 31.3, 33); arrow(31.3, 33, 31.3, 14); arrow(31.3, 14, 30, 14)
    arrow(29, 44, 32, 44); arrow(32, 44, 32, 17); arrow(32, 17, 30, 17)
    arrow(15, 23, 15, 19)
    box(35, 51, 15, 12, "Magnitude input32\nc16 + time8 + layer8\nLN32 →128 SiLU\n→128 SiLU\nγ/β magnitude σ", "#DDEBED", 6.5)
    box(52, 51, 16.5, 12, "Direction input64\nc16/h16/s16/t8/l8\nLN64 →128 SiLU\n→128 SiLU\nγ/β channel heads", "#DDEBED", 6.5)
    box(35, 35, 33.5, 11, "Patient state: c16+h16 (h=0)\nLN32 →32 SiLU → z16\nPatient direction input64 → LN64\n→64 SiLU → γ/β channel heads\nAll original B14 weights frozen", "#DDEBED", 6.8)
    box(35, 20, 33.5, 11, "D(c)−D(0) + 0.25[P(c)−P(0)]\nRMS-normalized directions × magnitudes\nβ safety sigmoid, layer-scale sigmoid\nOriginal γ/β values are not set to zero", "#DDEBED", 6.6)
    # Magnitude/direction/patient branches are parallel, not a sequential cascade.
    arrow(43, 51, 34, 49); arrow(34, 49, 34, 25); arrow(34, 25, 35, 25)
    arrow(60, 51, 70, 49); arrow(70, 49, 70, 25); arrow(70, 25, 68.5, 25)
    arrow(51, 35, 51, 31)
    box(35, 7.5, 33.5, 8, "ONLY switch: holder.enabled\nON=True / OFF=False\nP16 remains calculated", "#FAEDD9", 7)
    arrow(51, 20, 51, 15.5)
    # Condition enters the original controller branches, not the Boolean gate.
    arrow(30, 11, 33, 11); arrow(33, 11, 33, 65)
    arrow(33, 65, 43, 65); arrow(43, 65, 43, 63)
    arrow(43, 65, 60, 65); arrow(60, 65, 60, 63)
    arrow(33, 41, 35, 41)
    box(74, 51, 24.5, 12, "Original D0–D3\nfirst-ResNet input hooks\nChannels320/320/640/1280\n31 UNet calls / batch\nOriginal SD1.5 + medical LoRA", "#E3ECF5", 6.8)
    box(74, 33, 24.5, 12, "ON: original residual\ny=x+σ(scale)[(1+γ)u+β−u]\nu=HW standardization\n124 effective layer calls\nper six-image sampling batch", "#E3ECF5", 6.6)
    box(74, 19, 24.5, 9, "OFF: y=x, exact bypass\n0 effective modulation calls\nNo remaining P16→UNet path", "#FAEDD9", 6.8)
    arrow(86, 51, 86, 45); arrow(68.5, 11.5, 73.5, 23.5, OFF)
    arrow(98, 51, 99.3, 49); arrow(99.3, 49, 99.3, 23.5, style="--"); arrow(99.3, 23.5, 98.5, 23.5, style="--")
    arrow(68.5, 14, 71, 14, ON); arrow(71, 14, 71, 39, ON); arrow(71, 39, 74, 39, ON)
    box(74, 6, 24.5, 8, "Original PNDM30 / CFG7\nVAE → six RGB512 images\nSame six original seeds", "#E7EEE4", 6.8)
    arrow(86, 33, 72, 31); arrow(72, 31, 72, 10); arrow(72, 10, 74, 10)
    arrow(86, 19, 86, 14)
    fig.text(.5, .025, "Injection-path dependence in the original historical protocol; not independent P16/CPFM causal decomposition", ha="center", fontsize=7)
    write_csv(root / "figures/source_csv/Fig01_Source.csv", [
        {"object": "Original P16", "input_dim": 64, "hidden_dim": 128, "fusion_output": 64, "condition_output": 16},
        {"object": "B14 physical hooks per batch", "input_dim": 4, "hidden_dim": 31, "fusion_output": 124, "condition_output": "OFF0/ON124"}])
    return export(root, fig, "Fig01_CPFM_ON_OFF_Mechanism",
                  "Fig. 1 | Original Progression CPFM injection intervention. a The historical F16 and P16 computations retain their original encoders, masked pooling, fusion and adapter. b Original B14 magnitude, direction and patient-direction paths are frozen. Only the original enabled gate changes. c ON applies the original residual; OFF returns the original activation. Counts are physical calls per six-image batch (31 UNet calls and four layer hooks), not six independent count replicates. Progression is still computed but has no active UNet path in OFF. This identifies injection-path dependence, not independent module causality. No medical image is shown.",
                  "图1 | 原Original Progression的CPFM注入干预。a 原F16/P16编码、掩码均值池化、融合和adapter保持不变。b 原B14幅度、方向及患者方向路径冻结，只切换原enabled门。c ON应用原残差，OFF直接返回原激活。物理调用数以六图batch为单位（31次UNet、4层hook），不能再乘以6当作独立重复。OFF仍计算Progression，但无活跃UNet通路；衡量的是注入路径依赖，不是独立模块因果贡献。图中不含医学图像。")


def quantitative(root):
    qa = []
    table = pd.read_csv(root / "tables/Table06_Updated_Module_Ablation_Table.csv")
    labels = ["REAL", "Full", "Old Plain", "History", "Time", "Original ON", "New OFF"]
    colors = [GRAY] * 5 + [ON, OFF]
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 100/25.4))
    fig.subplots_adjust(left=.075, right=.98, bottom=.29, top=.86, wspace=.3)
    for axis, metric in zip(axes, ("ROC_AUC", "AP")):
        values = table[metric].astype(float)
        axis.bar(np.arange(7), values, color=colors, width=.65)
        axis.set(ylim=(0, 1), ylabel=metric.replace("_", "-"), xticks=np.arange(7))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        for i, value in enumerate(values): axis.text(i, value+.025, f"{value:.3f}", ha="center", fontsize=7)
    fig.suptitle("Historical references and new matched CPFM OFF", fontsize=10)
    shutil.copy2(root / "tables/Table06_Updated_Module_Ablation_Table.csv", root / "figures/source_csv/Fig02_Source.csv")
    qa.append(export(root, fig, "Fig02_Historical_References_New_OFF", "Fig. 2 | Historical internal-validation references and new CPFM OFF. Bars show three-seed ensemble ROC-AUC and average precision (AP) for 189 patients (58 positive, 131 negative). Grey bars retain the old six-branch background; only original ON (blue) and new OFF (orange) form the route-H controlled comparison. The old Plain-SD branch is not the new OFF intervention. No seed spread or confidence interval is implied by the bars. Val was used for checkpoint selection.", "图2 | 历史内部验证参考与新增CPFM OFF。条形为189名患者（阳性58/阴性131）的三seed ensemble ROC-AUC与average precision（AP）。灰色为旧路线背景，只有原ON（蓝）与新OFF（橙）构成路线H受控比较。旧Plain-SD不是新增OFF。条形不代表seed离散度或置信区间；Val用于选模。"))
    ci = pd.read_csv(root / "tables/Table04_Paired_Patient_Bootstrap.csv")
    fig, ax = plt.subplots(figsize=(183/25.4, 78/25.4))
    fig.subplots_adjust(left=.18, right=.96, bottom=.19, top=.84)
    x = ci.point_difference.to_numpy()
    ax.hlines([1, 0], ci.ci_lower_95, ci.ci_upper_95, color=ON, lw=1.2)
    ax.scatter(x, [1, 0], color=ON, s=28, zorder=3)
    for y_pos, lower, upper in zip([1, 0], ci.ci_lower_95, ci.ci_upper_95):
        ax.vlines([lower, upper], y_pos-.08, y_pos+.08, color=ON, lw=1.2)
    ax.axvline(0, color=GRAY, lw=.8, ls="--")
    ax.set(yticks=[1, 0], yticklabels=["ROC-AUC", "AP"], ylim=(-.6, 1.6), xlabel="ON − OFF (positive favours original ON)")
    low, high = min(0, ci.ci_lower_95.min(), x.min()), max(0, ci.ci_upper_95.max(), x.max())
    span = max(.025, high-low)
    ax.set_xlim(low-.15*span, high+.15*span)
    fig.suptitle("Paired patient-level uncertainty", fontsize=10)
    shutil.copy2(root / "tables/Table04_Paired_Patient_Bootstrap.csv", root / "figures/source_csv/Fig03_Source.csv")
    qa.append(export(root, fig, "Fig03_Paired_Bootstrap_Differences", "Fig. 3 | Paired ON-minus-OFF differences. Points show the observed ROC-AUC/AP differences. Whiskers are two-sided percentile 95% intervals from 2,000 stratified paired patient resamples (seed 20261012), preserving 58 positive and 131 negative draws. The same draw indexes both arms after averaging three model seeds within each patient. The dashed line is zero. These intervals describe reused internal Val, not independent clinical uncertainty; no multiplicity-adjusted significance is asserted.", "图3 | 配对ON减OFF差异。点为ROC-AUC/AP实测差值，误差线为2000次分层配对患者bootstrap（seed20261012）的双侧percentile95%区间，每次保留阳性58/阴性131。先在每名患者内平均三个模型seed，再以相同索引抽取两臂。虚线为零。区间描述复用内部Val的不确定性，不是独立临床区间，也不宣称经过多重校正的显著性。"))
    image_diff = pd.read_csv(root / "figures/source_csv/GENERATION_IMAGE_DIFFERENCES_ANONYMOUS_V1.csv")
    pilot = pd.read_csv(root / "pilot/metrics/FIXED_36_PAIR_METRICS_V1.csv")
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 135/25.4))
    fig.subplots_adjust(left=.09, right=.97, top=.9, bottom=.12, hspace=.48, wspace=.38)
    bins_out = []
    series = [(image_diff.on_off_pixel_rmse, "RGB RMSE (0–255)", "full9270"), (image_diff.on_off_generated_f16_rmse, "Generated F16 RMSE", "full9270"),
              (pilot.on_off_latent_rmse, "Final latent RMSE", "fixed36 pilot"), (pilot.early_unet_cond_rmse, "Early conditional noise RMSE", "fixed36 pilot")]
    for axis, (values, label, support) in zip(axes.flat, series):
        hist, edges = np.histogram(values, bins=30)
        axis.bar(edges[:-1], hist, width=np.diff(edges), align="edge", color=OFF, edgecolor="none")
        axis.set(xlabel=label, ylabel=f"Image count ({support})")
        bins_out.extend({"metric": label, "scope": support, "bin_left": float(a), "bin_right": float(b), "count": int(n)} for a, b, n in zip(edges[:-1], edges[1:], hist))
    zero = int(image_diff.on_off_pixel_bitexact.sum())
    fig.suptitle(f"Numerical transmission differences; zero RGB response {zero}/9270", fontsize=9)
    write_csv(root / "figures/source_csv/Fig04_Binned_Aggregated_Source.csv", bins_out)
    qa.append(export(root, fig, "Fig04_Engineering_Difference_Distributions", "Fig. 4 | Numerical ON/OFF transmission differences. Top panels show full-support RGB-pixel and generated-image F16 RMSE for 9,270 image pairs across 1,545 nodes and 860 patients. Bottom panels show final-latent and early conditional-UNet-noise RMSE for the predeclared 36 full-sampling pilot pairs only. Images are technical observations; neither 9,270 images nor 36 pilot images are independent patient samples. Zero-response outputs were retained. Differences do not establish pathology or anatomical realism. Public source data contain binned aggregates only.", "图4 | ON/OFF数值传递差异。上方是1545节点、860患者、9270对图的全量RGB像素与生成图F16 RMSE；下方仅为事前固定36对完整采样pilot的最终latent和early conditional UNet noise RMSE。图像为技术观察，9270或36都不是独立患者样本数。零响应输出保留。差异不证明病理或解剖真实性；公开源数据仅为分箱聚合。"))
    seed_off = pd.read_csv(root / "tables/Table02_New_CPFM_OFF_Three_Seeds.csv")
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    seed_on = pd.read_csv(Path(cfg["paths"]["replay_root"]) / "reports/SEED_LEVEL_ACTUAL_METRICS_V1.csv")
    seed_on = seed_on[seed_on.branch_id == "B16V1_PLUS_ORIGINAL_PROGRESSION"].rename(columns={"PR_AUC": "AP"})
    ensemble = pd.read_csv(root / "tables/Table03_New_ON_OFF_Ensemble.csv")
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 95/25.4))
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.17, wspace=.35)
    seed_source = []
    for j, (frame, arm, color) in enumerate(((seed_on, "ON", ON), (seed_off, "OFF", OFF))):
        for i, metric in enumerate(("ROC_AUC", "AP")):
            values = frame[metric].astype(float).to_numpy()
            axes[0].scatter(np.full(3, i)+j*.25-.125+np.array([-.035, 0, .035]), values, color=color, s=20, label=arm if i == 0 else None)
            seed_source.extend({"arm": arm, "seed": int(s), "metric": metric, "value": float(v)} for s, v in zip(frame.seed, values))
    axes[0].set(xticks=[0, 1], xticklabels=["ROC-AUC", "AP"], ylim=(0, 1), ylabel="Single model seed")
    axes[0].legend(loc="lower right")
    for j, color in enumerate((ON, OFF)):
        axes[1].bar(np.arange(3)+(j-.5)*.32, ensemble.iloc[j][["F1", "Sensitivity", "Specificity"]].to_numpy(dtype=float), width=.32, color=color)
    axes[1].set(xticks=np.arange(3), xticklabels=["F1", "Sens", "Spec"], ylim=(0, 1), ylabel="Three-seed ensemble")
    fig.suptitle("Seed stability and threshold metrics", fontsize=10)
    write_csv(root / "figures/source_csv/Fig05_Seed_Source.csv", seed_source)
    shutil.copy2(root / "tables/Table03_New_ON_OFF_Ensemble.csv", root / "figures/source_csv/Fig05_Ensemble_Source.csv")
    qa.append(export(root, fig, "Fig05_Seed_Stability_Threshold_Metrics", "Fig. 5 | Model-seed stability and ensemble threshold metrics. Left dots show three original ON model seeds and three newly trained OFF seeds, not additional patient replicates; no seed confidence interval is shown. Right bars show ensemble F1, sensitivity and specificity on the same 189 internal-Val patients. Each arm uses its own historical Youden-J threshold with the smallest-threshold tie rule; thresholds were not optimized for F1. Blue denotes ON and orange OFF.", "图5 | 模型seed稳定性与ensemble阈值指标。左侧点为三个原ON模型seed和三个新训练OFF seed，不是额外患者重复，也未给seed置信区间。右侧条形为同189内部Val患者的ensemble F1、敏感度和特异度。每臂按原Youden J及最小阈值tie规则取自身阈值，没有另调阈值优化F1。蓝色ON、橙色OFF。"))
    return qa


def private_medical_plate(root):
    # This assembly never leaves the private server root or enters public stage.
    fig, axes = plt.subplots(6, 12, figsize=(12, 7))
    fig.subplots_adjust(left=.025, right=.99, top=.91, bottom=.065, wspace=.025, hspace=.07)
    for n in range(6):
        for seed in range(6):
            for arm_ix, arm in enumerate(("ON", "OFF")):
                axis = axes[n, seed*2+arm_ix]
                with Image.open(root / "pilot/fixed_36_pairs" / f"node_{n+1:02d}/{arm}" / f"gen_{seed:02d}.png") as im:
                    axis.imshow(im.convert("RGB"))
                axis.set_xticks([]); axis.set_yticks([])
                if n == 0: axis.set_title(f"{arm} i{seed}", fontsize=8)
    fig.suptitle("PRIVATE — fixed full pilot: every node and every seed; no quality selection", fontsize=11)
    fig.text(.5, .015, "Synthetic 512×512 images; no physical scale calibration; medical expert review NOT_RUN", ha="center", fontsize=9)
    fig.savefig(root / "pilot/qa/PRIVATE_FIXED_36_ON_OFF_IMAGE_PLATE.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--mechanism-only", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    (root / "figures/source_csv").mkdir(parents=True, exist_ok=True)
    qa = [mechanism(root)]
    if not a.mechanism_only:
        assert read_json(root / "release/FINAL_METRICS_STATUS_V1.json")["status"] == "MATCHED_INTERNAL_VAL_ABLATION_COMPLETED"
        qa.extend(quantitative(root))
        private_medical_plate(root)
    atomic_json(root / "figures/qa/AUTOMATIC_FIGURE_QA_V1.json", {"status": "AUTO_PASS_ACTUAL_REVIEW_NOT_RUN", "figures": qa, "backend": "Python", "created_utc": now()})
