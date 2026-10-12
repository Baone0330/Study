"""Freeze H before any new downstream metric. M is refused by this H executor."""
import argparse
import json
from pathlib import Path

from cpfm_common import atomic_json, now, read_json


def main(root):
    root = Path(root)
    pilot = read_json(root / "pilot/qa/FIXED_36_PILOT_COMPLETED_V1.json")
    repeat = read_json(root / "pilot/qa/NEW_PROCESS_REPEAT_COMPLETED_V1.json")
    anonymous = read_json(root / "pilot/qa/ANONYMOUS_TOGGLE_TEST_V1.json")
    assert anonymous["status"] == repeat["status"] == "PASS"
    assert pilot["pairs"] == pilot["historical_on_exact_pixels"] == 36
    assert pilot["historical_on_max_abs_pixel_error"] == pilot["off_effective_modulations"] == 0
    cfg_path = root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json"
    cfg = read_json(cfg_path)
    if cfg["comparison_route"] == "H":
        print("Route H already frozen; no change")
        return
    assert cfg["comparison_route"] == "UNRESOLVED_BEFORE_PILOT"
    cfg.update({"comparison_route": "H", "route_frozen_utc": now(), "pair_validity": "HISTORICAL_ON_PARITY_ESTABLISHED",
                "parity_criterion": "36/36 decoded RGB pixels exactly equal at original batch6, full master/source/seed/slot lineage recovered; independent process replay exact",
                "historical_on_pixel_parity_scope": "fixed first6 stable node keys, all6 seeds; full9270 image identity/source support verified, not all9270 resampled",
                "downstream_metrics_observed_before_route": 0, "formal_generation_arms": ["HIST_ORIGINAL_P16_CPFM_OFF"],
                "formal_target_images": 9270, "downstream_target_runs": 3})
    atomic_json(cfg_path, cfg)
    atomic_json(root / "release/PAIR_VALIDITY_CLASSIFICATION_V1.json", {"status": "HISTORICAL_ON_PARITY_ESTABLISHED", "comparison_route": "H",
                "chosen_before_new_downstream_metrics": True, "pilot": pilot, "independent_process_replay": repeat, "route_frozen_utc": cfg["route_frozen_utc"]})
    contract = """# ON/OFF 单变量干预合同\n\n任务身份：HISTORICAL_ORIGINAL_PROGRESSION_CPFM_OFF_MATCHED_ABLATION_V1。\n\n使用原 Original Progression 20260921/epoch120、原 CurrentF16 和原 B14 权重，全部 eval/frozen。BEFORE/AFTER/query/逐检查差分/P16/fusion64/condition16 在两臂按原 make_condition 计算，源码与权重不改。\n\n唯一开关为旧 B14Hooks._active 读取的 holder.enabled；OFF 直接返回原 inputs，未将 condition16 或 gamma/beta 头改为零。两臂仍注册相同5条物理 pre-hook，原4层 ResNet pre-hook 每层31次。ON 启用124次调制，OFF 为0次。原 PNDM 30 requested steps 实际31个 UNet calls。原6图batch、六个旧seed、fp16、CFG7、512、prompt/negative prompt、未显式覆写 LoRA scale 全部不变。\n\n原 ON 的数学式是 x + sigmoid(layer_scale) * [((1+gamma)*u+beta)-u]，u 为按HW标准化的激活；OFF 是 x。原 condition_scale 仍计算，但原调用路径未消费它。\n\nCPFM是唯一活跃 P16→UNet 路径。OFF 的 P16 仍计算但没有活跃UNet路径，必须标记 PROGRESSION_CALCULATED_BUT_NO_ACTIVE_UNET_PATH / INTERVENTION_PATH_DEPENDENT。差异衡量该背景下的注入路径依赖，不证明P16与CPFM因果贡献完全分解。\n\n固定6node×6seed完整pilot中的条件、prompt embedding、initial latent严格相同；36张新ON与旧正式ON解码像素严格相同。新进程重采样6张ON+6张OFF latent/pixel严格复现，技术样本不计正式分母。所有临时hook在finally移除；匿名ON等价/OFF恒等/异常移除测试PASS。\n\n路线H在新下游指标出现前冻结。9270张旧正式ON完整可达，1545节点×6seed与原860 Train/189 Val/K6槽位逐项匹配；只有36张进行真实重采样，不能宣称9270张全部重新采样。旧Python库完整历史revision仍UNKNOWN，当前固定pilot的输出复現证据是实测而非版本号假定。\n\n全量OFF按原6图batch采样。SQLite事务保存逐图状态、原seed和合同；部分node恢复仍重新计算同6seed全batch，对已完成图先做解码像素等值检查且不覆盖，仅提交pending图。没有凭PNG存在跳过，FAILED记录和事件留存。不改变batch/steps/dtype/sampler/seed。\n\n正式全量逐图测量ON/OFF RGB与CurrentF16差异；UNet/latent全采样差异来自固定36对pilot，旧全量ON未保存latent，未测部分不得虚构。医学专家评阅/Official Test未运行。\n"""
    for relative in ("audit/ON_OFF_INTERVENTION_CONTRACT_V1.md", "protocol/ORIGINAL_A_REPLAY_PARITY_CONTRACT_V1.md", "protocol/BRANCH_INTERPRETATION_LIMITS_V1.md"):
        (root / relative).write_text(contract, encoding="utf-8")
    atomic_json(root / "release/EXECUTION_STATE_V1.json", {"status": "ROUTE_H_FROZEN_FORMAL_GENERATION_PENDING", "phase": 3, "updated_utc": now(), "test_access": 0})
    print(json.dumps({"status": "ROUTE_H_FROZEN", "formal_images": 9270, "new_downstream_runs": 3, "frozen_utc": cfg["route_frozen_utc"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
