from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "repro_baselines_final_l5_v3"
OUTPUT_MD = ROOT / "docs_database" / "BASELINE_REPRODUCTION_RESULTS.md"
OUTPUT_CSV = ROOT / "docs_database" / "baseline_reproduction_results.csv"
OUTPUT_ZH = ROOT / "docs_database" / "BASELINE_MODELS_AND_RESULTS_ZH.md"
MODEL_LABELS = {
    "global": "Global-feature Euclidean",
    "dtw": "Multivariate DTW",
    "svm": "RBF-SVM",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "two_stage_logistic": "Two-stage XGBoost + Logistic",
    "two_stage_xgb": "Two-stage XGBoost + XGBoost",
    "deepsets": "DeepSets (T2-only)",
    "cnn": "Siamese 1D-CNN",
    "bilstm": "Siamese BiLSTM",
    "transformer": "Siamese Transformer",
    "resnet18": "Siamese ResNet-18",
    "tarnn": "TA-RNN adapted",
    "tarnn_contrastive": "TA-RNN contrastive adapted",
    "lnps_rnn": "LNPS-RNN adapted",
    "synsig2vec": "SynSig2Vec official adapted",
}
MODEL_DETAILS = {
    "global": ("手工特征", "30 维全局统计特征经 Train 标准化后计算欧氏相似度"),
    "dtw": ("经典距离", "x/y/pressure/speed 多变量 DTW，20% Sakoe-Chiba window"),
    "svm": ("传统机器学习", "93 维对称 pair 特征 + RBF-SVM"),
    "rf": ("传统机器学习", "93 维对称 pair 特征 + 400 棵随机森林"),
    "xgb": ("传统机器学习", "93 维对称 pair 特征 + XGBoost"),
    "two_stage_logistic": ("集合判定", "XGBoost pair ranker + writer-group OOF Logistic gate"),
    "two_stage_xgb": ("集合判定", "XGBoost pair ranker + writer-group OOF XGBoost gate"),
    "deepsets": ("深度集合模型", "共享候选编码、mean/max 聚合、候选等变 rank head"),
    "cnn": ("深度序列", "128 点 10 通道动态序列 + Siamese 1D-CNN"),
    "bilstm": ("深度序列", "128 点 10 通道动态序列 + Siamese BiLSTM"),
    "transformer": ("深度序列", "128 点 10 通道动态序列 + Siamese Transformer"),
    "resnet18": ("深度图像", "CSV 确定性渲染 + ImageNet ResNet-18 Siamese head"),
    "tarnn": ("论文方法适配", "多变量 DTW 对齐 + Siamese BiLSTM + BCE"),
    "tarnn_contrastive": ("度量学习适配", "DTW 对齐 + L2 embedding + contrastive loss"),
    "lnps_rnn": ("论文方法适配", "长度归一化二维路径 + 二阶 prefix signature + BiLSTM"),
    "synsig2vec": ("官方代码适配", "Sigma-lognormal 合成 + selective-pooling 1D-CNN"),
}
FIELDS = (
    "t1_1v1_accuracy", "t1_1v1_eer", "t1_1v1_auc",
    "t1_5v1_accuracy", "t1_5v1_eer", "t1_5v1_auc",
    "t2_e0", "t2_e1", "t2_e2", "t2_a_balanced_accuracy",
    "t2_b_balanced_accuracy", "t2_rank1", "t2_rank3", "parameters", "runtime_seconds",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(model: str, seed: int) -> dict[str, Any]:
    metrics = read_json(RUNS / model / f"seed_{seed}" / "metrics.json")
    t1 = metrics.get("t1", {}).get("test", {})
    t2 = metrics["t2"]["test"]
    fit = metrics.get("t1_fit") or metrics.get("fit") or metrics.get("t2_fit") or {}
    one = t1.get("t1_1v1", {}).get("overall", {})
    five = t1.get("t1_5v1", {}).get("overall", {})
    return {
        "model": MODEL_LABELS[model], "model_id": model,
        "source": "final-l5-v3 reproduction", "seed": seed,
        "t1_1v1_accuracy": one.get("accuracy"), "t1_1v1_eer": one.get("eer"),
        "t1_1v1_auc": one.get("roc_auc"), "t1_5v1_accuracy": five.get("accuracy"),
        "t1_5v1_eer": five.get("eer"), "t1_5v1_auc": five.get("roc_auc"),
        "t2_e0": t2["e0_seven_class_accuracy"],
        "t2_e1": t2["e1_oracle_a_accuracy"], "t2_e2": t2["e2_oracle_ab_accuracy"],
        "t2_a_balanced_accuracy": t2["A_rf_vs_sf"]["balanced_accuracy"],
        "t2_b_balanced_accuracy": t2["B_source_in_pool_given_sf"]["balanced_accuracy"],
        "t2_rank1": t2["C_source_ranking"]["rank_1"],
        "t2_rank3": t2["C_source_ranking"]["rank_3"],
        "parameters": fit.get("parameters"), "runtime_seconds": metrics.get("total_seconds"),
    }


def documented_model_row() -> dict[str, Any]:
    return {
        "model": "Ours (final frozen documentation)", "model_id": "ours_documented",
        "source": "experiment_final/METRICS.md", "seed": 42,
        "t1_1v1_accuracy": 0.9484375, "t1_1v1_eer": 0.05625, "t1_1v1_auc": 0.988852,
        "t1_5v1_accuracy": 0.9734375, "t1_5v1_eer": 0.0260416667,
        "t1_5v1_auc": 0.996383,
        "t2_e0": 0.5620659722, "t2_e1": None, "t2_e2": None,
        "t2_a_balanced_accuracy": None, "t2_b_balanced_accuracy": None,
        "t2_rank1": 0.5034722222, "t2_rank3": 0.8350694444,
        "parameters": None, "runtime_seconds": None,
    }


def fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def mean_std(rows: list[dict[str, Any]], field: str) -> str:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return fmt(values[0])
    return f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"


def main() -> None:
    rows = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_LABELS:
        paths = sorted((RUNS / model).glob("seed_*/metrics.json"))
        model_rows = [metric_row(model, int(path.parent.name.removeprefix("seed_"))) for path in paths]
        if model_rows:
            by_model[model] = model_rows
            rows.extend(model_rows)
    if not rows:
        raise RuntimeError(f"No completed final-protocol runs found below {RUNS}")
    ours = documented_model_row()
    rows.append(ours)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = ("model", "model_id", "source", "seed", *FIELDS)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)

    seed42 = [row for values in by_model.values() for row in values if row["seed"] == 42]
    lines = [
        "# Final-protocol baseline reproduction results", "",
        "> Generated by `scripts/repro_baselines/summarize.py`; do not hand-edit numeric results.", "",
        "## Protocol", "",
        "- Final frozen writer-disjoint split: Train/Validation/Test = 55/12/12 writers.",
        "- Split digest: `ba64c09c53119be31f357c089987f706eb3266e4348d8398ff5d88e83dc0c48d`.",
        "- T2 uses five candidates and seven classes: five candidate positions, SF-source-absent, and RF-no-source.",
        "- Pair-score baselines use a Validation-only two-threshold adaptation for the three episode states; Test never selects thresholds or checkpoints.",
        "- Primary T2 metric is E0 seven-class accuracy. E1/E2 are oracle diagnostics; A/B/C expose rejection and ranking behavior.", "",
        "## Seed 42", "",
        "| Model | T1 1v1 Acc | EER | AUC | T1 5v1 Acc | EER | AUC | T2 E0 | E1 | E2 | A Bal Acc | B Bal Acc | Rank-1 | Rank-3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed42 + [ours]:
        lines.append("| " + " | ".join([
            row["model"], fmt(row["t1_1v1_accuracy"]), fmt(row["t1_1v1_eer"]),
            fmt(row["t1_1v1_auc"]), fmt(row["t1_5v1_accuracy"]), fmt(row["t1_5v1_eer"]),
            fmt(row["t1_5v1_auc"]), fmt(row["t2_e0"]), fmt(row["t2_e1"]), fmt(row["t2_e2"]),
            fmt(row["t2_a_balanced_accuracy"]), fmt(row["t2_b_balanced_accuracy"]),
            fmt(row["t2_rank1"]), fmt(row["t2_rank3"]),
        ]) + " |")
    lines.extend(["", "Constant SF-absent and constant RF-no-source controls both have theoretical E0 = `0.2500`; random candidate Rank-1/Rank-3 = `0.2000`/`0.6000`.", ""])

    multi = {model: values for model, values in by_model.items() if len(values) > 1}
    if multi:
        lines.extend(["## Multiple Seeds", "", "| Model | Seeds | T1 1v1 AUC | T1 5v1 AUC | T2 E0 | Rank-1 | Rank-3 |", "|---|---:|---:|---:|---:|---:|---:|"])
        for model, values in multi.items():
            lines.append(
                f"| {MODEL_LABELS[model]} | {len(values)} | {mean_std(values, 't1_1v1_auc')} | "
                f"{mean_std(values, 't1_5v1_auc')} | {mean_std(values, 't2_e0')} | "
                f"{mean_std(values, 't2_rank1')} | {mean_std(values, 't2_rank3')} |"
            )
        lines.append("")

    best = max(seed42, key=lambda row: row["t2_e0"])
    lines.extend([
        "## Interpretation", "",
        f"- The strongest reproduced seed-42 T2 E0 is {best['model']} at {best['t2_e0']:.4f}; the frozen self-model E0 is {ours['t2_e0']:.4f}.",
        "- E0 is the only primary T2 comparison. E1/E2 and A/B/C diagnose whether errors come from RF/SF separation, source presence, or candidate ranking.",
        "- TA-RNN, LNPS-RNN, DeepSets and the generic seven-class decision layer are benchmark adaptations, not claims of reproducing the original papers' published values.",
        "- Detailed predictions, calibration, checkpoints and environments remain under `runs/repro_baselines_final_l5_v3/`.", "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    zh = [
        "# 最终协议外部基线：模型、方法与结果", "",
        "> 由 `scripts/repro_baselines/summarize.py` 从运行产物自动生成，请勿手工修改数值。", "",
        "## 实验协议", "",
        "- Train/Validation/Test 为 55/12/12 个 writer，三者 writer-disjoint。",
        "- 最终 split digest：`ba64c09c53119be31f357c089987f706eb3266e4348d8398ff5d88e83dc0c48d`。",
        "- T1 包含 1v1 与 5v1；T2 为 5 个候选加 SF-source-absent、RF-no-source，共 7 类。",
        "- T2 主指标是 E0 七分类准确率。E1/E2 是 oracle 诊断，Rank-1/3 仅衡量 source-present 候选排序。",
        "- 所有模型只读取 CSV 动态信号衍生特征；身份、路径、候选位置和答案元数据不进入模型。",
        "- Pair-score 基线没有自研 ABC 头，统一用 Validation-only 两阈值把模型分数适配到三种 episode 状态；Test 不参与选阈值。", "",
        "## 模型与 Seed 42 结果", "",
        "| 模型 | 类型 | 核心做法 | T1 1v1 AUC | T1 5v1 AUC | T2 E0 | Rank-1 | Rank-3 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in seed42:
        model_id = row["model_id"]
        kind, method = MODEL_DETAILS[model_id]
        zh.append(
            f"| {row['model']} | {kind} | {method} | {fmt(row['t1_1v1_auc'])} | "
            f"{fmt(row['t1_5v1_auc'])} | {fmt(row['t2_e0'])} | "
            f"{fmt(row['t2_rank1'])} | {fmt(row['t2_rank3'])} |"
        )
    zh.append(
        f"| {ours['model']} | 自研模型 | 最终冻结文档权威结果 | {fmt(ours['t1_1v1_auc'])} | "
        f"{fmt(ours['t1_5v1_auc'])} | {fmt(ours['t2_e0'])} | "
        f"{fmt(ours['t2_rank1'])} | {fmt(ours['t2_rank3'])} |"
    )
    zh.extend([
        "", "## 稳定性与结论", "",
        "- 恒定预测 SF-source-absent 或 RF-no-source 的理论 E0 都是 0.2500；随机候选 Rank-1/3 为 0.2000/0.6000。",
        f"- Seed 42 最强外部基线是 {best['model']}，E0={best['t2_e0']:.4f}；自研模型 E0={ours['t2_e0']:.4f}。",
        "- T2 必须以 E0 作主比较。单独 Rank-1 较高不代表完整七分类任务更好，因为模型还需区分 RF、SF 来源缺失和来源在池。",
        "- TA-RNN、LNPS-RNN、DeepSets 以及通用七分类决策层均是本 benchmark 的方法适配，不声称复现原论文在原数据集上的数值。",
    ])
    if multi:
        zh.extend(["", "### 三种子结果", "", "| 模型 | T2 E0 mean +/- std | Rank-1 | Rank-3 |", "|---|---:|---:|---:|"])
        for model, values in multi.items():
            zh.append(
                f"| {MODEL_LABELS[model]} | {mean_std(values, 't2_e0')} | "
                f"{mean_std(values, 't2_rank1')} | {mean_std(values, 't2_rank3')} |"
            )
    zh.extend(["", "完整 E0/E1/E2、A/B/C 指标、逐 seed CSV、预测与校准信息分别见 `BASELINE_REPRODUCTION_RESULTS.md`、`baseline_reproduction_results.csv` 和 `runs/repro_baselines_final_l5_v3/`。", ""])
    OUTPUT_ZH.write_text("\n".join(zh), encoding="utf-8")
    print(OUTPUT_MD)
    print(OUTPUT_CSV)
    print(OUTPUT_ZH)


if __name__ == "__main__":
    main()
