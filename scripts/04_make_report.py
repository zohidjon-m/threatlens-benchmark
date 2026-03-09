from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_metrics(metrics_path: Path) -> list[dict]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a markdown report from model outputs.")
    parser.add_argument(
        "--features-parquet",
        type=Path,
        default=ROOT / "data" / "processed" / "features_v1.parquet",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=ROOT / "reports" / "metrics.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports" / "threatlens_lite_results.md",
    )
    args = parser.parse_args()

    feature_df = pd.read_parquet(args.features_parquet)
    metrics = load_metrics(args.metrics_json)
    best_model = metrics[0]

    lines: list[str] = []
    lines.append("# ThreatLens-lite Results\n")
    lines.append("## Problem\n")
    lines.append(
        "ThreatLens-lite predicts whether a network flow is benign or malicious using Zeek `conn.log` telemetry, "
        "feature engineering, temporal aggregation, and baseline machine-learning models.\n"
    )

    lines.append("## Dataset Summary\n")
    lines.append(f"- Total rows: {len(feature_df):,}")
    lines.append(f"- Malicious rows: {int(feature_df['label'].sum()):,}")
    lines.append(f"- Normal rows: {int((feature_df['label'] == 0).sum()):,}")
    lines.append(f"- Unique capture IDs: {feature_df['capture_id'].nunique():,}")
    lines.append(f"- Unique host groups: {feature_df['group_id'].nunique():,}\n")

    lines.append("## Baseline Results\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for item in metrics:
        lines.append(
            f"| {item['model']} | {item['accuracy']:.4f} | {item['precision']:.4f} | "
            f"{item['recall']:.4f} | {item['f1']:.4f} | {item['roc_auc']:.4f} |"
        )
    lines.append("")

    importance_csv = ROOT / "reports" / f"feature_importance_{best_model['model']}.csv"
    if importance_csv.exists():
        importance_df = pd.read_csv(importance_csv).head(10)
        lines.append("## Top Features (Best Model)\n")
        lines.append(f"Best model: **{best_model['model']}**\n")
        lines.append("| Feature | Importance |")
        lines.append("|---|---:|")
        for _, row in importance_df.iterrows():
            lines.append(f"| {row['feature']} | {row['importance']:.6f} |")
        lines.append("")

    lines.append("## Notes\n")
    lines.append("- Temporal features summarize host behavior over a rolling 30-second window.")
    lines.append("- Labels are assigned from the source folder (`normal` vs `malicious`).")
    lines.append("- Results should be interpreted carefully if benign and malicious captures come from very different environments.")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote markdown report to: {args.output_md}")


if __name__ == "__main__":
    main()
