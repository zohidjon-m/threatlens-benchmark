from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from threatlens_lite.modeling import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on ThreatLens-lite features.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=ROOT / "data" / "processed" / "features_v1.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    feature_df = pd.read_parquet(args.input_parquet)
    metrics = train_and_evaluate(
        feature_df=feature_df,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )

    print("Finished training baselines.")
    for item in metrics:
        print(
            f"{item['model']}: roc_auc={item['roc_auc']:.4f}, "
            f"f1={item['f1']:.4f}, precision={item['precision']:.4f}, recall={item['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
