from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from threatlens_lite.features import build_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build base + temporal features from combined conn data.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=ROOT / "data" / "interim" / "conn_raw.parquet",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=ROOT / "data" / "processed" / "features_v1.parquet",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "features_v1.csv",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=30,
        help="Temporal aggregation window in seconds.",
    )
    args = parser.parse_args()

    raw_df = pd.read_parquet(args.input_parquet)
    feature_df = build_feature_table(raw_df, window_seconds=args.window_seconds)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_parquet(args.output_parquet, index=False)
    feature_df.to_csv(args.output_csv, index=False)

    dataset_card = {
        "name": "ThreatLens-lite v1",
        "rows": int(len(feature_df)),
        "window_seconds": int(args.window_seconds),
        "n_malicious": int(feature_df["label"].sum()),
        "n_normal": int((feature_df["label"] == 0).sum()),
        "captures": feature_df["capture_id"].nunique(),
        "hosts": feature_df["group_id"].nunique(),
        "columns": feature_df.columns.tolist(),
    }
    with (args.output_parquet.parent / "dataset_card.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_card, handle, indent=2)

    print(f"Saved features to: {args.output_parquet}")
    print(f"Rows: {len(feature_df):,}")
    print(f"Columns: {len(feature_df.columns)}")


if __name__ == "__main__":
    main()
