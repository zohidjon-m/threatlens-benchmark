from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threatlens_lite.zeek import discover_conn_files, load_conn_file


LABEL_MAP = {
    "normal": 0,
    "malicious": 1,
}


def load_class_logs(class_dir: Path, class_name: str) -> list[pd.DataFrame]:
    conn_files = discover_conn_files(class_dir)
    if not conn_files:
        raise FileNotFoundError(f"No conn.log or conn.log.csv files found under {class_dir}")

    frames: list[pd.DataFrame] = []
    for index, conn_file in enumerate(conn_files, start=1):
        df = load_conn_file(conn_file)
        df["label"] = LABEL_MAP[class_name]
        df["capture_type"] = class_name
        df["source_file"] = str(conn_file.relative_to(ROOT))
        df["capture_id"] = f"{class_name}_{index:03d}"
        frames.append(df)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined conn.log dataset.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "data" / "raw" / "logs",
        help="Root folder containing normal/ and malicious/ directories.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=ROOT / "data" / "interim" / "conn_raw.parquet",
        help="Where to write the combined parquet file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "interim" / "conn_raw.csv",
        help="Where to write the combined CSV file.",
    )
    args = parser.parse_args()

    normal_dir = args.input_root / "normal"
    malicious_dir = args.input_root / "malicious"

    frames: list[pd.DataFrame] = []
    frames.extend(load_class_logs(normal_dir, "normal"))
    frames.extend(load_class_logs(malicious_dir, "malicious"))

    combined = pd.concat(frames, ignore_index=True)

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    combined.to_parquet(args.output_parquet, index=False)
    combined.to_csv(args.output_csv, index=False)

    print(f"Saved combined dataset to: {args.output_parquet}")
    print(f"Saved combined CSV to:     {args.output_csv}")
    print(f"Rows: {len(combined):,}")
    print("Class distribution:")
    print(combined["capture_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
