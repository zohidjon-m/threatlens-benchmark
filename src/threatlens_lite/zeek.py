from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


NUMERIC_CANDIDATES = [
    "ts",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "missed_bytes",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "id.resp_p",
]


def discover_conn_files(root: Path) -> list[Path]:
    patterns = ["**/conn.log", "**/conn.log.csv", "**/conn.csv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(files))


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common Zeek missing values
    df = df.replace({"-": pd.NA, "(empty)": pd.NA, "": pd.NA})

    # Convert likely numeric columns
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Make sure essential columns exist even if missing in one class
    essential = [
        "ts",
        "uid",
        "id.orig_h",
        "id.orig_p",
        "id.resp_h",
        "id.resp_p",
        "proto",
        "service",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "conn_state",
        "orig_pkts",
        "resp_pkts",
    ]
    for col in essential:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def load_zeek_json(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return _postprocess(pd.DataFrame(rows))


def load_zeek_ascii(path: Path) -> pd.DataFrame:
    """
    Parse standard Zeek ASCII logs with header lines like:
    #separator \x09
    #fields ts uid id.orig_h ...
    #types  time string addr ...
    """
    separator = "\t"
    fields = None
    data_rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line:
                continue

            if line.startswith("#separator"):
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    sep_token = parts[1].strip()
                    if sep_token == r"\x09":
                        separator = "\t"
                    else:
                        separator = sep_token.encode("utf-8").decode("unicode_escape")
                continue

            if line.startswith("#fields"):
                fields = line.split(separator)
                if fields and fields[0] == "#fields":
                    fields = fields[1:]
                continue

            if line.startswith("#"):
                continue

            if fields is not None:
                values = line.split(separator)
                # Pad or trim to match columns
                if len(values) < len(fields):
                    values += [None] * (len(fields) - len(values))
                elif len(values) > len(fields):
                    values = values[: len(fields)]
                data_rows.append(values)

    if fields is None:
        raise ValueError(f"Could not find #fields header in {path}")

    df = pd.DataFrame(data_rows, columns=fields)
    return _postprocess(df)


def load_csv_like(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if len(df.columns) > 1:
            return _postprocess(df)
    except Exception:
        pass

    try:
        df = pd.read_csv(path, sep="\t")
        if len(df.columns) > 1:
            return _postprocess(df)
    except Exception:
        pass

    raise ValueError(f"Could not parse CSV/TSV-like file: {path}")


def detect_format(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("{"):
                return "json"
            if line.startswith("#separator") or line.startswith("#fields"):
                return "ascii"
            if "," in line:
                return "csv"
            if "\t" in line:
                return "tsv"
            return "unknown"

    return "unknown"


def load_conn_file(path: Path) -> pd.DataFrame:
    fmt = detect_format(path)

    if fmt == "json":
        return load_zeek_json(path)

    if fmt == "ascii":
        return load_zeek_ascii(path)

    if fmt in {"csv", "tsv"}:
        return load_csv_like(path)

    # fallback attempts
    for loader in (load_zeek_ascii, load_zeek_json, load_csv_like):
        try:
            return loader(path)
        except Exception:
            continue

    raise ValueError(f"Unsupported or unreadable conn log format: {path}")