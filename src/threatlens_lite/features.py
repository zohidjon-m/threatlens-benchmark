from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "conn_state",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "label",
    "capture_type",
    "source_file",
    "capture_id",
]

NUMERIC_COLUMNS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "resp_port",
]

CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "conn_state",
]

METADATA_COLUMNS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "id.resp_p",
    "label",
    "capture_type",
    "source_file",
    "capture_id",
    "group_id",
]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def _to_datetime_ts(ts_series: pd.Series) -> pd.Series:
    numeric_ts = pd.to_numeric(ts_series, errors="coerce")
    return pd.to_datetime(numeric_ts, unit="s", utc=True, errors="coerce")


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_columns(df)
    df = df.copy()

    df["ts"] = _to_datetime_ts(df["ts"])
    df = df.dropna(subset=["ts"]).reset_index(drop=True)

    df["duration"] = _safe_numeric(df["duration"])
    df["orig_bytes"] = _safe_numeric(df["orig_bytes"])
    df["resp_bytes"] = _safe_numeric(df["resp_bytes"])
    df["orig_pkts"] = _safe_numeric(df["orig_pkts"])
    df["resp_pkts"] = _safe_numeric(df["resp_pkts"])
    df["resp_port"] = _safe_numeric(df["id.resp_p"])

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("unknown").astype(str)

    df["bytes_total"] = df["orig_bytes"] + df["resp_bytes"]
    df["pkts_total"] = df["orig_pkts"] + df["resp_pkts"]
    df["bytes_ratio"] = df["orig_bytes"] / (df["resp_bytes"] + 1.0)
    df["pkts_ratio"] = df["orig_pkts"] / (df["resp_pkts"] + 1.0)
    df["bytes_per_sec"] = df["bytes_total"] / (df["duration"] + 1e-6)
    df["pkts_per_sec"] = df["pkts_total"] / (df["duration"] + 1e-6)
    df["resp_to_orig"] = df["resp_bytes"] / (df["orig_bytes"] + 1.0)
    df["orig_share_bytes"] = df["orig_bytes"] / (df["bytes_total"] + 1.0)
    df["orig_share_pkts"] = df["orig_pkts"] / (df["pkts_total"] + 1.0)
    df["is_short_flow"] = (df["duration"] < 1.0).astype(int)
    df["is_zero_resp_bytes"] = (df["resp_bytes"] == 0).astype(int)

    df["id.orig_h"] = df["id.orig_h"].fillna("unknown_orig_h").astype(str)
    df["id.resp_h"] = df["id.resp_h"].fillna("unknown_resp_h").astype(str)
    df["capture_id"] = df["capture_id"].fillna(df["capture_type"]).astype(str)
    df["group_id"] = df["capture_id"] + "::" + df["id.orig_h"]
    df["label"] = _safe_numeric(df["label"]).astype(int)

    return df


def _decrement_counter(counter: Counter, key: str | int) -> None:
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]


def _add_temporal_features_to_group(group: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    group = group.sort_values("ts").copy().reset_index(drop=True)

    ts_seconds = group["ts"].astype("int64") / 1e9
    orig_bytes = group["orig_bytes"].to_numpy(dtype=float)
    resp_bytes = group["resp_bytes"].to_numpy(dtype=float)
    orig_pkts = group["orig_pkts"].to_numpy(dtype=float)
    resp_pkts = group["resp_pkts"].to_numpy(dtype=float)
    duration = group["duration"].to_numpy(dtype=float)
    resp_hosts = group["id.resp_h"].astype(str).to_list()
    resp_ports = group["id.resp_p"].fillna("unknown_port").astype(str).to_list()
    services = group["service"].fillna("unknown").astype(str).to_list()

    n = len(group)
    flows_count = np.zeros(n, dtype=float)
    bytes_sent = np.zeros(n, dtype=float)
    bytes_recv = np.zeros(n, dtype=float)
    pkts_sent = np.zeros(n, dtype=float)
    pkts_recv = np.zeros(n, dtype=float)
    avg_duration = np.zeros(n, dtype=float)
    unique_dst_ips = np.zeros(n, dtype=float)
    unique_dst_ports = np.zeros(n, dtype=float)
    unique_services = np.zeros(n, dtype=float)

    left = 0
    sum_orig_bytes = 0.0
    sum_resp_bytes = 0.0
    sum_orig_pkts = 0.0
    sum_resp_pkts = 0.0
    sum_duration = 0.0

    host_counter: Counter = Counter()
    port_counter: Counter = Counter()
    service_counter: Counter = Counter()

    for right in range(n):
        sum_orig_bytes += orig_bytes[right]
        sum_resp_bytes += resp_bytes[right]
        sum_orig_pkts += orig_pkts[right]
        sum_resp_pkts += resp_pkts[right]
        sum_duration += duration[right]

        host_counter[resp_hosts[right]] += 1
        port_counter[resp_ports[right]] += 1
        service_counter[services[right]] += 1

        current_ts = ts_seconds.iloc[right]
        while current_ts - ts_seconds.iloc[left] > window_seconds:
            sum_orig_bytes -= orig_bytes[left]
            sum_resp_bytes -= resp_bytes[left]
            sum_orig_pkts -= orig_pkts[left]
            sum_resp_pkts -= resp_pkts[left]
            sum_duration -= duration[left]

            _decrement_counter(host_counter, resp_hosts[left])
            _decrement_counter(port_counter, resp_ports[left])
            _decrement_counter(service_counter, services[left])
            left += 1

        window_size = right - left + 1
        flows_count[right] = window_size
        bytes_sent[right] = sum_orig_bytes
        bytes_recv[right] = sum_resp_bytes
        pkts_sent[right] = sum_orig_pkts
        pkts_recv[right] = sum_resp_pkts
        avg_duration[right] = sum_duration / max(window_size, 1)
        unique_dst_ips[right] = len(host_counter)
        unique_dst_ports[right] = len(port_counter)
        unique_services[right] = len(service_counter)

    group[f"flows_last_{window_seconds}s"] = flows_count
    group[f"bytes_sent_last_{window_seconds}s"] = bytes_sent
    group[f"bytes_recv_last_{window_seconds}s"] = bytes_recv
    group[f"pkts_sent_last_{window_seconds}s"] = pkts_sent
    group[f"pkts_recv_last_{window_seconds}s"] = pkts_recv
    group[f"avg_duration_last_{window_seconds}s"] = avg_duration
    group[f"unique_dst_ips_last_{window_seconds}s"] = unique_dst_ips
    group[f"unique_dst_ports_last_{window_seconds}s"] = unique_dst_ports
    group[f"unique_services_last_{window_seconds}s"] = unique_services
    group[f"flow_rate_last_{window_seconds}s"] = flows_count / float(window_seconds)
    group[f"avg_total_bytes_per_flow_last_{window_seconds}s"] = (
        (bytes_sent + bytes_recv) / np.maximum(flows_count, 1.0)
    )

    return group


def add_temporal_features(df: pd.DataFrame, window_seconds: int = 30) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["capture_id", "id.orig_h", "ts"]).reset_index(drop=True)
    enriched_groups = []
    for _, group in df.groupby(["capture_id", "id.orig_h"], dropna=False, sort=False):
        enriched_groups.append(_add_temporal_features_to_group(group, window_seconds=window_seconds))
    result = pd.concat(enriched_groups, ignore_index=True)
    return result


def build_feature_table(raw_df: pd.DataFrame, window_seconds: int = 30) -> pd.DataFrame:
    base_df = add_base_features(raw_df)
    feature_df = add_temporal_features(base_df, window_seconds=window_seconds)
    return feature_df
