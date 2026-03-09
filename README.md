# ThreatLens-Benchmark

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20benchmark-orange)
![Zeek](https://img.shields.io/badge/data-Zeek%20conn.log-green)

**A reproducible Zeek-to-feature pipeline and baseline benchmark for benign vs malicious-labeled network traffic.**

ThreatLens-Benchmark is a lightweight security ML project that transforms Zeek `conn.log` telemetry into tabular flow features, adds temporal behavior aggregation, and benchmarks baseline classifiers for benign-versus-malicious labeled traffic.

The goal of this repository is not to claim a production-ready malware detector. The goal is to provide a **reproducible research-style pipeline** for:

- converting Zeek network telemetry into ML-ready features
- experimenting with flow-level and temporal features
- benchmarking baseline models
- documenting evaluation limitations clearly

---

## Why this project exists

Raw PCAP files are difficult to use directly in machine-learning workflows. Security researchers and engineers often repeat the same pipeline:

**PCAP → Zeek logs → cleaning → feature engineering → labels → baseline models**

ThreatLens-Benchmark packages that workflow into a simple, reproducible benchmark so that experiments can start from structured Zeek telemetry instead of raw packet traces.

---

## What the project does

- parses Zeek `conn.log` files from **normal** and **malicious** traffic folders
- builds a unified tabular dataset
- engineers flow-level features such as bytes, packets, durations, and ratios
- adds **temporal aggregation features** over rolling time windows
- trains baseline classifiers:
  - Logistic Regression
  - Random Forest
  - LightGBM
- generates:
  - metrics JSON
  - confusion matrices
  - feature-importance plots
  - a markdown results report

---

## Repository structure

```text
threatlens-benchmark/
├── data/
│   ├── raw/
│   │   └── logs/
│   │       ├── malicious/
│   │       │   └── conn.log
│   │       └── normal/
│   │           └── conn.log
│   ├── interim/
│   └── processed/
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── threatlens_lite_results.md
├── scripts/
│   ├── 01_build_dataset.py
│   ├── 02_build_features.py
│   ├── 03_train_baselines.py
│   └── 04_make_report.py
├── src/
│   └── threatlens_lite/
│       ├── __init__.py
│       ├── features.py
│       ├── modeling.py
│       └── zeek.py
├── README.md
└── requirements.txt
```

---

## Expected input layout

Place Zeek connection logs under:

```text
data/raw/logs/
├── malicious/
│   └── conn.log
└── normal/
    └── conn.log
```

The parser supports common Zeek log formats, including:

- standard Zeek ASCII logs
- JSON Zeek logs
- CSV/TSV-like conn exports

---

## Quickstart

### 1. Create environment and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the combined raw dataset

```bash
python scripts/01_build_dataset.py
```

This creates:

- `data/interim/conn_raw.parquet`
- `data/interim/conn_raw.csv`

### 3. Build engineered features

```bash
python scripts/02_build_features.py --window-seconds 30
```

This creates:

- `data/processed/features_v1.parquet`

### 4. Train baseline models

```bash
python scripts/03_train_baselines.py
```

This creates:

- `reports/metrics.json`
- confusion matrix plots
- feature-importance outputs

### 5. Generate the results report

```bash
python scripts/04_make_report.py
```

This creates:

- `reports/threatlens_lite_results.md`

---

## Method

### 1. Zeek log ingestion
The pipeline loads `conn.log` files from benign and malicious-labeled capture folders.

### 2. Flow-level features
Core features include:

- `duration`
- `orig_bytes`, `resp_bytes`
- `orig_pkts`, `resp_pkts`
- `proto`
- `service`
- `conn_state`

Derived features include:

- total bytes / packets
- byte and packet ratios
- bytes-per-second
- packets-per-second
- response-to-origin ratio

### 3. Temporal aggregation
To capture host behavior over time, the pipeline computes rolling-window features such as:

- number of flows in the last 30 seconds
- bytes sent/received in the last 30 seconds
- packets sent/received in the last 30 seconds
- average duration in the last 30 seconds
- number of unique destination IPs in the last 30 seconds
- number of unique destination ports in the last 30 seconds
- number of unique services in the last 30 seconds

### 4. Baseline models
The benchmark currently includes:

- Logistic Regression
- Random Forest
- LightGBM

### 5. Evaluation
Metrics include:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- confusion matrices

---

## Current results

Initial experiments on the current dataset produced the following baseline results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9917 | 1.0000 | 0.9898 | 0.9949 | 1.0000 |
| Random Forest | 0.8714 | 1.0000 | 0.8415 | 0.9139 | 0.9996 |
| LightGBM | 0.9068 | 1.0000 | 0.8851 | 0.9391 | 0.9995 |

### Important interpretation note

These results show **strong separability on the current dataset**, but they should be interpreted carefully.

The current benchmark version uses:

- **18,171 total rows**
- **2 capture IDs**
- labels assigned at the **capture/folder level**

Because of this, the strong results may reflect a mixture of:

- attack-related behavioral differences
- environment-specific artifacts
- capture-level bias
- coarse label granularity

This repository therefore presents the results as **initial benchmark results**, not as evidence of fully generalizable malware detection.

---

## Example output files

After a full run, the main outputs are:

```text
data/interim/conn_raw.parquet
data/processed/features_v1.parquet
reports/metrics.json
reports/threatlens_lite_results.md
reports/figures/confusion_logistic_regression.png
reports/figures/confusion_random_forest.png
reports/figures/confusion_lightgbm.png
```

---

## Limitations

This project is intentionally lightweight, and the current version has important limitations:

1. **Capture-level labels**  
   The current labels reflect whether a flow came from a benign or malicious-labeled capture, not guaranteed per-flow ground truth.

2. **Small number of captures**  
   The current dataset contains only two capture IDs, which increases the risk of capture-level leakage or environment bias.

3. **Initial benchmark scope**  
   The current benchmark focuses on `conn.log` features. It does not yet integrate richer telemetry from:
   - `dns.log`
   - `http.log`
   - `files.log`
   - `ssl.log`

4. **Not a production detector**  
   This repository is a research-style benchmark and pipeline, not a deployed detection system.

---

## Roadmap

Planned improvements include:

- multi-capture evaluation across several benign and malicious sources
- stricter leakage controls and ablation studies
- integration of `dns.log`, `http.log`, and `files.log`
- richer temporal and graph-based features
- anomaly detection baselines
- more robust benchmark splits by capture source
- explainability analysis with SHAP

---

## Use cases

ThreatLens-Benchmark may be useful for:

- students learning security ML pipelines
- researchers experimenting with Zeek-based features
- engineers prototyping traffic classification baselines
- reproducible benchmarking for flow-based network telemetry

---

## Suggested citation

If you use this project, cite it as:

```bibtex
@software{mahmudjonov2026threatlensbenchmark,
  author = {Mahmudjonov, Zohidjon},
  title = {ThreatLens-Benchmark: A Reproducible Zeek-to-Feature Pipeline and Baseline Benchmark for Benign vs Malicious-Labeled Traffic},
  year = {2026}
}
```

---

## Author

**Zohidjon Mahmudjonov**  
Computer Science student, Sejong University  
Focus areas: AI infrastructure, security ML, multi-agent systems, open-source tooling

---

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.
