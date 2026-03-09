# ThreatLens-lite Results

## Problem

ThreatLens-lite predicts whether a network flow is benign or malicious using Zeek `conn.log` telemetry, feature engineering, temporal aggregation, and baseline machine-learning models.

## Dataset Summary

- Total rows: 18,171
- Malicious rows: 11,655
- Normal rows: 6,516
- Unique capture IDs: 2
- Unique host groups: 2,363

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.9917 | 1.0000 | 0.9898 | 0.9949 | 1.0000 |
| random_forest | 0.8714 | 1.0000 | 0.8415 | 0.9139 | 0.9996 |
| lightgbm | 0.9068 | 1.0000 | 0.8851 | 0.9391 | 0.9995 |

## Top Features (Best Model)

Best model: **logistic_regression**

| Feature | Importance |
|---|---:|
| num__local_resp | 6.255645 |
| num__ip_proto | 1.054626 |
| num__pkts_ratio | 0.736219 |
| num__unique_services_last_30s | 0.634934 |
| num__resp_to_orig | 0.468270 |
| num__avg_duration_last_30s | 0.448338 |
| cat__history_ShAFar | 0.424633 |
| num__duration | 0.400661 |
| num__unique_dst_ips_last_30s | 0.387486 |
| num__resp_pkts | 0.376625 |

## Notes

- Temporal features summarize host behavior over a rolling 30-second window.
- Labels are assigned from the source folder (`normal` vs `malicious`).
- Results should be interpreted carefully if benign and malicious captures come from very different environments.