# Repository Audit

Status: **Research Prototype**

## Verified existing functionality

- `src/preprocess.py` performs 50-step/60-step agent-centric preprocessing.
- `src/dataset.py` filters pedestrian, cyclist, and motorcyclist focal agents from Argoverse parquet files.
- `models/gru.py` provides the deterministic GRU baseline.
- `src/tail_dataset.py` reads tail metadata and native scenarios.
- `scripts/create_classwise_tail_labels.py` computes class-wise labels.

## Critical finding

The existing label script normalizes features and fits the percentile independently for every requested split. Running it on validation therefore uses validation-distribution information to define validation labels. This branch adds a training-only immutable statistics artifact and tests that applying it does not refit or mutate thresholds.

## Implemented in this branch

- installable `rare_event_vru` package;
- stable local/global coordinate transforms with stationary handling;
- expanded motion features;
- training-only class-wise robust rarity statistics and fingerprint;
- constant-position, constant-velocity, and constant-acceleration baselines;
- genuine K-mode GRU with normalized probabilities and heteroscedastic scales;
- deterministic/multimodal metrics;
- ensemble uncertainty decomposition;
- split-conformal endpoint regions;
- deterministic dataset-absent fixture pipeline;
- focused tests, public CI, Docker, and a minimal Streamlit status lab.

## Evidence generated in the available execution environment

- `pytest`: **7 passed** locally before publishing the branch.
- `python scripts/run_all.py --mode fixture`: completed and generated `artifacts/tail_statistics.json`, fixture metrics, labels, and one figure.
- Evidence label: **Synthetic Validation**.

## Not reproduced

Historical Argoverse metrics and checkpoints were not recreated because the full dataset and original local outputs are unavailable in this execution environment. They must remain labelled Historical / Previously Reported until rerun.

## Repository-size warning

The repository reports roughly 79 MB. A complete tracked-file inventory is still required to verify that no generated caches, checkpoints, or accidental binaries are committed.
