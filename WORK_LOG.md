# Work Log

- Audited existing preprocessing, dataset, label-generation, GRU, tail dataset, and README interfaces.
- Confirmed validation leakage in split-local normalization and percentile fitting.
- Created `feature/publication-grade-rare-event-vru`.
- Implemented training-only class-wise robust rarity statistics with artifact fingerprinting.
- Added hardened coordinate transforms, expanded motion features, kinematic baselines, multimodal GRU, uncertainty, conformal evaluation, deterministic fixtures, tests, CI, Docker, and a Streamlit status lab.
- Local validation in the available environment: `pytest` completed with 7 passing tests; fixture mode executed and generated synthetic metrics and a figure.
- Full Argoverse and GPU validation remain blocked and are not claimed.
