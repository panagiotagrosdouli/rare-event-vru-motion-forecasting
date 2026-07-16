from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Rare-Event VRU Lab", layout="wide")
st.title("Rare-Event VRU Motion Forecasting Lab")
st.caption("Research Prototype — fixture evidence is labelled Synthetic Validation.")

metrics_path = Path("results/fixture_metrics.json")
if metrics_path.exists():
    st.subheader("Synthetic fixture evidence")
    st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
else:
    st.warning("Run `python scripts/run_all.py --mode fixture` first.")

st.subheader("System status")
st.json(
    {
        "Training-only rarity artifact": "Implemented",
        "Multimodal GRU interface": "Research Prototype",
        "Conformal endpoint regions": "Synthetic Validation",
        "Full Argoverse evaluation": "Pending Full-Dataset Validation",
    }
)
