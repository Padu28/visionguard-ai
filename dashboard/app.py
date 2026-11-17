"""VisionGuard AI inspection dashboard.

Run:  streamlit run dashboard/app.py --server.port 8503
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="VisionGuard AI · Defect Inspection", page_icon="🔬", layout="wide",
                   initial_sidebar_state="collapsed")

import theme  # noqa: E402

import data  # noqa: E402

theme.apply_theme()
data.ensure_pipeline()

pages = {
    "Inspection": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("views/inspect.py", title="Inspect Image", icon=":material/center_focus_strong:", url_path="inspect"),
        st.Page("views/error_analysis.py", title="Error Analysis", icon=":material/bug_report:", url_path="error_analysis"),
    ],
    "Research": [
        st.Page("views/evaluation.py", title="Model Evaluation", icon=":material/leaderboard:", url_path="evaluation"),
        st.Page("views/robustness.py", title="Robustness Study", icon=":material/blur_on:", url_path="robustness"),
        st.Page("views/dataset_audit.py", title="Dataset Audit", icon=":material/photo_library:", url_path="dataset_audit"),
    ],
}

nav = st.navigation(pages, position="top")

# ---- production-line status strip (replaces the sidebar)
s = data.summary()
m = s.get("models", {})
c1, c2 = st.columns([7, 1])
with c1:
    theme.strip([
        ("line", "ONLINE", theme.SUCCESS),
        ("images", f"{s.get('audit', {}).get('n_images', 0):,}", theme.SECONDARY),
        ("models", " · ".join(m) or "—", theme.PRIMARY),
        ("macro-F1", f"{m.get('improved', {}).get('macro_f1', 0):.3f}", theme.PRIMARY),
        ("review", "human required", theme.ACCENT),
    ])
with c2:
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

nav.run()
st.markdown('<div class="disclaimer"><b>Human review required.</b> Decision support for visual inspection only. '
            'Synthetic training data — not validated on production imagery.</div>', unsafe_allow_html=True)
theme.footer()
