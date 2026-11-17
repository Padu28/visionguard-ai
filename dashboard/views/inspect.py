import numpy as np
import plotly.graph_objects as go
import streamlit as st
import theme as t

import data
from visionguard.explainability.maps import overlay
from visionguard.inference.predictor import HUMAN_REVIEW
from visionguard.transforms.augment import CORRUPTIONS

t.hero("Inspect Image", "Upload a surface image or pick a test sample. See the predicted class, calibrated probabilities, "
       "an occlusion heatmap and how the prediction changes under controlled corruptions.", badge="Live inference", icon="🔍")

meta = data.metadata()
c1, c2 = st.columns([1, 1.3])
with c1:
    src = st.radio("Source", ["Test sample", "Upload"], horizontal=True)
    model_name = st.segmented_control("Model", ["improved", "robust", "baseline"], default="improved") or "improved"
    raw, true_cls, mask = None, None, None
    if src == "Upload":
        up = st.file_uploader("PNG / JPG (converted to 96×96 grayscale)", type=["png", "jpg", "jpeg"])
        if up:
            raw = up.read()
    else:
        cls = st.selectbox("Class", ["any"] + data.settings.classes)
        pool = meta[(meta.split == "test") & ((meta.class_name == cls) if cls != "any" else True)]
        k = st.slider("Sample", 0, len(pool) - 1, 0)
        row = pool.iloc[k]
        raw, true_cls, mask = data.image(row.path), row.class_name, data.mask(row.path)

    st.markdown("**Corruption (what-if)**")
    kind = st.selectbox("Type", ["none", "brightness", "blur_sigma", "noise_std", "resolution_scale"])
    level = 0.0
    if kind == "brightness":
        level = st.slider("Δ brightness", -0.4, 0.4, 0.0, 0.05)
    elif kind == "blur_sigma":
        level = st.slider("Blur σ", 0.0, 3.0, 0.0, 0.25)
    elif kind == "noise_std":
        level = st.slider("Noise σ", 0.0, 0.12, 0.0, 0.01)
    elif kind == "resolution_scale":
        level = st.slider("Resolution scale", 0.25, 1.0, 1.0, 0.05)
    explain = st.toggle("Compute occlusion heatmap", True)

if raw is None:
    st.info("Upload an image or choose a sample to start.")
    st.stop()

pred = data.predictor(model_name)
img = pred.preprocess(raw)
if kind != "none":
    img = CORRUPTIONS[kind](img, level)
res = pred.predict(img.astype(np.float32), explain=explain)
data.db().log_inference("dashboard", model_name, res.label, res.confidence, res.defect_score, res.latency_ms)

with c2:
    a, b = st.columns(2)
    a.image(img, caption="Input (as seen by the model)", use_container_width=True, clamp=True)
    if res.heatmap is not None:
        b.image(overlay(img, res.heatmap), caption="Occlusion heatmap", use_container_width=True)
    color = t.SUCCESS if res.label == "normal" else t.DANGER
    verdict = "Matches ground truth ✓" if true_cls and true_cls == res.label else (f"Ground truth: {true_cls}" if true_cls else "")
    st.markdown(f"""<div class="card" style="border-top:3px solid {color};text-align:center">
        <div style="color:{t.MUTED};font-size:.75rem;font-weight:700;letter-spacing:.08em">PREDICTION · {model_name.upper()}</div>
        <div style="font-size:2.2rem;font-weight:800;color:{color}">{res.label.upper()}</div>
        <div style="color:{t.MUTED}">confidence {res.confidence:.1%} · defect score {res.defect_score:.2f} · {res.latency_ms:.1f} ms</div>
        <div style="margin-top:6px">{verdict}</div></div>""", unsafe_allow_html=True)
    if res.needs_review:
        st.warning(f"⚠️ Route to human inspector — {HUMAN_REVIEW}")

p = res.probabilities
fig = go.Figure(go.Bar(x=list(p.values()), y=list(p.keys()), orientation="h",
                       marker_color=[t.CLASS_COLORS[k] for k in p], text=[f"{v:.1%}" for v in p.values()], textposition="outside"))
fig.update_layout(xaxis_range=[0, 1.1], yaxis=dict(autorange="reversed"))
st.plotly_chart(t.style_fig(fig, 260, "Calibrated class probabilities"), use_container_width=True)

if mask is not None and mask.sum() and res.heatmap is not None:
    from visionguard.explainability.maps import energy_in_mask, pointing_game
    t.kpis([("Pointing game", "hit ✓" if pointing_game(res.heatmap, mask) else "miss ✗", "heatmap peak on defect?", "🎯", t.PRIMARY),
            ("Energy in mask", f"{energy_in_mask(res.heatmap, mask):.0%}", f"mask area {mask.mean():.1%}", "🗺️", t.ACCENT)])
