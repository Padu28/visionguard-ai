import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme as t
from plotly.subplots import make_subplots

import data
from visionguard.transforms.augment import CORRUPTIONS

t.hero("Robustness Study", "Controlled brightness, blur, noise and resolution shifts applied to the test set. The robust "
       "model is trained with milder corruptions than the strongest test levels, so the tail measures extrapolation.",
       badge="Distribution shift", icon="🌫️")

rob = data.robustness()
metric = st.segmented_control("Metric", ["macro_f1", "defect_recall", "false_alarm_rate", "accuracy"], default="macro_f1") or "macro_f1"
titles = {"brightness": "Brightness Δ", "blur_sigma": "Gaussian blur σ", "noise_std": "Gaussian noise σ", "resolution_scale": "Resolution scale"}
fig = make_subplots(rows=1, cols=4, subplot_titles=[titles[k] for k in titles], horizontal_spacing=0.05)
for j, kind in enumerate(titles, start=1):
    g = rob[rob.corruption == kind]
    for mname, gg in g.groupby("model"):
        fig.add_trace(go.Scatter(x=gg.level, y=gg[metric], name=mname, mode="lines+markers", legendgroup=mname,
                                 showlegend=j == 1, line=dict(color=t.MODEL_COLORS[mname], width=3, shape="spline"),
                                 marker=dict(size=8)), 1, j)
    if kind == "resolution_scale":
        fig.update_xaxes(autorange="reversed", row=1, col=j)
fig.update_yaxes(range=[0, 1.05])
st.plotly_chart(t.style_fig(fig, 380), use_container_width=True)

t.section("Area under the degradation curve", "higher = more robust (normalised by clean score)")
rows = []
for (kind, mname), g in rob.groupby(["corruption", "model"]):
    g = g.sort_values("level")
    clean = g.iloc[(g.level - (1.0 if kind == "resolution_scale" else 0.0)).abs().argmin()]["macro_f1"]
    rows.append({"corruption": kind, "model": mname, "relative_robustness": float(g.macro_f1.mean() / max(clean, 1e-9))})
rr = pd.DataFrame(rows)
fig = go.Figure()
for mname, g in rr.groupby("model"):
    fig.add_trace(go.Bar(x=g.corruption, y=g.relative_robustness, name=mname, marker_color=t.MODEL_COLORS[mname],
                         text=g.relative_robustness.round(2), textposition="outside"))
fig.update_layout(barmode="group", yaxis_range=[0, 1.15])
st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)

t.section("See the corruption", "same image at increasing severity")
meta = data.metadata()
row = meta[(meta.split == "test") & (meta.class_name == "crack")].iloc[0]
img = data.image(row.path)
kind = st.selectbox("Corruption", list(titles), format_func=titles.get)
levels = sorted(rob[rob.corruption == kind].level.unique(), reverse=kind == "resolution_scale")
cols = st.columns(len(levels))
pred = data.predictor("improved")
for c, lv in zip(cols, levels):
    x = CORRUPTIONS[kind](img, lv)
    r = pred.predict(x)
    c.image(x, use_container_width=True, clamp=True)
    c.markdown(f"<div style='text-align:center;font-size:.78rem'>{lv:g}<br>{t.chip(r.label, t.CLASS_COLORS[r.label])}</div>",
               unsafe_allow_html=True)
