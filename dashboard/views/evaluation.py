import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme as t
from sklearn.metrics import precision_recall_curve

import data

t.hero("Model Evaluation", "Baseline vs improved vs robust on the same held-out sheets: macro-F1 with bootstrap CIs, "
       "per-class precision/recall, confusion matrices, defect PR curves, latency and size.", badge="Held-out test", icon="📊")

s = data.summary()
classes = s["classes"]
mt = data.models_table()

rows = []
for r in mt.itertuples():
    m = r.metrics
    rows.append({"model": r.name, "features / estimator": r.kind, "macro-F1": m["macro_f1"],
                 "95% CI": f"{m['macro_f1_ci95'][0]:.3f}–{m['macro_f1_ci95'][1]:.3f}", "accuracy": m["accuracy"],
                 "defect recall": m["defect_recall"], "false alarms": m["false_alarm_rate"],
                 "missed defects": m["missed_defects"], "p95 ms": r.latency_ms_p95, "size KB": r.size_kb,
                 "#features": r.n_features})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config={
    "macro-F1": st.column_config.ProgressColumn("Macro-F1", min_value=0, max_value=1, format="%.3f"),
    "defect recall": st.column_config.ProgressColumn("Defect recall", min_value=0, max_value=1, format="%.3f"),
    "accuracy": st.column_config.NumberColumn(format="%.3f"), "false alarms": st.column_config.NumberColumn(format="%.3f"),
    "p95 ms": st.column_config.NumberColumn(format="%.1f"), "size KB": st.column_config.NumberColumn(format="%.0f")})

model = st.segmented_control("Inspect model", list(mt.name), default="improved") or "improved"
m = mt[mt.name == model].iloc[0].metrics
c1, c2 = st.columns([1, 1.15])
with c1:
    t.section("Confusion matrix", model)
    cm = np.array(m["confusion"])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=norm, x=classes, y=classes, text=cm, texttemplate="%{text}", zmin=0, zmax=1,
                               colorscale=[[0, t.HEAT_LOW], [1, t.PRIMARY]], colorbar=dict(title="row %")))
    fig.update_yaxes(autorange="reversed", title="true")
    fig.update_xaxes(title="predicted")
    st.plotly_chart(t.style_fig(fig, 380), use_container_width=True)
with c2:
    t.section("Per-class precision & recall")
    pc = pd.DataFrame(m["per_class"]).T.reset_index(names="class")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pc["class"], y=pc.precision, name="precision", marker_color=t.SECONDARY))
    fig.add_trace(go.Bar(x=pc["class"], y=pc.recall, name="recall", marker_color=t.PRIMARY))
    fig.add_trace(go.Scatter(x=pc["class"], y=pc.f1, name="F1", mode="markers+lines", marker=dict(size=11, color=t.INK)))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
    st.plotly_chart(t.style_fig(fig, 380), use_container_width=True)

t.section("Defect vs. normal — precision-recall", "score = 1 − P(normal)")
fig = go.Figure()
for name in mt.name:
    p = data.predictions(name)
    prec, rec, _ = precision_recall_curve((p.label > 0).astype(int), p.defect_score)
    ap = mt[mt.name == name].iloc[0].metrics["defect_pr_auc"]
    fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{name} (AP {ap:.3f})", line=dict(color=t.MODEL_COLORS[name], width=3)))
fig.update_layout(xaxis_title="defect recall", yaxis_title="precision", yaxis_range=[0.4, 1.02])
st.plotly_chart(t.style_fig(fig, 360), use_container_width=True)

t.section("Latency vs. quality", "CPU, single image")
fig = go.Figure()
for r in mt.itertuples():
    fig.add_trace(go.Scatter(x=[r.latency_ms_p95], y=[r.metrics["macro_f1"]], mode="markers+text", text=[r.name],
                             textposition="top center", marker=dict(size=12 + r.size_kb ** 0.5 / 2, color=t.MODEL_COLORS[r.name],
                                                                     line=dict(color=t.INK, width=1)), showlegend=False))
fig.update_layout(xaxis_title="p95 latency (ms) · bubble = model size", yaxis_title="macro-F1")
st.plotly_chart(t.style_fig(fig, 320), use_container_width=True)
