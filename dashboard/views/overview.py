import plotly.graph_objects as go
import streamlit as st
import theme as t

import data

s = data.summary()
m = s["models"]
b, i, r = m["baseline"], m["improved"], m["robust"]
t.hero("VisionGuard AI", "Classifies steel-surface images as normal or one of four defect types, measures behaviour on rare "
       "classes and under lighting, blur, noise and resolution shifts, and verifies visual explanations against true defect masks.",
       badge="Inspection model ready", icon="🔬")

t.kpis([
    ("Macro-F1 (improved)", f"{i['macro_f1']:.3f}", f"{i['macro_f1'] - b['macro_f1']:+.3f} vs baseline", "🎯", t.PRIMARY),
    ("Defect recall", f"{i['defect_recall']:.1%}", f"{i['missed_defects']} missed defects", "🧲", t.SUCCESS),
    ("False alarms on normal", f"{i['false_alarm_rate']:.1%}", f"baseline {b['false_alarm_rate']:.1%}", "🔔", t.WARNING),
    ("Latency p95", f"{i['latency_ms_p95']:.1f} ms", f"{i['size_kb']:.0f} KB model", "⚡", t.SECONDARY),
    ("Heatmap on defect", f"{s['explanations']['pointing_game']:.0%}", "pointing-game hit rate", "🗺️", t.ACCENT),
])

t.section("Inspection pipeline")
t.flow([("Image", "96×96 grayscale"), ("Normalise", "illumination-invariant"), ("Features", "HOG · LBP · FFT · DoG"),
        ("RBF-SVM", "Platt-calibrated"), ("Occlusion map", "where it looked"), ("Human review", "flag + low confidence")])

c1, c2 = st.columns([1.3, 1])
with c1:
    t.section("Per-class recall", "rare classes matter most")
    fig = go.Figure()
    for name, mm in m.items():
        pc = mm["per_class"]
        fig.add_trace(go.Bar(x=list(pc), y=[v["recall"] for v in pc.values()], name=name, marker_color=t.MODEL_COLORS[name]))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05], yaxis_title="recall")
    st.plotly_chart(t.style_fig(fig, 340), use_container_width=True)
with c2:
    t.section("Robustness at a glance", "noise σ = 0.04")
    rob = data.robustness()
    sub = rob[(rob.corruption == "noise_std") & (rob.level == 0.04)]
    fig = go.Figure(go.Bar(x=sub.model, y=sub.macro_f1, marker_color=[t.MODEL_COLORS[x] for x in sub.model],
                           text=sub.macro_f1.round(2), textposition="outside"))
    fig.update_layout(yaxis_range=[0, 1.05], yaxis_title="macro-F1")
    st.plotly_chart(t.style_fig(fig, 340), use_container_width=True)

t.section("Sample inspections", "test images · improved model")
pred = data.predictions("improved").sample(6, random_state=3)
cols = st.columns(6)
for c, row in zip(cols, pred.itertuples()):
    ok = row.label == row.pred
    c.image(data.image(row.path), use_container_width=True, clamp=True)
    c.markdown(f"<div style='text-align:center;font-size:.8rem'>{t.chip(s['classes'][row.pred], t.CLASS_COLORS[s['classes'][row.pred]])}"
               f"<br><span style='color:{t.SUCCESS if ok else t.DANGER}'>{'✓' if ok else '✗'} true {row.class_name}</span> · "
               f"{row.confidence:.0%}</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    t.card("Leakage-safe split", f"{s['audit']['instances']} steel sheets, split by sheet — "
           f"<b>{s['audit']['instance_leakage']}</b> sheets shared between train and test.")
with c2:
    t.card("Robustness trade-off", f"Corruption-aware training costs {i['macro_f1'] - r['macro_f1']:.3f} clean macro-F1 but "
           "recovers large parts of performance under noise, blur and low resolution.", t.ACCENT)
with c3:
    t.card("Honest errors", "Scratches remain the hardest class; missed defects are inspected visually on the Error Analysis "
           "page rather than hidden behind a single accuracy number.", t.DANGER)
