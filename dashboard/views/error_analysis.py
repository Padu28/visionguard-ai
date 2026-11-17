import streamlit as st
import theme as t

import data
from visionguard.explainability.maps import energy_in_mask, occlusion_map, overlay, pointing_game

t.hero("Error Analysis", "A missed defect is usually costlier than a false alarm. Inspect false negatives and false "
       "positives with heatmaps and ground-truth contours — honest analysis, not cherry-picked wins.",
       badge="FN · FP review", icon="🐞")

s = data.summary()
model = st.segmented_control("Model", ["improved", "robust", "baseline"], default="improved") or "improved"
p = data.predictions(model)
classes = s["classes"]
fn = p[(p.label > 0) & (p.pred == 0)]
fp = p[(p.label == 0) & (p.pred > 0)]
wrong_type = p[(p.label > 0) & (p.pred > 0) & (p.label != p.pred)]

t.kpis([
    ("Missed defects (FN)", f"{len(fn)}", "defect predicted as normal", "🚫", t.DANGER),
    ("False alarms (FP)", f"{len(fp)}", "normal predicted as defect", "🔔", t.WARNING),
    ("Wrong defect type", f"{len(wrong_type)}", "defect found, class confused", "🔀", t.ACCENT),
    ("Low-confidence (<70%)", f"{(p.confidence < 0.7).sum()}", "routed to review", "🧑‍🔧", t.SECONDARY),
])

ex = data.explanations()
if not ex.empty:
    t.section("Explanation quality (improved model)", f"{len(ex)} defect images with masks")
    c1, c2, c3 = st.columns(3)
    with c1:
        t.ring(100 * ex.pointing_hit.mean(), "Pointing-game hit rate", t.PRIMARY)
    with c2:
        t.ring(100 * ex.energy_in_mask.mean(), "Heatmap energy in mask", t.ACCENT)
    with c3:
        t.ring(100 * s["explanations"]["random_baseline_energy"], "Area-matched random", t.MUTED)

tab_fn, tab_fp, tab_wt = st.tabs([f"Missed defects ({len(fn)})", f"False alarms ({len(fp)})", f"Wrong type ({len(wrong_type)})"])
pred = data.predictor(model)


def gallery(df, key: str) -> None:
    if df.empty:
        st.success("No cases in this category 🎉")
        return
    n = st.slider("Cases to show", 1, min(12, len(df)), min(6, len(df)), key=key)
    cols = st.columns(3)
    for k, row in enumerate(df.sort_values("confidence", ascending=False).head(n).itertuples()):
        img, m = data.image(row.path), data.mask(row.path)
        heat, tgt, prob = occlusion_map(pred.model.predict_proba, img)
        with cols[k % 3]:
            st.image(overlay(img, heat), use_container_width=True)
            extra = ""
            if m.sum():
                extra = f" · peak on defect: {'✓' if pointing_game(heat, m) else '✗'} · energy {energy_in_mask(heat, m):.0%}"
            st.markdown(f"<div class='tl-item' style='--sev:{t.DANGER}'><div class='meta'>image {row.image_id} · sheet {row.instance}</div>"
                        f"<div class='msg'>true <b>{row.class_name}</b> → predicted <b>{classes[row.pred]}</b> "
                        f"({row.confidence:.0%}){extra}</div></div>", unsafe_allow_html=True)


with tab_fn:
    gallery(fn, "fn")
with tab_fp:
    gallery(fp, "fp")
with tab_wt:
    gallery(wrong_type, "wt")
