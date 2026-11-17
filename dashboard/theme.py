"""VisionGuard AI design system — "Inspection Lab" (light).

Clean industrial look: blueprint grid, soft raised panels, measurement-ruler
details, and an animated hero inspection station — a steel plate (SVG
turbulence texture) scanned by a laser that locks a bounding box onto a defect
while a conveyor belt moves underneath.
"""
from __future__ import annotations

import html
from collections.abc import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

BRAND = "VisionGuard AI"
TAGLINE = "Inspection lab · robust visual QA"
PRIMARY = "#059669"     # emerald
SECONDARY = "#0891b2"   # teal
ACCENT = "#d97706"      # amber
SUCCESS, WARNING, DANGER, MUTED = "#059669", "#d97706", "#dc2626", "#5b6b65"
BG, SURFACE, BORDER, TEXT = "#f3f5f4", "#ffffff", "#dde4e0", "#0f1f1a"
INK = TEXT
HEAT_LOW = "#ecfdf5"
COLORWAY = [PRIMARY, SECONDARY, ACCENT, "#7c3aed", DANGER, "#0f766e", "#65a30d", "#475569"]
MODEL_COLORS = {"baseline": "#94a3b8", "improved": PRIMARY, "robust": ACCENT}
CLASS_COLORS = {"normal": "#64748b", "scratch": "#0284c7", "inclusion": "#7c3aed", "crack": "#dc2626", "patch": "#d97706"}
SEVERITY_COLORS = {"critical": DANGER, "high": DANGER, "medium": WARNING, "low": SECONDARY, "info": PRIMARY}

pio.templates["visionguard"] = go.layout.Template(layout=dict(
    font=dict(family="Manrope, system-ui, sans-serif", color=TEXT, size=12.5),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", colorway=COLORWAY,
    xaxis=dict(gridcolor="#eef2f0", zeroline=False, linecolor=BORDER, ticks="outside", tickcolor=BORDER),
    yaxis=dict(gridcolor="#eef2f0", zeroline=False, linecolor=BORDER, ticks="outside", tickcolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0),
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(bgcolor="#0f1f1a", bordercolor=PRIMARY, font=dict(family="IBM Plex Mono", color="#fff")),
    transition=dict(duration=500, easing="cubic-in-out"),
))
pio.templates.default = "plotly_white+visionguard"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
@property --fill { syntax: '<percentage>'; inherits: false; initial-value: 0%; }
@property --ring { syntax: '<number>'; inherits: false; initial-value: 0; }
html, body, [class*="css"], .stMarkdown { font-family: 'Manrope', system-ui, sans-serif; }
.mono, code { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }
.stApp { color: __TEXT__; background:
  linear-gradient(#0596690b 1px, transparent 1px) 0 0 / 28px 28px,
  linear-gradient(90deg, #0596690b 1px, transparent 1px) 0 0 / 28px 28px,
  linear-gradient(#0596690f 1px, transparent 1px) 0 0 / 140px 140px,
  linear-gradient(90deg, #0596690f 1px, transparent 1px) 0 0 / 140px 140px, __BG__; }
[data-testid="stHeader"] { background: #f3f5f4e6; backdrop-filter: blur(8px); border-bottom: 1px solid __BORDER__; }
.block-container { padding-top: 4.4rem; max-width: 1400px; }

@keyframes up { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes fillBar { from { --fill: 0%; } }
@keyframes ringFill { from { --ring: 0; } }
@keyframes belt { to { background-position: 40px 0; } }
@keyframes led { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.element-container, .stPlotlyChart { animation: up .6s cubic-bezier(.22,1,.36,1) both; }
.stPlotlyChart { background: #fff; border-radius: 14px; border: 1px solid __BORDER__; padding: 6px; box-shadow: 0 1px 2px #0f1f1a0a, 0 8px 24px -16px #0f1f1a33; }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 10px 0 22px; }
.kpi { position: relative; background: #fff; border: 1px solid __BORDER__; border-radius: 14px; padding: 16px 18px 20px;
  box-shadow: 0 1px 2px #0f1f1a0d, 0 10px 30px -18px #0f1f1a40; animation: up .6s cubic-bezier(.22,1,.36,1) both;
  transition: transform .25s, box-shadow .25s; overflow: hidden; }
.kpi:hover { transform: translateY(-4px); box-shadow: 0 2px 4px #0f1f1a10, 0 22px 40px -20px color-mix(in srgb, var(--c1) 60%, transparent); }
.kpi::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 7px;
  background: repeating-linear-gradient(90deg, #c9d3ce 0 1px, transparent 1px 8px); opacity: .9; }
.kpi::after { content: ""; position: absolute; left: 0; bottom: 0; height: 3px; width: var(--fill); --fill: 100%;
  background: linear-gradient(90deg, var(--c1), color-mix(in srgb, var(--c1) 40%, white)); animation: fillBar 1.4s cubic-bezier(.22,1,.36,1) both; }
.kpi .label { margin-top: 6px; color: __MUTED__; font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
.kpi .value { font-size: 1.95rem; font-weight: 800; letter-spacing: -.03em; color: __TEXT__; margin-top: 2px; }
.kpi .delta { font: 500 .76rem 'IBM Plex Mono', ui-monospace, monospace; }
.kpi .icon { position: absolute; right: 14px; top: 16px; width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  background: color-mix(in srgb, var(--c1) 12%, white); font-size: 1rem; }

.section { display: flex; align-items: center; gap: 10px; margin: 30px 0 12px; animation: up .5s ease both; }
.section .tag { font: 600 .68rem 'IBM Plex Mono', ui-monospace, monospace; color: #fff; background: __TEXT__; padding: 3px 8px; border-radius: 5px; }
.section h3 { margin: 0; font-size: 1.15rem; font-weight: 800; letter-spacing: -.02em; color: __TEXT__; }
.section .sub { color: __MUTED__; font-size: .84rem; }

.card { background: #fff; border: 1px solid __BORDER__; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;
  box-shadow: 0 1px 2px #0f1f1a0d, 0 10px 30px -20px #0f1f1a40; animation: up .6s ease both; transition: transform .25s; }
.card:hover { transform: translateY(-3px); }
.card h4 { margin: 0 0 6px; font-size: 1rem; font-weight: 800; color: __TEXT__; }
.card p { margin: 0; color: __MUTED__; font-size: .9rem; line-height: 1.55; }

.tl-item { background: #fff; border: 1px solid __BORDER__; border-left: 4px solid var(--sev); border-radius: 10px; padding: 10px 14px;
  margin-bottom: 8px; animation: up .45s ease both; }
.tl-item .meta { font: .74rem 'IBM Plex Mono', ui-monospace, monospace; color: __MUTED__; }
.tl-item .msg { font-size: .9rem; color: __TEXT__; margin-top: 3px; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 700;
  color: var(--c); background: color-mix(in srgb, var(--c) 12%, white); border: 1px solid color-mix(in srgb, var(--c) 30%, white); }

.ring { width: 140px; height: 140px; margin: auto; border-radius: 50%; display: grid; place-items: center; --ring: var(--val);
  background: conic-gradient(var(--rc) calc(var(--ring) * 1%), #e8eeeb 0); animation: ringFill 1.6s cubic-bezier(.22,1,.36,1) both;
  box-shadow: 0 10px 30px -14px color-mix(in srgb, var(--rc) 70%, transparent); }
.ring > div { width: 112px; height: 112px; border-radius: 50%; background: #fff; display: grid; place-items: center;
  font: 800 1.7rem 'Manrope', system-ui, sans-serif; color: __TEXT__; box-shadow: inset 0 2px 6px #0f1f1a14; }

.flow { display: flex; align-items: stretch; flex-wrap: wrap; gap: 0; margin: 8px 0 14px; }
.flow .node { background: #fff; border: 1px solid __BORDER__; border-radius: 12px; padding: 10px 14px; font-weight: 800; font-size: .86rem;
  color: __TEXT__; animation: up .5s ease both; box-shadow: 0 6px 18px -14px #0f1f1a55; }
.flow .node small { display: block; font: 500 .7rem 'IBM Plex Mono', ui-monospace, monospace; color: __MUTED__; }
.flow .belt { width: 46px; align-self: center; height: 6px; border-radius: 3px;
  background: repeating-linear-gradient(90deg, __PRIMARY__ 0 10px, #a7f3d0 10px 20px); background-size: 40px 6px; animation: belt .7s linear infinite; }

.strip { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; background: #fff; border: 1px solid __BORDER__; border-radius: 12px;
  padding: 9px 16px; font: 500 .78rem 'IBM Plex Mono', ui-monospace, monospace; color: __MUTED__; box-shadow: 0 6px 18px -16px #0f1f1a66; }
.strip b { color: __TEXT__; }
.strip .led { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: var(--c); box-shadow: 0 0 0 3px color-mix(in srgb, var(--c) 20%, white); animation: led 2s infinite; }

.stButton > button, .stDownloadButton > button { border-radius: 10px; background: __TEXT__; color: #fff; border: 0; font-weight: 700; transition: transform .15s, box-shadow .2s; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 20px -10px #0f1f1a; color: #fff; background: __PRIMARY__; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid __BORDER__; }
.stTabs [data-baseweb="tab"] { font-weight: 700; }
.disclaimer { font-size: .8rem; color: __MUTED__; background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 10px 14px; margin-top: 18px; }
.footer { text-align: center; color: __MUTED__; font: .74rem 'IBM Plex Mono', ui-monospace, monospace; margin-top: 40px; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; } }
</style>
"""
_TOK = {"__BG__": BG, "__BORDER__": BORDER, "__TEXT__": TEXT, "__MUTED__": MUTED, "__PRIMARY__": PRIMARY}


def _fill(s: str) -> str:
    for k, v in _TOK.items():
        s = s.replace(k, v)
    return s


def _e(s: object) -> str:
    return html.escape(str(s))


def apply_theme() -> None:
    st.markdown(_fill(_CSS), unsafe_allow_html=True)


_HERO = """<!DOCTYPE html><html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;800&family=IBM+Plex+Mono:wght@500&display=swap');
html,body{margin:0;background:transparent;font-family:Manrope,system-ui,sans-serif;color:#0f1f1a;overflow:hidden}
.w{position:relative;height:__H__px;border-radius:20px;background:#fff;border:1px solid #dde4e0;overflow:hidden;
 box-shadow:0 1px 2px #0f1f1a0d,0 24px 50px -30px #0f1f1a66;display:grid;grid-template-columns:1.25fr 1fr}
.l{padding:28px 32px;position:relative}
.kicker{font:500 11px 'IBM Plex Mono',monospace;color:#059669;letter-spacing:.14em;text-transform:uppercase;display:flex;align-items:center;gap:8px}
.kicker i{width:8px;height:8px;border-radius:50%;background:#059669;box-shadow:0 0 0 4px #05966922;animation:p 1.6s infinite}
@keyframes p{50%{box-shadow:0 0 0 8px #05966900}}
h1{font-size:42px;font-weight:800;letter-spacing:-.04em;margin:12px 0 8px;position:relative;display:inline-block}
h1::after{content:"";position:absolute;inset:0;background:#fff;transform-origin:right;animation:wipe 1.1s cubic-bezier(.77,0,.18,1) .15s forwards}
h1 span{position:relative}
@keyframes wipe{to{transform:scaleX(0)}}
h1 em{font-style:normal;color:#059669}
p{margin:0;color:#5b6b65;font-size:14.5px;line-height:1.6;max-width:560px;opacity:0;animation:f .8s ease .7s forwards}
@keyframes f{to{opacity:1}}
.pills{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap;opacity:0;animation:f .8s ease 1s forwards}
.pills span{font:500 11px 'IBM Plex Mono',monospace;border:1px solid #dde4e0;border-radius:999px;padding:4px 10px;color:#0f1f1a;background:#f8faf9}
.r{position:relative;background:#0f1f1a;overflow:hidden}
.stage{position:absolute;left:50%;top:45%;transform:translate(-50%,-50%);width:min(66%,270px);aspect-ratio:3/2}
.stage svg{width:100%;height:100%;display:block;border-radius:10px;box-shadow:0 20px 40px -10px #000c}
.laser{position:absolute;left:-4%;right:-4%;height:2px;background:#22d3ee;box-shadow:0 0 12px 3px #22d3eeaa,0 0 40px 8px #22d3ee44;top:12%;animation:scan 3.2s ease-in-out infinite}
@keyframes scan{0%{top:2%}50%{top:96%}100%{top:2%}}
.box{position:absolute;border:2px solid #f59e0b;border-radius:4px;opacity:0;animation:lock 3.2s infinite}
.box::before,.box::after{content:"";position:absolute;width:10px;height:10px;border:2px solid #fff}
.box::before{left:-4px;top:-4px;border-right:0;border-bottom:0}.box::after{right:-4px;bottom:-4px;border-left:0;border-top:0}
.box b{position:absolute;left:-2px;top:-24px;background:#f59e0b;color:#0f1f1a;font:600 11px 'IBM Plex Mono',monospace;padding:3px 7px;border-radius:3px;white-space:nowrap}
@keyframes lock{0%,30%{opacity:0;transform:scale(1.4)}38%{opacity:1;transform:scale(1)}85%{opacity:1}100%{opacity:0}}
.belt{position:absolute;left:0;right:0;bottom:0;height:26px;background:repeating-linear-gradient(90deg,#2a3a35 0 18px,#1b2824 18px 36px);background-size:72px 26px;animation:bm 1.2s linear infinite;border-top:2px solid #3b4d47}
@keyframes bm{to{background-position:72px 0}}
.hud{position:absolute;right:14px;top:12px;font:500 10px 'IBM Plex Mono',monospace;color:#7dd3c0;text-align:right;line-height:1.6}
.ap{position:absolute;left:14px;top:12px;width:30px;height:30px;animation:rot 6s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
@media(max-width:820px){.w{grid-template-columns:1fr}.r{display:none}h1{font-size:32px}}
</style></head><body><div class="w">
<div class="l"><div class="kicker"><i></i>__BADGE__</div><h1><span>__TITLE_HTML__</span></h1><p>__SUB__</p>
<div class="pills"><span>96×96 grayscale</span><span>5 classes</span><span>split by sheet</span><span>human-in-the-loop</span></div></div>
<div class="r">
<svg class="ap" viewBox="0 0 40 40"><g fill="none" stroke="#7dd3c0" stroke-width="2"><circle cx="20" cy="20" r="17"/>
<path d="M20 3 L26 18 M37 20 L22 26 M20 37 L14 22 M3 20 L18 14"/></g></svg>
<div class="hud">CAM-02 · 120 fps<br>EXPOSURE AUTO<br><span id="n">0</span> parts inspected</div>
<div class="stage"><svg viewBox="0 0 300 200"><defs><filter id="t"><feTurbulence type="fractalNoise" baseFrequency=".9 .03" numOctaves="3" seed="4"/>
<feColorMatrix values="0 0 0 0 .55  0 0 0 0 .58  0 0 0 0 .6  0 0 0 .55 0"/></filter>
<linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#9aa4a8"/><stop offset=".5" stop-color="#c7cdd0"/><stop offset="1" stop-color="#8e989c"/></linearGradient></defs>
<rect width="300" height="200" fill="url(#g)"/><rect width="300" height="200" filter="url(#t)"/>
<path d="M70 150 L115 120 L128 126 L170 92" stroke="#3b4246" stroke-width="2.4" fill="none" stroke-linecap="round"/>
<circle cx="230" cy="60" r="4" fill="#4b5357"/><circle cx="241" cy="69" r="2.6" fill="#4b5357"/></svg>
<div class="box" style="left:20%;top:40%;width:40%;height:40%"><b>CRACK 0.93</b></div><div class="laser"></div></div>
<div class="belt"></div></div></div>
<script>let n=1280;setInterval(()=>{n+=1;document.getElementById('n').textContent=n.toLocaleString()},420)</script>
</body></html>"""


def hero(title: str, subtitle: str, badge: str = "Live", icon: str = "🔬", height: int = 250) -> None:
    words = _e(title).split(" ")
    title_html = " ".join(words[:-1] + [f"<em>{words[-1]}</em>"]) if len(words) > 1 else _e(title)
    doc = (_HERO.replace("__H__", str(height - 8)).replace("__BADGE__", _e(badge)).replace("__TITLE_HTML__", title_html)
           .replace("__SUB__", _e(subtitle)))
    components.html(doc, height=height)


def strip(items: list[tuple[str, str, str]]) -> None:
    """Production-line status strip: (label, value, colour)."""
    body = "".join(f'<span><span class="led" style="--c:{c}"></span>{_e(lab)} <b>{_e(v)}</b></span>' for lab, v, c in items)
    st.markdown(f'<div class="strip">{body}</div>', unsafe_allow_html=True)


def kpis(items: Iterable[tuple]) -> None:
    cards = []
    for i, it in enumerate(items):
        label, value, delta, icon, color = (list(it) + [None] * 5)[:5]
        color = color or PRIMARY
        d = ""
        if delta:
            dc = SUCCESS if str(delta).startswith(("+", "▲")) else DANGER if str(delta).startswith(("-", "−", "▼")) else MUTED
            d = f'<div class="delta" style="color:{dc}">{_e(delta)}</div>'
        cards.append(f'<div class="kpi" style="--c1:{color};animation-delay:{i * .08:.2f}s"><div class="icon">{icon or ""}</div>'
                     f'<div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>{d}</div>')
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


_N = {"i": 0}


def section(title: str, sub: str = "") -> None:
    _N["i"] = _N["i"] % 99 + 1
    st.markdown(f'<div class="section"><span class="tag">§{_N["i"]:02d}</span><h3>{_e(title)}</h3><span class="sub">{_e(sub)}</span></div>',
                unsafe_allow_html=True)


def card(title: str, body: str, accent: str = PRIMARY) -> None:
    st.markdown(f'<div class="card" style="border-top:4px solid {accent}"><h4>{_e(title)}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def chip(text: str, color: str) -> str:
    return f'<span class="chip" style="--c:{color}">{_e(text)}</span>'


def timeline(events: list[dict], limit: int = 8) -> None:
    out = []
    for i, ev in enumerate(events[:limit]):
        sev = SEVERITY_COLORS.get(ev.get("severity", "info"), PRIMARY)
        out.append(f'<div class="tl-item" style="--sev:{sev};animation-delay:{i * .05:.2f}s"><div class="meta">'
                   f'{chip(ev.get("event_type", ""), sev)} {_e(ev.get("batch_id", ""))} · {_e(ev.get("ts", ""))}</div>'
                   f'<div class="msg">{_e(ev.get("message", ""))}</div></div>')
    st.markdown("".join(out) or '<div class="card"><p>No events.</p></div>', unsafe_allow_html=True)


def ring(value: float, label: str, color: str | None = None) -> None:
    color = color or (SUCCESS if value >= 80 else WARNING if value >= 60 else DANGER)
    st.markdown(f'<div style="text-align:center"><div class="ring" style="--val:{value:.0f};--rc:{color}"><div>{value:.0f}</div></div>'
                f'<div style="color:{MUTED};font-size:.78rem;font-weight:700;margin-top:10px">{_e(label)}</div></div>', unsafe_allow_html=True)


def flow(nodes: list[tuple[str, str]]) -> None:
    parts = [f'<div class="node" style="animation-delay:{i * .1:.1f}s">{_e(n)}<small>{_e(s)}</small></div>' for i, (n, s) in enumerate(nodes)]
    st.markdown(f'<div class="flow">{"<div class=belt></div>".join(parts)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="footer">{BRAND} · research prototype · decision support only — human inspection required</div>',
                unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None)
    return fig
