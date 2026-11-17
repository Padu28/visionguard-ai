"""Capture a screenshot of every dashboard page for the README.

Usage (with the dashboard already running in another terminal):

    pip install playwright
    python -m playwright install chromium
    python scripts/capture_screenshots.py            # uses the default port below
    python scripts/capture_screenshots.py --port 8600

Images are written to docs/screenshots/<page>.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PORT = 8503
PAGES = [('', 'overview'), ('inspect', 'inspect'), ('error_analysis', 'error_analysis'), ('evaluation', 'evaluation'), ('robustness', 'robustness'), ('dataset_audit', 'dataset_audit')]  # (url_path, file name) — "" is the default page

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"

# Let Streamlit's scroll container grow so a full-page screenshot captures everything.
EXPAND_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main {
  height: auto !important; min-height: 0 !important; overflow: visible !important; position: static !important; }
[data-testid="stStatusWidget"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--wait", type=float, default=7.0, help="seconds to let charts and animations render")
    ap.add_argument("--theme", choices=["dark", "light"], default="light")
    a = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install first:  pip install playwright  &&  python -m playwright install chromium")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    base = f"http://localhost:{a.port}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": a.width, "height": 1000}, color_scheme=a.theme, device_scale_factor=1)
        try:
            page.goto(base, wait_until="networkidle", timeout=60_000)
        except Exception:
            print(f"Could not reach {base}. Start the dashboard first (see README → Quick start).")
            browser.close()
            return 1
        for url_path, name in PAGES:
            url = f"{base}/{url_path}"
            print(f"→ {name:<24} {url}")
            page.goto(url, wait_until="networkidle", timeout=120_000)
            page.wait_for_selector('[data-testid="stMain"], section.main', timeout=60_000)
            time.sleep(a.wait)
            page.add_style_tag(content=EXPAND_CSS)
            time.sleep(0.8)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        browser.close()
    print(f"Saved {len(PAGES)} screenshots to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
