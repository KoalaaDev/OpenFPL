import sys, time
from playwright.sync_api import sync_playwright
out = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1200})
    pg.goto("http://127.0.0.1:8413/", wait_until="networkidle")
    time.sleep(4)
    pg.click("nav.tabs button:has-text('Live')", timeout=6000)
    time.sleep(5)
    pg.screenshot(path=out + "/live.png")
    pg.screenshot(path=out + "/live_full.png", full_page=True)
    b.close()
print("done")
