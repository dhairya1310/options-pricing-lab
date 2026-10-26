"""
Visit the live app in a real browser so Streamlit Community Cloud keeps it awake.

Free Streamlit apps go to sleep after 12 hours without a visitor, and commits to
the repo do not count as visits. A GitHub Actions job runs this every 6 hours.

Run it yourself:
    python scripts/keep_alive.py
    python scripts/keep_alive.py --url http://localhost:8501   (test against a local app)

What it does:
    1. opens the app
    2. if the "get this app back up" button is showing, clicks it
    3. waits until the app's title appears, in any frame on the page
       (Streamlit Cloud serves the app inside an iframe)
    4. exits with an error if the app never loads, so the job turns red
       and GitHub sends an email
"""

import argparse
import re
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = "https://options-pricing-lab.streamlit.app"
APP_TITLE = "Options Pricing Lab"
WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
LOAD_TIMEOUT_S = 240  # a cold start can take a few minutes


def app_loaded(page) -> bool:
    """True once the app's title text shows up in the page or in any iframe."""
    for frame in page.frames:
        try:
            if frame.get_by_text(APP_TITLE, exact=False).count() > 0:
                return True
        except Exception:
            continue  # frames can detach while the app reloads
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=APP_URL)
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(10)  # let the sleep screen or the app render

        button = page.get_by_role("button", name=WAKE_BUTTON)
        if button.count() > 0:
            print("App was asleep, clicking the wake up button")
            button.first.click()
        else:
            print("App was awake")

        start = time.time()
        while time.time() - start < LOAD_TIMEOUT_S:
            if app_loaded(page):
                print(f"App loaded after {time.time() - start:.0f} seconds")
                browser.close()
                return
            time.sleep(5)

        page.screenshot(path="keep_alive_failure.png", full_page=True)
        browser.close()
        print(f"App did not load within {LOAD_TIMEOUT_S} seconds", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
