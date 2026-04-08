"""
utils/media_controller.py

YouTube playback using Selenium browser automation.
  - Searches YouTube for the query
  - Automatically clicks the FIRST real video result
  - Video plays in a real Chrome browser with full audio
  - Falls back to plain webbrowser.open() if Selenium is not installed

Requirements (install once):
    pip install selenium webdriver-manager
"""

import re
import time
import threading
import webbrowser
import urllib.parse
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Clean-up words stripped from voice commands before searching ──────────────
_STRIP = re.compile(
    r'\b(please|hey|ok|okay|can you|could you|play|music|song|video'
    r'|on youtube|youtube|for me|now)\b',
    flags=re.I,
)


def _clean_query(raw: str) -> str:
    """Strip filler words but keep the actual song / artist name."""
    cleaned = _STRIP.sub(" ", raw)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned or raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
class MediaController:

    def __init__(self):
        self._driver = None          # Selenium WebDriver instance (reused)
        self._driver_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def play_youtube(self, query: str) -> str:
        """
        Search YouTube and auto-click the first video result so it plays.
        Runs the browser in a background thread so Streamlit doesn't block.
        """
        query = _clean_query(query)
        if not query:
            query = "relaxing music"

        logger.info(f"MediaController.play_youtube: '{query}'")

        # Try Selenium first
        try:
            from selenium import webdriver  # quick import check
            thread = threading.Thread(
                target=self._selenium_play,
                args=(query,),
                daemon=True,
            )
            thread.start()
            return f"▶️ Opening YouTube and playing **'{query}'** now…"

        except ImportError:
            # Selenium not installed → plain browser fallback
            logger.warning("Selenium not installed — falling back to webbrowser.open()")
            return self._plain_open(query)

    def stop(self) -> str:
        """Close the YouTube browser window."""
        with self._driver_lock:
            if self._driver:
                try:
                    self._driver.quit()
                except Exception:
                    pass
                self._driver = None
                return "⏹️ YouTube stopped."
        return "Nothing is playing."

    # ── Selenium player (runs in background thread) ───────────────────────────

    def _selenium_play(self, query: str):
        """
        1. Open Chrome via Selenium
        2. Navigate to YouTube search results
        3. Find and click the first real video (skip ads/shorts)
        4. Wait for the video to load and start playing
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service

            options = Options()
            options.add_argument("--autoplay-policy=no-user-gesture-required")
            options.add_argument("--disable-notifications")
            options.add_argument("--mute-audio=false")
            options.add_argument("--start-maximized")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            # DO NOT use headless — audio won't play in headless mode

            with self._driver_lock:
                # Close old window if open
                if self._driver:
                    try:
                        self._driver.quit()
                    except Exception:
                        pass

                service = Service(ChromeDriverManager().install())
                driver  = webdriver.Chrome(service=service, options=options)
                self._driver = driver

            # ── Step 1: Go to YouTube search ─────────────────────────────────
            encoded    = urllib.parse.quote_plus(query)
            search_url = f"https://www.youtube.com/results?search_query={encoded}"
            driver.get(search_url)
            logger.info(f"Navigated to: {search_url}")

            # ── Step 2: Wait for results to load ─────────────────────────────
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "ytd-video-renderer")
            ))
            time.sleep(1.5)   # let lazy-loaded thumbnails settle

            # ── Step 3: Find first real video (skip ads / shorts) ─────────────
            video_link = None
            candidates = driver.find_elements(
                By.CSS_SELECTOR,
                "ytd-video-renderer #video-title"
            )
            for el in candidates:
                href = el.get_attribute("href") or ""
                # skip shorts and ads
                if "/watch?v=" in href and "/shorts/" not in href:
                    video_link = href
                    logger.info(f"Found video: {el.text.strip()[:60]}")
                    break

            if not video_link:
                # Fallback: grab first /watch?v= anchor on the page
                anchors = driver.find_elements(By.TAG_NAME, "a")
                for a in anchors:
                    href = a.get_attribute("href") or ""
                    if "/watch?v=" in href and "/shorts/" not in href:
                        video_link = href
                        break

            if not video_link:
                logger.warning("No video link found — staying on search page")
                return

            # ── Step 4: Navigate to the video ────────────────────────────────
            driver.get(video_link)
            logger.info(f"Playing: {video_link}")

            # ── Step 5: Click play button if video is paused ─────────────────
            time.sleep(3)   # wait for video player to load
            try:
                # Use JS click — more reliable than Selenium click for YouTube
                driver.execute_script(
                    "var v = document.querySelector('video'); if(v && v.paused) v.play();"
                )
                logger.info("Triggered video play via JS")
            except Exception:
                pass
            try:
                play_btn = driver.find_element(
                    By.CSS_SELECTOR, "button.ytp-play-button"
                )
                aria = play_btn.get_attribute("aria-label") or ""
                if "Play" in aria:
                    driver.execute_script("arguments[0].click();", play_btn)
                    logger.info("Clicked play button")
            except Exception:
                pass

            # ── Step 6: Dismiss cookie / sign-in popups if any ───────────────
            time.sleep(1)
            for selector in [
                "button[aria-label='Accept all']",
                "button[aria-label='Reject all']",
                "tp-yt-paper-dialog #dismiss-button",
            ]:
                try:
                    btn = driver.find_element(By.CSS_SELECTOR, selector)
                    btn.click()
                    time.sleep(0.5)
                except Exception:
                    pass

            logger.info("Video is playing ✅")

        except Exception as e:
            logger.error(f"Selenium playback error: {e}")
            # Ultimate fallback — open in default browser
            self._plain_open(query)

    # ── Plain fallback ────────────────────────────────────────────────────────

    def _plain_open(self, query: str) -> str:
        encoded = urllib.parse.quote_plus(query)
        url     = f"https://www.youtube.com/results?search_query={encoded}"
        webbrowser.open(url)
        return (
            f"▶️ Opened YouTube for **'{query}'**.\n"
            f"_(Install `selenium` and `webdriver-manager` for auto-play.)_"
        )

    # ── Local file playback ───────────────────────────────────────────────────

    def play_music_file(self, filepath: str) -> str:
        try:
            import subprocess
            import platform
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(["start", filepath], shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", filepath])
            else:
                subprocess.Popen(["xdg-open", filepath])
            return f"🎵 Playing: {filepath}"
        except Exception as e:
            return f"❌ Playback error: {e}"
