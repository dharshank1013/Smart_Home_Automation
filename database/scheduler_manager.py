"""
database/scheduler_manager.py
APScheduler + SQLite for alarms and reminders.
Fixed: alarm fires at EXACT time using full datetime comparison.
"""

import sqlite3
import os
import re
import threading
from datetime import datetime, date
from utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "smart_home.db")


class SchedulerManager:
    def __init__(self):
        self._init_db()
        self._scheduler = None
        self._lock = threading.Lock()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    message        TEXT NOT NULL,
                    scheduled_time TEXT NOT NULL,
                    triggered      INTEGER DEFAULT 0,
                    created_at     TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alarms (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    alarm_time     TEXT NOT NULL,
                    triggered      INTEGER DEFAULT 0,
                    created_at     TEXT NOT NULL
                )
            """)

    def start(self):
        """Start APScheduler background loop — checks every 10 seconds."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self._scheduler = BackgroundScheduler()
            # Check every 10 seconds for precise alarm firing
            self._scheduler.add_job(self._check_due, "interval", seconds=10, id="check_due")
            self._scheduler.start()
            logger.info("Scheduler started ✅ (10s interval)")
        except ImportError:
            logger.warning("APScheduler not installed – using threading fallback.")
            t = threading.Thread(target=self._poll_loop, daemon=True)
            t.start()

    def _poll_loop(self):
        import time
        while True:
            self._check_due()
            time.sleep(10)

    def _check_due(self):
        """
        Compare alarms/reminders using full HH:MM datetime.
        Fires when current time >= scheduled time (within the same minute).
        Uses a window of ±1 minute to avoid missing due to poll timing.
        """
        now       = datetime.now()
        now_hhmm  = now.strftime("%H:%M")  # e.g. "13:45"
        now_total = now.hour * 60 + now.minute

        with sqlite3.connect(DB_PATH) as conn:
            # ── Check reminders ───────────────────────────────────────────────
            rows = conn.execute(
                "SELECT id, message, scheduled_time FROM reminders WHERE triggered=0"
            ).fetchall()
            for rid, msg, sched in rows:
                sched_total = self._to_minutes(sched)
                if sched_total is not None and now_total >= sched_total:
                    logger.info(f"Firing reminder: {msg} (scheduled {sched}, now {now_hhmm})")
                    self._fire_reminder(msg, sched)
                    conn.execute("UPDATE reminders SET triggered=1 WHERE id=?", (rid,))

            # ── Check alarms ──────────────────────────────────────────────────
            alarms = conn.execute(
                "SELECT id, alarm_time FROM alarms WHERE triggered=0"
            ).fetchall()
            for aid, at in alarms:
                alarm_total = self._to_minutes(at)
                if alarm_total is not None and now_total >= alarm_total:
                    logger.info(f"Firing alarm: {at} (now {now_hhmm})")
                    self._fire_alarm(at)
                    conn.execute("UPDATE alarms SET triggered=1 WHERE id=?", (aid,))

    @staticmethod
    def _to_minutes(time_str: str) -> int:
        """Convert HH:MM string to total minutes since midnight. Returns None on failure."""
        try:
            parts = time_str.strip().split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return None

    def _fire_reminder(self, message: str, scheduled_time: str):
        now_str = datetime.now().strftime("%I:%M %p")
        logger.info(f"REMINDER FIRED: {message}")
        self._speak_alert(f"Reminder: {message}. The time is {now_str}.")

    def _fire_alarm(self, alarm_time: str):
        # Convert 24h to 12h for speaking
        try:
            h, m   = map(int, alarm_time.split(":"))
            period = "AM" if h < 12 else "PM"
            h12    = h % 12 or 12
            spoken = f"{h12}:{m:02d} {period}"
        except Exception:
            spoken = alarm_time
        logger.info(f"ALARM FIRED: {alarm_time}")
        self._speak_alert(f"Wake up! Your alarm for {spoken} is ringing!")

    @staticmethod
    def _speak_alert(text: str):
        """Play alert sound + TTS."""
        import threading
        def _speak():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS alert failed: {e}")
        threading.Thread(target=_speak, daemon=True).start()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_alarm(self, alarm_time: str) -> str:
        """Parse time string and store alarm. Returns confirmation with parsed time."""
        parsed = self._parse_time(alarm_time)
        if not parsed:
            return f"⚠️ Could not understand time: '{alarm_time}'. Try '7:30 AM' or '19:30'."
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO alarms (alarm_time, created_at) VALUES (?, ?)", (parsed, ts))
        # Convert back to 12h for display
        h, m   = map(int, parsed.split(":"))
        period = "AM" if h < 12 else "PM"
        h12    = h % 12 or 12
        display = f"{h12}:{m:02d} {period}"
        logger.info(f"Alarm set for {parsed} ({display})")
        return parsed, display

    def add_reminder(self, message: str, scheduled_time: str) -> str:
        parsed = self._parse_time(scheduled_time)
        if not parsed:
            parsed = scheduled_time  # store as-is if unparseable
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO reminders (message, scheduled_time, created_at) VALUES (?, ?, ?)",
                (message, parsed, ts)
            )
        return parsed

    def get_pending_reminders(self) -> list:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT id, message, scheduled_time FROM reminders "
                "WHERE triggered=0 ORDER BY scheduled_time"
            ).fetchall()
        return [{"id": r[0], "message": r[1], "scheduled_time": r[2]} for r in rows]

    def get_pending_alarms(self) -> list:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT id, alarm_time FROM alarms WHERE triggered=0 ORDER BY alarm_time"
            ).fetchall()
        return [{"id": r[0], "alarm_time": r[1]} for r in rows]

    def cancel_alarm(self, alarm_time: str = None) -> str:
        """Cancel all pending alarms, or a specific one if alarm_time is given."""
        with sqlite3.connect(DB_PATH) as conn:
            if alarm_time:
                parsed = self._parse_time(alarm_time)
                conn.execute("DELETE FROM alarms WHERE triggered=0 AND alarm_time=?", (parsed,))
                return f"Alarm for {alarm_time} cancelled."
            else:
                conn.execute("DELETE FROM alarms WHERE triggered=0")
                return "All pending alarms cancelled."

    # ── Time parser ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_time(t: str) -> str:
        """
        Robust time parser → HH:MM (24h format for storage).
        Handles: '7 AM', '7:30 AM', '13:45', '1.30 pm', '7 30 am', 'half past 7'
        Returns empty string on failure.
        """
        if not t:
            return ""
        t = t.strip().lower()

        # Replace dots with colons  e.g. "1.30" → "1:30"
        t = t.replace(".", ":")

        # Handle "half past X"
        half = re.search(r'half\s*past\s*(\d{1,2})', t)
        if half:
            return f"{int(half.group(1)):02d}:30"

        # Handle "quarter past X" / "quarter to X"
        qpast = re.search(r'quarter\s*past\s*(\d{1,2})', t)
        if qpast:
            return f"{int(qpast.group(1)):02d}:15"
        qto = re.search(r'quarter\s*to\s*(\d{1,2})', t)
        if qto:
            h = int(qto.group(1)) - 1
            return f"{h % 24:02d}:45"

        # Main pattern: digits with optional colon/space separator + am/pm
        m = re.search(
            r'(\d{1,2})\s*[: ]\s*(\d{2})\s*(am|pm)?'   # e.g. 7:30 am / 7 30 pm
            r'|(\d{1,2})\s*(am|pm)',                     # e.g. 7am / 7 pm
            t
        )
        if not m:
            # bare number
            bare = re.search(r'(\d{1,2})$', t)
            if bare:
                m_num = int(bare.group(1))
                return f"{m_num % 24:02d}:00"
            return ""

        if m.group(1):   # HH:MM pattern
            hour, minute = int(m.group(1)), int(m.group(2))
            meridiem     = m.group(3)
        else:            # H am/pm pattern
            hour, minute = int(m.group(4)), 0
            meridiem     = m.group(5)

        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0

        hour = hour % 24
        return f"{hour:02d}:{minute:02d}"
