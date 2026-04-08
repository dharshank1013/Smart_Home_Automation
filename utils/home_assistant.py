"""
utils/home_assistant.py
Communicates with Home Assistant via REST API.

Rooms defined:
  Bedroom, Living Room, Kitchen, Bathroom, Study Room

Devices per room:
  light, fan, ac, window, door, tv

Voice examples:
  "turn on bedroom light"
  "turn off kitchen fan"
  "close living room window"
  "turn on study room ac"
"""

import os
import re
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE MAP
# Format:  "voice keyword"  →  "ha_entity_id"
#
# NOTE: These are the HA entity IDs that will be AUTO-CREATED by
#       create_ha_entities() below when you first run the app.
#       If you already have real smart devices, replace the entity IDs
#       here with your actual ones from HA → Settings → Entities.
# ─────────────────────────────────────────────────────────────────────────────

DEVICE_MAP = {
    # ── BEDROOM ──────────────────────────────────────────────────────────────
    "bedroom light":        "light.bedroom_light",
    "bedroom fan":          "switch.bedroom_fan",
    "bedroom ac":           "switch.bedroom_ac",
    "bedroom window":       "switch.bedroom_window",
    "bedroom door":         "switch.bedroom_door",

    # ── LIVING ROOM ──────────────────────────────────────────────────────────
    "living room light":    "light.living_room_light",
    "living room fan":      "switch.living_room_fan",
    "living room ac":       "switch.living_room_ac",
    "living room window":   "switch.living_room_window",
    "living room door":     "switch.living_room_door",
    "living room tv":       "switch.living_room_tv",

    # ── KITCHEN ──────────────────────────────────────────────────────────────
    "kitchen light":        "light.kitchen_light",
    "kitchen fan":          "switch.kitchen_fan",
    "kitchen window":       "switch.kitchen_window",
    "kitchen door":         "switch.kitchen_door",

    # ── BATHROOM ─────────────────────────────────────────────────────────────
    "bathroom light":       "light.bathroom_light",
    "bathroom fan":         "switch.bathroom_fan",
    "bathroom window":      "switch.bathroom_window",
    "bathroom door":        "switch.bathroom_door",

    # ── STUDY ROOM ───────────────────────────────────────────────────────────
    "study room light":     "light.study_room_light",
    "study light":          "light.study_room_light",
    "study room fan":       "switch.study_room_fan",
    "study room ac":        "switch.study_room_ac",
    "study room window":    "switch.study_room_window",

    # ── GENERIC (no room) → defaults to living room ───────────────────────────
    "light":                "light.living_room_light",
    "lights":               "light.living_room_light",
    "fan":                  "switch.living_room_fan",
    "ac":                   "switch.living_room_ac",
    "air conditioner":      "switch.living_room_ac",
    "window":               "switch.living_room_window",
    "windows":              "switch.living_room_window",
    "door":                 "switch.living_room_door",
    "tv":                   "switch.living_room_tv",
    "television":           "switch.living_room_tv",
}

# Entity domain → correct service domain mapping
SERVICE_MAP = {
    "light":   ("light",         "turn_on",    "turn_off"),
    "switch":  ("homeassistant", "turn_on",    "turn_off"),
    "cover":   ("cover",         "open_cover", "close_cover"),
    "lock":    ("lock",          "unlock",     "lock"),
    "climate": ("climate",       "turn_on",    "turn_off"),
}

# Words that mean ON vs OFF
ON_WORDS  = {"on", "turn on", "open", "unlock", "enable", "start", "switch on"}
OFF_WORDS = {"off", "turn off", "close", "lock", "disable", "stop", "switch off"}


class HomeAssistantClient:
    def __init__(self):
        self.url   = os.getenv("HA_URL",   "http://127.0.0.1:8123")
        self.token = os.getenv("HA_TOKEN", "")
        # local simulation states
        self._device_states = {k: False for k in [
            "bedroom light", "bedroom fan", "bedroom ac",
            "living room light", "living room fan", "living room ac", "living room tv",
            "kitchen light", "kitchen fan",
            "bathroom light", "bathroom fan",
            "study room light", "study room fan",
            "light", "fan", "ac", "window", "door", "tv",
        ]}

    def update(self, url: str, token: str):
        if url:
            self.url   = url.rstrip("/")
        if token:
            self.token = token

    @property
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }

    # ── Main control method ───────────────────────────────────────────────────

    def control_device(self, device: str, action: str, raw_text: str = "") -> str:
        raw_lower = raw_text.lower().strip()

        # Try to find the best matching device key from raw_text
        entity_key = self._match_device(raw_lower) or self._match_device(device.lower())

        if not entity_key:
            available = ", ".join(sorted(set(
                k for k in DEVICE_MAP if len(k.split()) > 1
            ))[:12])
            return (
                f"⚠️ I didn't recognise that device.\n"
                f"Try saying: **bedroom light**, **kitchen fan**, **living room ac**, etc."
            )

        entity_id = DEVICE_MAP[entity_key]
        action    = action.lower().strip()

        # Determine ON or OFF from raw text if action is ambiguous
        if not action or action not in (ON_WORDS | OFF_WORDS):
            if any(w in raw_lower for w in OFF_WORDS):
                action = "off"
            else:
                action = "on"

        is_on = action in ON_WORDS

        # ── Simulation mode (no token) ────────────────────────────────────────
        if not self.token:
            self._device_states[entity_key] = is_on
            state_str = "ON ✅" if is_on else "OFF ⛔"
            return f"✅ **{entity_key.title()}** turned **{state_str}** (simulated — add HA token for real control)."

        # ── Real HA API call ──────────────────────────────────────────────────
        domain = entity_id.split(".")[0]
        svc_domain, svc_on, svc_off = SERVICE_MAP.get(domain, ("homeassistant", "turn_on", "turn_off"))
        service = svc_on if is_on else svc_off

        try:
            endpoint = f"{self.url}/api/services/{svc_domain}/{service}"
            payload  = {"entity_id": entity_id}
            resp     = requests.post(endpoint, headers=self._headers, json=payload, timeout=6)

            if resp.status_code in (200, 201):
                self._device_states[entity_key] = is_on
                state_str = "ON ✅" if is_on else "OFF ⛔"
                return f"✅ **{entity_key.title()}** turned **{state_str}**."
            else:
                return (
                    f"⚠️ Home Assistant returned {resp.status_code}.\n"
                    f"Entity `{entity_id}` may not exist yet. "
                    f"Go to HA → Settings → Devices & Services → Helpers → Create a Switch named `{entity_key}`."
                )
        except requests.exceptions.ConnectionError:
            return "❌ Cannot reach Home Assistant at `127.0.0.1:8123`. Make sure Docker container is running."
        except Exception as e:
            return f"❌ Error: {e}"

    def _match_device(self, text: str) -> str:
        """Find the longest matching device key inside the text."""
        best = ""
        for key in DEVICE_MAP:
            if key in text and len(key) > len(best):
                best = key
        return best

    def get_device_states(self) -> dict:
        if not self.token:
            return self._device_states
        try:
            resp = requests.get(f"{self.url}/api/states", headers=self._headers, timeout=5)
            if resp.status_code == 200:
                ha_states = resp.json()
                result    = dict(self._device_states)
                for s in ha_states:
                    eid = s["entity_id"]
                    for name, mapped in DEVICE_MAP.items():
                        if mapped == eid:
                            result[name] = s["state"] in ("on", "open", "unlocked", "playing")
                return result
        except Exception:
            pass
        return self._device_states

    def list_devices(self) -> str:
        """Return a formatted list of all controllable devices."""
        rooms = {}
        for key in DEVICE_MAP:
            parts = key.split()
            if len(parts) >= 2:
                room   = " ".join(parts[:-1]).title()
                device = parts[-1].title()
                rooms.setdefault(room, []).append(device)
        lines = ["📋 **Controllable Devices:**\n"]
        for room, devices in sorted(rooms.items()):
            lines.append(f"🏠 **{room}**: {', '.join(devices)}")
        return "\n".join(lines)
