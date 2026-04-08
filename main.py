"""
Smart Home Automation AI Assistant
Main entry point - Streamlit UI
"""

import streamlit as st
import threading
import queue
import time
import os
from mymodel import SmartHomeAssistant
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Home AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-title { font-size: 2.5rem; font-weight: 700; color: #00d4ff; text-align: center; }
    .status-card { background: #1e2130; border-radius: 12px; padding: 16px; margin: 8px 0; border-left: 4px solid #00d4ff; }
    .action-badge { background: #00d4ff22; border: 1px solid #00d4ff; color: #00d4ff; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; display: inline-block; margin: 4px; }
    .response-box { background: #1e2130; border-radius: 12px; padding: 20px; border: 1px solid #2d3250; color: #e0e0e0; }
    .device-card { background: #1e2130; border-radius: 10px; padding: 14px; text-align: center; }
    .device-on  { border: 2px solid #00ff88; }
    .device-off { border: 2px solid #555; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "device_states" not in st.session_state:
    st.session_state.device_states = {
        "lights": False, "fan": False, "ac": False,
        "door": False, "windows": False, "tv": False
    }
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False

# ── Load assistant ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading AI models…")
def load_assistant():
    return SmartHomeAssistant()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🏠 Home Assistant")
    ha_url = st.text_input("HA URL", value=os.getenv("HA_URL", "http://127.0.0.1:8123"), key="ha_url")
    ha_token = st.text_input("Long-Lived Token", value=os.getenv("HA_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIyN2UwZGFlNzUyMGQ0MmZmOTFiYmI0NTRhZmQzOTA1YSIsImlhdCI6MTc3NTI4NjMwOCwiZXhwIjoyMDkwNjQ2MzA4fQ.Qt9VwdUkJo6Fp3VN4MO2DPxXBgTIiasFbzPM4mAW3tM"), type="password", key="ha_token")

    st.markdown("### 🔍 Search")
    st.success("✅ Free search active\nWikipedia + DuckDuckGo\n(No API key needed)")
    google_api_key = ""
    google_cx = ""

    st.markdown("### 🎙️ Audio Settings")
    mic_sensitivity = st.slider("Mic Sensitivity", 0.1, 1.0, 0.5)

    st.markdown("---")
    if st.button("💾 Save & Apply Config", use_container_width=True):
        os.environ["HA_URL"]         = ha_url
        os.environ["HA_TOKEN"]       = ha_token
        os.environ["GOOGLE_API_KEY"] = google_api_key
        os.environ["GOOGLE_CX"]      = google_cx
        st.session_state.assistant   = None          # force reload
        st.success("Config saved!")

    st.markdown("---")
    st.markdown("### 📊 System Status")
    try:
        assistant = load_assistant()
        st.success("✅ All models loaded")
        st.info(f"🎙️ Whisper: Ready\n\n🔊 Svara TTS: Ready\n\n🧠 Phi-4: Ready")
    except Exception as e:
        st.error(f"❌ Model load error: {e}")

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏠 Smart Home AI Assistant</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;'>Voice & Text Control · Powered by Whisper · Phi-4 · Svara TTS</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Voice Assistant", "📱 Device Control", "📋 Notes & Reminders", "📜 History"])

# ══ TAB 1: Voice / Text Assistant ════════════════════════════════════════════
with tab1:
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.markdown("### 🎤 Input")

        input_mode = st.radio("Mode", ["🎙️ Voice", "⌨️ Text"], horizontal=True)

        if input_mode == "⌨️ Text":
            user_text = st.text_area("Type your command:", placeholder="e.g. Turn on the lights, Play music on YouTube…", height=120)
            if st.button("🚀 Send Command", use_container_width=True):
                if user_text.strip():
                    with st.spinner("Processing…"):
                        try:
                            assistant = load_assistant()
                            assistant.update_config(ha_url, ha_token, google_api_key, google_cx)
                            result = assistant.process_command(user_text.strip())
                            st.session_state.conversation_history.append({
                                "time": time.strftime("%H:%M:%S"),
                                "user": user_text.strip(),
                                "assistant": result["response"],
                                "action": result.get("action", "general"),
                            })
                            if "device_states" in result:
                                st.session_state.device_states.update(result["device_states"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        else:  # Voice mode
            st.info("🎙️ Press Record to start, then Stop when done speaking.")

            # Session state for streaming recording
            if "rec_thread" not in st.session_state:
                st.session_state.rec_thread = None
            if "stop_event" not in st.session_state:
                st.session_state.stop_event = None
            if "audio_bucket" not in st.session_state:
                st.session_state.audio_bucket = None  # list used as mutable container

            col_rec, col_stop = st.columns(2)

            with col_rec:
                if st.button("🔴 Record", use_container_width=True,
                             disabled=st.session_state.is_listening):
                    stop_event = threading.Event()
                    bucket = []   # thread writes audio here
                    _asst = load_assistant()
                    _asst.stop_tts()  # stop any playing TTS before recording

                    def _record(asst=_asst, ev=stop_event, b=bucket):
                        audio = asst.record_audio_stream(ev)
                        b.append(audio)

                    t = threading.Thread(target=_record, daemon=True)
                    t.start()
                    st.session_state.rec_thread = t
                    st.session_state.stop_event = stop_event
                    st.session_state.audio_bucket = bucket
                    st.session_state.is_listening = True
                    st.rerun()

            with col_stop:
                if st.button("⏹️ Stop", use_container_width=True,
                             disabled=not st.session_state.is_listening):
                    # Signal the recording thread to stop
                    if st.session_state.stop_event:
                        st.session_state.stop_event.set()
                    # Wait for it to finish (max 10 s) so audio_bucket is populated
                    if st.session_state.rec_thread:
                        st.session_state.rec_thread.join(timeout=10)
                    st.session_state.is_listening = False
                    st.rerun()

            if st.session_state.is_listening:
                st.markdown('<div class="status-card">🔴 Recording… press Stop when done</div>',
                            unsafe_allow_html=True)

            # Process audio right after Stop — bucket is already filled because we joined
            bucket = st.session_state.get("audio_bucket")
            if not st.session_state.is_listening and bucket:
                with st.spinner("Transcribing & processing…"):
                    try:
                        audio = bucket[0]
                        _asst = load_assistant()
                        _asst.update_config(ha_url, ha_token, google_api_key, google_cx)
                        result = _asst.process_voice_from_audio(audio)
                        st.session_state.conversation_history.append({
                            "time": time.strftime("%H:%M:%S"),
                            "user": result.get("transcription", ""),
                            "assistant": result["response"],
                            "action": result.get("action", "general"),
                        })
                        if "device_states" in result:
                            st.session_state.device_states.update(result["device_states"])
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        st.session_state.audio_bucket = None
                        st.rerun()

    with col_output:
        st.markdown("### 💬 Response")
        if st.session_state.conversation_history:
            latest = st.session_state.conversation_history[-1]
            st.markdown(f'<div class="action-badge">🏷️ {latest["action"].upper()}</div>', unsafe_allow_html=True)
            if latest.get("user"):
                st.markdown(f"**You:** {latest['user']}")
            # Use st.markdown so newlines and formatting render correctly
            st.markdown(
                f'<div class="response-box">',
                unsafe_allow_html=True
            )
            st.markdown(f"🤖 {latest['assistant']}")
            st.markdown('</div>', unsafe_allow_html=True)
            # Show stop button when media is playing
            if latest.get('action') == 'media':
                if st.button('⏹️ Stop Music / Video', use_container_width=True):
                    try:
                        assistant = load_assistant()
                        msg = assistant.media.stop()
                        st.info(msg)
                    except Exception as e:
                        st.error(str(e))
        else:
            st.markdown('<div class="response-box">👋 Say something or type a command to get started!</div>', unsafe_allow_html=True)

# ══ TAB 2: Device Control ═════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📱 Smart Devices by Room")

    rooms_devices = {
        "🛏️ Bedroom":     ["bedroom light", "bedroom fan", "bedroom ac", "bedroom window"],
        "🛋️ Living Room": ["living room light", "living room fan", "living room ac", "living room tv"],
        "🍳 Kitchen":     ["kitchen light", "kitchen fan", "kitchen window"],
        "🚿 Bathroom":    ["bathroom light", "bathroom fan"],
        "📚 Study Room":  ["study room light", "study room fan", "study room ac"],
    }

    for room, device_list in rooms_devices.items():
        st.markdown(f"**{room}**")
        cols = st.columns(len(device_list))
        for i, device_key in enumerate(device_list):
            with cols[i]:
                state    = st.session_state.device_states.get(device_key, False)
                label    = device_key.split()[-1].title()
                card_cls = "device-card device-on" if state else "device-card device-off"
                st.markdown(
                    f'<div class="{card_cls}">{label}<br>'
                    f'<b>{"ON ✅" if state else "OFF ⛔"}</b></div>',
                    unsafe_allow_html=True
                )
                if st.button(f"{'OFF' if state else 'ON'}", key=f"toggle_{device_key}", use_container_width=True):
                    with st.spinner("Sending…"):
                        try:
                            assistant = load_assistant()
                            assistant.update_config(ha_url, ha_token, "", "")
                            action = "off" if state else "on"
                            result = assistant.ha_client.control_device(device_key, action, f"turn {action} {device_key}")
                            st.session_state.device_states[device_key] = not state
                            st.toast(result)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
        st.markdown("---")

# ══ TAB 3: Notes & Reminders ══════════════════════════════════════════════════
with tab3:
    col_notes, col_remind = st.columns(2)

    with col_notes:
        st.markdown("### 📝 Notes")
        try:
            assistant = load_assistant()
            notes = assistant.note_manager.get_all_notes()
            new_note = st.text_area("New note:", height=100)
            if st.button("💾 Save Note", use_container_width=True):
                if new_note.strip():
                    assistant.note_manager.save_note(new_note.strip())
                    st.success("Note saved!")
                    st.rerun()
            for note in notes[:10]:
                st.markdown(f'<div class="status-card">📌 {note["content"]}<br><small style="color:#888">{note["created_at"]}</small></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Notes unavailable: {e}")

    with col_remind:
        st.markdown("### ⏰ Reminders & Alarms")
        try:
            assistant = load_assistant()
            reminders = assistant.scheduler_manager.get_pending_reminders()
            remind_text = st.text_input("Reminder text:")
            remind_time = st.time_input("At time:")
            if st.button("➕ Set Reminder", use_container_width=True):
                if remind_text.strip():
                    assistant.scheduler_manager.add_reminder(remind_text.strip(), str(remind_time))
                    st.success("Reminder set!")
                    st.rerun()
            for r in reminders[:10]:
                st.markdown(f'<div class="status-card">⏰ {r["message"]}<br><small style="color:#888">{r["scheduled_time"]}</small></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Reminders unavailable: {e}")

# ══ TAB 4: History ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📜 Conversation History")
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
        for entry in reversed(st.session_state.conversation_history):
            st.markdown(f'<div class="status-card"><small style="color:#888">{entry["time"]} · {entry["action"]}</small><br>👤 {entry["user"]}<br>🤖 {entry["assistant"]}</div>', unsafe_allow_html=True)
    else:
        st.info("No conversation history yet.")
