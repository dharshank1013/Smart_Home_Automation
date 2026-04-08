"""
mymodel.py - Core pipeline orchestrating Whisper + Phi-4 + Svara TTS
"""

import os
import json
import re
import time
import tempfile
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# ── Sub-modules ───────────────────────────────────────────────────────────────
from utils.logger import get_logger
from utils.home_assistant import HomeAssistantClient
from utils.media_controller import MediaController
from utils.google_search import GoogleSearchClient
from database.note_manager import NoteManager
from database.scheduler_manager import SchedulerManager

logger = get_logger(__name__)

# ── Model paths (relative to project root – copy your HuggingFace downloads here) ──
BASE_DIR   = Path(__file__).parent
MODEL_ROOT = BASE_DIR / "models"

WHISPER_PATH = MODEL_ROOT / "whisper-medium"
PHI4_PATH    = MODEL_ROOT / "phi-4"
SVARA_PATH   = MODEL_ROOT / "svara-tts"


# ═════════════════════════════════════════════════════════════════════════════
class SmartHomeAssistant:
    """
    Main pipeline:
      Audio → Whisper (STT) → Phi-4 (intent + response) → Svara TTS (speech)
      + Home Assistant · YouTube · Scheduler · SQLite Notes · Google Search
    """

    def __init__(self):
        logger.info("Initialising SmartHomeAssistant…")
        self._load_stt()
        self._load_llm()
        self._load_tts()

        # Aux services
        self.ha_client      = HomeAssistantClient()
        self.media          = MediaController()
        self.google         = GoogleSearchClient()
        self.note_manager   = NoteManager()
        self.scheduler_mgr  = SchedulerManager()
        # alias used by main.py
        self.scheduler_manager = self.scheduler_mgr

        # Start background scheduler
        self.scheduler_mgr.start()
        logger.info("SmartHomeAssistant ready ✅")

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _load_stt(self):
        """Load Whisper medium for speech-to-text."""
        try:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            logger.info(f"Loading Whisper from {WHISPER_PATH} …")
            self.whisper_processor = WhisperProcessor.from_pretrained(str(WHISPER_PATH))
            self.whisper_model     = WhisperForConditionalGeneration.from_pretrained(str(WHISPER_PATH))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model.to(self.device)
            self.whisper_model.eval()
            logger.info("Whisper loaded ✅")
        except Exception as e:
            logger.error(f"Whisper load failed: {e}")
            self.whisper_processor = None
            self.whisper_model     = None

    def _load_llm(self):
        """Load Phi-4 for intent classification + response generation."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Loading Phi-4 from {PHI4_PATH} …")
            self.phi_tokenizer = AutoTokenizer.from_pretrained(
                str(PHI4_PATH), trust_remote_code=True
            )
            self.phi_model = AutoModelForCausalLM.from_pretrained(
                str(PHI4_PATH),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                self.phi_model.to("cpu")
            self.phi_model.eval()
            logger.info("Phi-4 loaded ✅")
        except Exception as e:
            logger.error(f"Phi-4 load failed: {e}")
            self.phi_tokenizer = None
            self.phi_model     = None

    def _load_tts(self):
        """Load Svara TTS for text-to-speech."""
        try:
            from transformers import VitsModel, AutoTokenizer
            import torch

            logger.info(f"Loading Svara TTS from {SVARA_PATH} …")
            self.tts_tokenizer = AutoTokenizer.from_pretrained(str(SVARA_PATH))
            self.tts_model     = VitsModel.from_pretrained(str(SVARA_PATH))
            self.tts_model.eval()
            logger.info("Svara TTS loaded ✅")
        except Exception as e:
            logger.warning(f"Svara TTS load failed (will use pyttsx3 fallback): {e}")
            self.tts_tokenizer = None
            self.tts_model     = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update_config(self, ha_url: str, ha_token: str,
                      google_api_key: str, google_cx: str):
        self.ha_client.update(ha_url, ha_token)
        self.google.update(google_api_key, google_cx)

    def process_voice_command(self) -> Dict[str, Any]:
        """Record audio → transcribe → process."""
        audio = self._record_audio()
        text  = self._transcribe(audio)
        logger.info(f"Transcription: {text}")
        result = self.process_command(text)
        result["transcription"] = text
        self._speak(result["response"])
        return result

    def process_command(self, text: str) -> Dict[str, Any]:
        """Text → intent → action → response."""
        intent_data = self._classify_intent(text)
        intent      = intent_data.get("intent", "general")
        entities    = intent_data.get("entities", {})

        logger.info(f"Intent: {intent} | Entities: {entities}")

        action_map = {
            "home_control":  self._handle_home_control,
            "media":         self._handle_media,
            "alarm":         self._handle_alarm,
            "cancel_alarm":  self._handle_cancel_alarm,
            "reminder":      self._handle_reminder,
            "note":          self._handle_note,
            "search":        self._handle_search,
            "time_check":    self._handle_time_check,
            "general":       self._handle_general,
        }

        handler  = action_map.get(intent, self._handle_general)
        response = handler(text, entities)

        return {
            "response":      response,
            "action":        intent,
            "device_states": self.ha_client.get_device_states(),
        }

    # ── Intent classification (Phi-4) ─────────────────────────────────────────

    def _classify_intent(self, text: str) -> Dict[str, Any]:
        SYSTEM = """You are an intent classifier for a smart home assistant.
Classify the user command into exactly ONE of these intents:
- home_control  (lights, fan, AC, door, windows, TV on/off/set)
- media         (play music, YouTube, video)
- alarm         (set alarm, wake me up at)
- reminder      (remind me, set reminder)
- note          (take note, write down, remember that)
- search        (search, google, what is, tell me about)
- general       (anything else – conversation, jokes, help)

Respond ONLY with valid JSON like:
{"intent": "home_control", "entities": {"device": "lights", "action": "on"}}"""

        if self.phi_model and self.phi_tokenizer:
            try:
                prompt  = f"<|system|>{SYSTEM}<|end|><|user|>{text}<|end|><|assistant|>"
                inputs  = self.phi_tokenizer(prompt, return_tensors="pt").to(self.device)
                with __import__("torch").no_grad():
                    outputs = self.phi_model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.phi_tokenizer.eos_token_id,
                    )
                raw = self.phi_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                # extract first JSON object
                m = re.search(r'\{.*?\}', raw, re.DOTALL)
                if m:
                    return json.loads(m.group())
            except Exception as e:
                logger.error(f"Phi-4 inference error: {e}")

        # ── Rule-based fallback ──────────────────────────────────────────────
        return self._rule_based_intent(text)

    def _rule_based_intent(self, text: str) -> Dict[str, Any]:
        t = text.lower()

        # ── Cancel alarm (must check before alarm set) ────────────────────────
        if any(w in t for w in ["cancel alarm", "cancel the alarm", "delete alarm",
                                  "remove alarm", "stop alarm", "clear alarm"]):
            return {"intent": "cancel_alarm", "entities": {}}

        # ── Time check ────────────────────────────────────────────────────────
        if any(w in t for w in ["what time", "what is the time", "current time",
                                  "what's the time", "tell me the time", "time now",
                                  "நேரம்", "என்ன நேரம்"]):
            return {"intent": "time_check", "entities": {}}

        # ── Note (check BEFORE home_control to avoid "ac" in long sentences) ──
        if any(w in t for w in ["take a note", "take note", "write down",
                                  "remember that", "note that", "குறிப்பு"]):
            return {"intent": "note", "entities": {"content": t}}

        # ── Home control — room-aware (English + Tamil keywords) ──────────────
        rooms   = ["bedroom", "living room", "kitchen", "bathroom", "study room", "study",
                   "படுக்கை", "சமையல்", "கழிவறை"]
        # Only match standalone device words, not substrings inside other words
        devices_en = ["lights", "light", "fan", "ac", "air conditioner",
                      "door", "window", "windows", "tv", "television"]
        devices_ta = ["விளக்கு", "விளக்கை", "மின்விசிறி", "கதவு", "ஜன்னல்",
                      "லைட்", "லைட்டை", "லைட்டா", "மின் விளக்கு", "மின் விலக்கை"]
        actions_en = ["on", "off", "open", "close", "turn on", "turn off",
                      "switch on", "switch off", "toggle"]
        actions_ta = ["போடு", "அணை", "அணைத்து", "ஆண்", "ஆன்", "திற", "மூடு",
                      "பண்ணு", "வை", "வையுங்க"]

        # Check Tamil device + action combos
        ta_device = next((d for d in devices_ta if d in t), None)
        ta_action = next((a for a in actions_ta if a in t), None)
        if ta_device and ta_action:
            # Determine on/off from Tamil action
            off_words = ["அணை", "அணைத்து", "மூடு"]
            action = "off" if any(w in t for w in off_words) else "on"
            room = next((r for r in rooms if r in t), "")
            device_key = f"{room} light".strip() if room else "light"
            return {"intent": "home_control",
                    "entities": {"device": device_key, "action": action}}

        # Check English device + action combos (use word boundary to avoid "ac" in words)
        for d in devices_en:
            # Use word boundary check to avoid matching "ac" inside "transportation"
            pattern = r'\b' + re.escape(d) + r'\b'
            if re.search(pattern, t):
                for a in actions_en:
                    if re.search(r'\b' + re.escape(a) + r'\b', t):
                        room = next((r for r in rooms if r in t), "")
                        device_key = f"{room} {d}".strip() if room else d
                        return {"intent": "home_control",
                                "entities": {"device": device_key, "action": a}}

        # ── Media (English + Tamil) ───────────────────────────────────────────
        if any(w in t for w in ["play", "youtube", "music", "song", "video",
                                  "பாடல்", "பாட்டு", "திரைப்படம்"]):
            # Keep the actual song/artist — only strip command words
            query = re.sub(
                r'\b(play|on youtube|youtube|music|video|song|for me|please)\b',
                '', text, flags=re.I).strip()
            query = re.sub(r'\s+', ' ', query).strip()
            return {"intent": "media", "entities": {"query": query or text}}

        # ── Alarm ─────────────────────────────────────────────────────────────
        if any(w in t for w in ["alarm", "wake me", "wake up", "அலாரம்"]):
            time_m = re.search(
                r'(\d{1,2}\s*[:.]\s*\d{2}\s*(?:am|pm)?'
                r'|\d{1,2}\s*(?:am|pm)'
                r'|\d{1,2}\s+\d{2}\s*(?:am|pm)?)', t, re.I)
            return {"intent": "alarm",
                    "entities": {"time": time_m.group(1).strip() if time_m else ""}}

        # ── Reminder ──────────────────────────────────────────────────────────
        if any(w in t for w in ["remind", "reminder", "நினைவூட்டு"]):
            time_m = re.search(r'at\s+(\d{1,2}[.: ]\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))', t, re.I)
            msg    = re.sub(r'remind me|reminder|set a|set|at \d.*', '', t, flags=re.I).strip()
            return {"intent": "reminder",
                    "entities": {"message": msg or t,
                                 "time": time_m.group(1).strip() if time_m else ""}}

        # ── Search ────────────────────────────────────────────────────────────
        if any(w in t for w in ["search", "google", "what is", "what are",
                                  "who is", "who was", "tell me about",
                                  "explain", "define", "meaning of", "meant by",
                                  "என்ன", "யார்", "எப்படி"]):
            return {"intent": "search", "entities": {"query": t}}

        return {"intent": "general", "entities": {}}

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_home_control(self, text: str, entities: Dict) -> str:
        device = entities.get("device", "")
        action = entities.get("action", "")
        return self.ha_client.control_device(device, action, text)

    def _handle_media(self, text: str, entities: Dict) -> str:
        query = entities.get("query", text)
        return self.media.play_youtube(query)

    def _handle_alarm(self, text: str, entities: Dict) -> str:
        alarm_time = entities.get("time", "").strip()
        if not alarm_time:
            m = re.search(
                r'(\d{1,2}\s*[:.]\s*\d{2}\s*(?:am|pm)?'
                r'|\d{1,2}\s*(?:am|pm)'
                r'|\d{1,2}\s+\d{2}\s*(?:am|pm)?)',
                text, re.I)
            alarm_time = m.group(1).strip() if m else ""

        if not alarm_time:
            return "⚠️ Couldn't understand the time. Try: 'Set alarm at 7:30 AM' or 'Wake me up at 6 AM'."

        result = self.scheduler_mgr.set_alarm(alarm_time)
        if isinstance(result, tuple):
            parsed, display = result
            return f"⏰ Alarm set for {display}. I'll alert you on time!"
        return f"⚠️ {result}"

    def _handle_cancel_alarm(self, text: str, entities: Dict) -> str:
        # Check if a specific time was mentioned
        time_m = re.search(
            r'(\d{1,2}\s*[:.]\s*\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))',
            text, re.I)
        alarm_time = time_m.group(1).strip() if time_m else None
        result = self.scheduler_mgr.cancel_alarm(alarm_time)
        return f"🔕 {result}"

    def _handle_reminder(self, text: str, entities: Dict) -> str:
        remind_time = entities.get("time", "").strip()
        msg         = entities.get("message", "").strip()

        if not remind_time:
            m = re.search(r'at\s+(\d{1,2}[.: ]\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))', text, re.I)
            remind_time = m.group(1).strip() if m else ""
        if not msg:
            msg = re.sub(r'remind me|reminder|set a|set|at \d.*', '', text, flags=re.I).strip()

        parsed_time = self.scheduler_mgr.add_reminder(msg or text, remind_time or "00:00")
        display_time = remind_time or parsed_time or "soon"
        return f"🔔 Reminder set: '{msg or text}' at {display_time}."

    def _handle_note(self, text: str, entities: Dict) -> str:
        content = entities.get("content", text)
        content = re.sub(
            r'\b(take a note|take note|write down|remember that|note that|note:)\b',
            '', content, flags=re.I).strip()
        self.note_manager.save_note(content)
        return f"📝 Note saved: '{content}'"

    def _handle_time_check(self, text: str, entities: Dict) -> str:
        now       = datetime.now()
        time_12h  = now.strftime("%I:%M %p")
        date_full = now.strftime("%A, %d %B %Y")
        return f"🕐 Current Time: {time_12h}\n📅 Date: {date_full}"

    def _handle_search(self, text: str, entities: Dict) -> str:
        query = entities.get("query", text)
        # Strip question/command prefixes to get the actual search term
        query = re.sub(
            r'^(search for|search|google|tell me about|what is|what are'
            r'|who is|who was|explain|define|meaning of|meant by'
            r'|what\'s|whats|what does|how does|how is)\s+',
            '', query.strip(), flags=re.I)
        # Also strip trailing punctuation
        query = query.rstrip('?.!').strip()
        if not query:
            return "Please tell me what to search for."
        return self.google.search(query)

    def _handle_general(self, text: str, entities: Dict) -> str:
        if self.phi_model and self.phi_tokenizer:
            try:
                import torch
                SYSTEM = "You are a helpful smart home AI assistant. Be concise and friendly. Do not use markdown formatting."
                prompt = f"<|system|>{SYSTEM}<|end|><|user|>{text}<|end|><|assistant|>"
                inputs = self.phi_tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.phi_model.generate(
                        **inputs, max_new_tokens=200, temperature=0.7,
                        do_sample=True, pad_token_id=self.phi_tokenizer.eos_token_id
                    )
                return self.phi_tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            except Exception as e:
                logger.error(f"General response error: {e}")
        return (
            "I'm your Smart Home AI Assistant! I can:\n"
            "• Control devices — say 'Turn on bedroom light'\n"
            "• Play music — say 'Play AR Rahman songs'\n"
            "• Set alarms — say 'Set alarm at 7 AM'\n"
            "• Reminders — say 'Remind me to take medicine at 8 PM'\n"
            "• Take notes — say 'Take a note buy groceries'\n"
            "• Search web — say 'What is machine learning?'\n"
            "• Check time — say 'What time is it?'"
        )

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _record_audio(self, duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
        try:
            import sounddevice as sd
            logger.info(f"Recording {duration}s …")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return np.zeros(16000 * 5, dtype=np.float32)

    def record_audio_stream(self, stop_event, sample_rate: int = 16000) -> np.ndarray:
        """Record audio until stop_event is set, then return the captured numpy array."""
        import sounddevice as sd
        chunks = []
        def callback(indata, frames, time_info, status):
            chunks.append(indata.copy())
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
            logger.info("Streaming recording started…")
            stop_event.wait()
        logger.info("Streaming recording stopped.")
        if chunks:
            return np.concatenate(chunks, axis=0).flatten()
        return np.zeros(sample_rate, dtype=np.float32)

    def process_voice_from_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe pre-recorded audio → process command → speak response."""
        text = self._transcribe(audio)
        logger.info(f"Transcription: {text}")
        result = self.process_command(text)
        result["transcription"] = text
        self._speak(result["response"])
        return result

    def _transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self.whisper_model is None:
            return ""
        try:
            import torch
            inputs = self.whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            with torch.no_grad():
                # No forced_decoder_ids → Whisper auto-detects language (Tamil, Hindi, English, etc.)
                predicted_ids = self.whisper_model.generate(
                    input_features,
                    task="transcribe",
                )
            return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _speak(self, text: str):
        """Stop any current TTS, then start new one."""
        self.stop_tts()
        self._tts_stop_event = threading.Event()
        self._tts_thread = threading.Thread(
            target=self._speak_async, args=(text, self._tts_stop_event), daemon=True
        )
        self._tts_thread.start()

    def stop_tts(self):
        """Interrupt TTS immediately — works for both sounddevice and pyttsx3."""
        # Signal the async thread to stop
        if hasattr(self, "_tts_stop_event") and self._tts_stop_event:
            self._tts_stop_event.set()
        # Kill sounddevice playback instantly
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        # Kill pyttsx3 subprocess if running
        if hasattr(self, "_tts_proc") and self._tts_proc:
            try:
                self._tts_proc.terminate()
                self._tts_proc = None
            except Exception:
                pass
        # Wait briefly for thread to exit
        if hasattr(self, "_tts_thread") and self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=0.5)

    def _speak_async(self, text: str, stop_event: threading.Event):
        # Strip markdown so TTS doesn't say "asterisk"
        clean = re.sub(r'[*_`#>\-]', '', text)
        clean = re.sub(r'\s+', ' ', clean).strip()

        if stop_event.is_set():
            return

        # ── Svara TTS (sounddevice) ───────────────────────────────────────────
        if self.tts_model and self.tts_tokenizer:
            try:
                import torch
                import sounddevice as sd
                inputs = self.tts_tokenizer(clean, return_tensors="pt")
                with torch.no_grad():
                    output = self.tts_model(**inputs).waveform
                waveform = output.squeeze().numpy()
                if stop_event.is_set():
                    return
                sample_rate = self.tts_model.config.sampling_rate
                sd.play(waveform, samplerate=sample_rate)
                # Poll instead of sd.wait() so we can interrupt mid-playback
                chunk = 0.05  # check every 50ms
                total = len(waveform) / sample_rate
                elapsed = 0.0
                while elapsed < total:
                    if stop_event.is_set():
                        sd.stop()
                        return
                    time.sleep(chunk)
                    elapsed += chunk
                sd.stop()
                return
            except Exception as e:
                logger.warning(f"Svara TTS error, falling back: {e}")

        if stop_event.is_set():
            return

        # ── pyttsx3 fallback — run in subprocess so we can kill it instantly ──
        try:
            import subprocess, sys
            script = (
                "import pyttsx3, sys\n"
                "engine = pyttsx3.init()\n"
                f"engine.say({repr(clean)})\n"
                "engine.runAndWait()\n"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._tts_proc = proc
            # Poll for stop signal while subprocess runs
            while proc.poll() is None:
                if stop_event.is_set():
                    proc.terminate()
                    self._tts_proc = None
                    return
                time.sleep(0.05)
            self._tts_proc = None
        except Exception as e:
            logger.error(f"TTS fallback failed: {e}")
