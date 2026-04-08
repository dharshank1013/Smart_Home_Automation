Smart Home Automation AI Assistant
====================================

A conversational AI pipeline for smart home control using locally-hosted
HuggingFace models (Whisper · Phi-4 · Svara TTS) with a Streamlit UI.

FEATURES
--------
1. Voice control (Whisper STT) → lights, fan, AC, door, windows, TV via Home Assistant
2. Play music / videos on YouTube by voice
3. Set alarms and reminders (APScheduler + SQLite)
4. Save and retrieve notes (SQLite)
5. Google Custom Search by voice
6. Text-to-speech responses (Svara TTS / pyttsx3 fallback)
7. Beautiful Streamlit UI at http://localhost:8501

PROJECT STRUCTURE
-----------------
smart_home_automation/
├── main.py                  – Streamlit UI entry point
├── mymodel.py               – Core AI pipeline (STT→LLM→TTS)
├── preprocess.py            – Audio & text pre-processing
├── requirements.txt         – Python dependencies
├── .env.example             – API key template
├── dataset_sample.csv       – Sample intent training data
├── results_output.txt       – Benchmark results
├── architecture_diagram.jpg – System architecture (auto-generated)
├── models/
│   ├── whisper-medium/      ← PASTE your HuggingFace download here
│   ├── phi-4/               ← PASTE your HuggingFace download here
│   └── svara-tts/           ← PASTE your HuggingFace download here
├── utils/
│   ├── home_assistant.py    – HA REST API client
│   ├── media_controller.py  – YouTube / local media
│   ├── google_search.py     – Google Custom Search
│   └── logger.py            – Logging helper
├── database/
│   ├── note_manager.py      – SQLite notes
│   └── scheduler_manager.py – APScheduler alarms & reminders
└── gemini/                  – Reserved for future Gemini integration

QUICK START
-----------

Step 1 – Copy your HuggingFace model folders
  Place the downloaded model folders exactly as:
    models/whisper-medium/   (from openai/whisper-medium)
    models/phi-4/            (from microsoft/phi-4)
    models/svara-tts/        (from ai4bharat/indic-parler-tts or similar)

Step 2 – Install dependencies
  pip install -r requirements.txt

  On Windows, for sounddevice you may also need:
    pip install pipwin && pipwin install pyaudio

Step 3 – Configure API keys (optional but recommended)
  Copy .env.example → .env and fill in:
    HA_URL       – your Home Assistant URL
    HA_TOKEN     – Long-Lived Access Token from HA → Profile
    GOOGLE_API_KEY / GOOGLE_CX – from Google Cloud Console

Step 4 – Run
  streamlit run main.py

  The app opens at http://localhost:8501
  All API keys can also be entered in the sidebar at runtime.

HOME ASSISTANT SETUP
--------------------
1. Install Home Assistant (https://www.home-assistant.io/)
2. Go to Profile → Long-Lived Access Tokens → Create Token
3. Paste the token in the sidebar or .env
4. Update entity IDs in utils/home_assistant.py → DEVICE_MAP to match your HA setup

DEVICE MAP (edit to match your HA entities)
-------------------------------------------
  "lights"   → light.living_room
  "fan"      → fan.bedroom_fan
  "ac"       → climate.living_room_ac
  "door"     → lock.front_door
  "windows"  → cover.living_room_windows
  "tv"       → media_player.living_room_tv

VOICE COMMAND EXAMPLES
----------------------
  "Turn on the lights"
  "Switch off the fan"
  "Set the AC to 22 degrees"
  "Play Coldplay on YouTube"
  "Set alarm at 7 AM"
  "Remind me to take medicine at 8 PM"
  "Take a note: buy groceries tomorrow"
  "Search for weather in Chennai"
  "What is the capital of France?"

TROUBLESHOOTING
---------------
- Models not loading: Make sure all 3 model folders exist under models/
- No audio: Install sounddevice + portaudio (brew/apt/choco install portaudio)
- HA not reachable: Check HA_URL and that HA is on the same network
- TTS silent: pyttsx3 fallback is used; check system audio

PAPER CITATION
--------------
This project was developed as part of:
"Smart Home Automation using Conversational AI"
Pipeline: Whisper (STT) → Phi-4 (NLU/NLG) → Svara TTS (TTS)
