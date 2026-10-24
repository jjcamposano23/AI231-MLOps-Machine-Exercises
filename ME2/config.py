"""Shared configuration for the Voice Command Model (VCM).

Every other module imports constants from here so the front-end, model,
training, export, inference and benchmark all agree on the same settings.

Two label schemas live here:
  * Option B SLU schema (INTENTS + SLOT_VALUES) -- the class dataset, used by the
    SLU multi-head model (train_slu.py / benchmark_slu.py). THIS IS THE MAIN ONE.
  * Legacy KWS schema (LABELS) -- our earlier 12-class taxonomy, kept so the
    synthetic-data smoke tests and single-head scripts still run.
"""

# --- Audio / feature front-end -------------------------------------------------
SR       = 16000          # sample rate (Hz) -- Option B WAVs are already 16k mono
# CLIP_S=3.5 covers the 99.5th pct of Option B durations: truncates only 0.50%
# of clips (vs 2.84% at 3.0). Matters because slot words often end the phrase.
CLIP_S   = 3.5            # fixed clip length (seconds)
CLIP_LEN = int(SR * CLIP_S)

N_MELS   = 40             # mel bands
N_FFT    = 400            # 25 ms window
WIN      = 400
HOP      = 160            # 10 ms hop
# MelSpectrogram (center=True) produces this many frames for a fixed clip:
N_FRAMES = CLIP_LEN // HOP + 1          # = 351 for CLIP_S=3.5

CONF_THRESHOLD = 0.50     # below this, a prediction is rejected as OUT_OF_SCOPE.
# Chosen from eval_reject.py on the gold test: thr=0.50 rejects ~90% of out-of-scope
# inputs while keeping ~79% of valid commands (label-smoothing deflates confidence,
# so 0.5 is already aggressive). Live-tunable via pi_assistant.py --threshold.

# ============================================================================ #
#  Option B SLU schema  (github.com/markandrian30/AI231  MEX2/OptionB)          #
#  Mirrors the dataset's labels.json (19 intents) + slots.json (6 slotted).     #
# ============================================================================ #
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",          # 13 fixed intents
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    "OUT_OF_SCOPE",                                # 20th: reject class (class gold dataset, Oct 2)
]                                                  # 19 command intents + OUT_OF_SCOPE
N_INTENTS = len(INTENTS)                            # 20
INTENT2IDX = {name: i for i, name in enumerate(INTENTS)}
OOS = "OUT_OF_SCOPE"

# slots.json: intent -> (slot name, [closed-vocab values])
SLOT_SPEC = {
    "TIMER":           ("duration", ["10 seconds", "30 seconds", "1 minute"]),
    "ALARM":           ("time",     ["6 AM", "8 AM", "9 PM"]),
    "TEMPERATURE":     ("degrees",  ["18 degrees", "22 degrees", "26 degrees"]),
    "BRIGHTNESS":      ("percent",  ["20 percent", "60 percent", "100 percent"]),
    "COLOR":           ("color",    ["red", "blue", "green"]),
    "CREATE_REMINDER": ("task",     ["drink water", "study", "exercise"]),
}

# One flat slot-value vocabulary for the slot head. "none" = fixed intents.
NO_SLOT = "none"
SLOT_VALUES = [NO_SLOT] + [v for _, vals in SLOT_SPEC.values() for v in vals]
N_SLOTS = len(SLOT_VALUES)                          # 1 + 18 = 19
SLOT2IDX = {v: i for i, v in enumerate(SLOT_VALUES)}


def slot_index(slot_value):
    """Map a manifest slot_value (may be blank/NaN for fixed intents) to an idx."""
    if not isinstance(slot_value, str) or not slot_value.strip():
        return SLOT2IDX[NO_SLOT]
    return SLOT2IDX.get(slot_value.strip(), SLOT2IDX[NO_SLOT])


def flat_label(intent, slot_value):
    """Reconstruct the KWS-style combined label, e.g. ('ALARM','6 AM')->'ALARM|6 AM'."""
    sv = slot_value if isinstance(slot_value, str) and slot_value.strip() else NO_SLOT
    return intent if sv == NO_SLOT else f"{intent}|{sv}"


# ============================================================================ #
#  Spoken / OLED response for each recognized command (narration).              #
#  "{slot}" is filled with the slot value. WEATHER is a canned on-device answer #
#  (no cloud); TIME uses the local clock.                                       #
# ============================================================================ #
RESPONSES = {
    "LIGHT_ON": "Turning on the lights",
    "LIGHT_OFF": "Turning off the lights",
    "PLAY_MUSIC": "Playing music",
    "PAUSE": "Pausing",
    "STOP": "Stopping the music",
    "NEXT": "Skipping to the next track",
    "VOLUME_UP": "Turning the volume up",
    "VOLUME_DOWN": "Turning the volume down",
    "WEATHER": "Today is sunny, around 30 degrees",
    "TIME": "",                                   # filled with the current time
    "CALL": "Calling your mom",
    "MESSAGE": "Sending your message",
    "LIST_REMINDERS": "Here are your reminders",
    "TIMER": "Timer set for {slot}",
    "ALARM": "Alarm set for {slot}",
    "TEMPERATURE": "Setting the temperature to {slot}",
    "BRIGHTNESS": "Setting the brightness to {slot}",
    "COLOR": "Changing the color to {slot}",
    "CREATE_REMINDER": "I'll remind you to {slot}",
    "OUT_OF_SCOPE": "Sorry, I didn't catch that",
}


def response_text(intent, slot=""):
    """Friendly response string for narration (TTS) + OLED display."""
    if intent == "TIME":
        import datetime
        return datetime.datetime.now().strftime("It is %I:%M %p")
    t = RESPONSES.get(intent, intent)
    return t.replace("{slot}", (slot or "").strip()).strip()


# ============================================================================ #
#  Legacy KWS schema (12 classes) -- kept for the synthetic smoke-test path.    #
# ============================================================================ #
LABELS = [
    "play_music", "query_search", "lights_onoff", "lights_dim",
    "set_timer", "set_alarm", "set_thermostat", "media_control",
    "reminders", "calls_messaging", "_unknown_", "_silence_",
]
N_CLASSES = len(LABELS)
LABEL2IDX = {name: i for i, name in enumerate(LABELS)}
REJECT = {"_unknown_", "_silence_"}
