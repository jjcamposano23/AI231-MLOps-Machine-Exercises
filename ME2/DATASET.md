# ME2 VCM Dataset — Recording Spec (class contract)

Read this before recording. Follow it exactly so everyone's clips merge into one
trainable dataset. Questions / speaker-id assignments: **[coordinator name here]**.

## What we're building

A spoken-command dataset for **12 classes** — the 10 smart-device commands plus
two rejection classes. One command per clip. The phrasings are fixed in
[`prompts.json`](prompts.json) (the single source of truth).

| Class | What to say (see prompts.json for the full list) |
|---|---|
| `play_music` | "play music", "put on a song", … |
| `query_search` | "what's the weather", "what time is it", … |
| `lights_onoff` | "turn on the lights", "lights off", … |
| `lights_dim` | "dim the lights to fifty percent", "brighten the lights", … |
| `set_timer` | "set a timer for five minutes", … |
| `set_alarm` | "set an alarm for seven a m", … |
| `set_thermostat` | "set the temperature to twenty four degrees", "turn up the heat", … |
| `media_control` | "pause", "stop", "next", "volume up", "louder", … |
| `reminders` | "remind me to call mom", "what are my reminders", … |
| `calls_messaging` | "call mom", "send a message", … |
| `_unknown_` | off-topic speech: "tell me a joke", counting, random sentences |
| `_silence_` | **no speech** — room tone / ambient only |

## Format (non-negotiable — the trainer depends on it)

- **WAV, mono, 16 kHz.** (Record higher and the validator will downsample.)
- **~1–2 s per clip**, one command each, trimmed.
- **Folder per class:** `data/<label>/`
- **Filename:** `spk<ID>_<label>_<NNN>.wav` — e.g. `spk07_lights_onoff_003.wav`.
  The `spk<ID>` prefix is **mandatory**: the split keeps each speaker entirely in
  train **or** val **or** test, so a missing/duplicated id corrupts the benchmark.
- Get a **unique `<ID>`** from the coordinator (use your student number).

## Easiest path: use the recorder

```bash
python record.py --speaker 07 --takes 3 --device-name "USB mic"
```
It reads `prompts.json`, walks you through every phrase, enforces the format and
naming, and logs `metadata.csv`. Controls: `[Enter]`=record, `s`=skip, `q`=quit;
then `[Enter]`=keep, `r`=redo.

## Your quota

For each class, record **every phrasing × 3 takes**. That's ~180 clips/person and
takes ~20–30 min. Team target: **≥ 500 clips/class and ≥ 15 speakers**, balanced.

> **Record extra `set_thermostat` and `calls_messaging`.** Public datasets barely
> cover phone calls ("call mom") and exact-degree thermostat commands, so these
> two classes lean on our own voices — aim for **1.5× the usual takes** on them.

## Record with variety (this is what makes it generalize)

Spread your takes across:
- **Distance:** some near (~30 cm), some far (~2–3 m).
- **Device:** phone / laptop, and **at least a few on the actual demo mic**.
- **Environment:** quiet, plus some with fan/aircon/TV background.
- **Delivery:** normal, a few fast, a few slow.

You do **not** need to add noise manually — the trainer augments with noise +
SpecAugment automatically. Just give clean, varied takes (and a few noisy ones).

## Before you submit: validate

```bash
python validate_dataset.py --data data --fix
```
Fixes sample rate / channels, flags silent or clipped clips and bad filenames,
and writes `manifest.csv` with per-class and per-speaker counts. Fix anything
flagged, then hand off your `data/<label>/` folders + your `metadata.csv`.

## Consent & privacy

Voice is personal data. By recording you consent to your (anonymized) clips being
used for this class exercise. Use `spk<ID>` only — **no real names in filenames or
prompts**, and no sensitive content.

## Sources beyond our own voices (optional, to scale up)

Public corpora that map to these commands (verify each license before use):
**SLURP** (IoT/media/alarm/query/lists/calls), **Fluent Speech Commands**
(smart-home phrasings), **Google Speech Commands v2** (media words, digits,
background noise), **timers-and-such** (spoken numbers), **MUSAN / ESC-50** (noise
+ `_silence_`), **Common Voice / LibriSpeech** (`_unknown_` negatives, speaker
diversity). Multi-voice **TTS** can pad thin classes (dataset prep only — never at
inference).

### Merging public data — use the label map

These datasets use their own (finer or single-word) labels, so **don't drop them
in raw** — map each source label onto our 12 classes with
[`labelmap.json`](labelmap.json), which collapses sub-intents (e.g. FSC
`increase_heat`/`decrease_heat` → `set_thermostat`) and sends anything outside our
10 commands to `_unknown_`. Then:

1. Look up each clip's dataset label in `labelmap.json` → get our class.
2. Convert to **16 kHz mono WAV** and file it under `data/<our_class>/`, keeping a
   `spk<sourceID>_<our_class>_NNN.wav` name (prefix so speakers stay grouped).
3. Run `validate_dataset.py` on the merged tree like any other data.

`labelmap.json` is a starter template — confirm each dataset's exact label strings
against its docs (they drift between releases) and extend the map as you import.
