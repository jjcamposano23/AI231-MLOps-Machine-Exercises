"""Offline AEC check: mic = (delayed, quieter copy of Iris's speech) + (a user voice). Echo should mostly vanish."""
import sys, os, time, base64, numpy as np, httpx
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")
from assistant.aec import EchoCanceller
k = os.environ["DEEPGRAM_API_KEY"]
def tts(text, sr, voice):
    r = httpx.post("https://api.deepgram.com/v1/speak", params={"model": voice, "encoding": "linear16", "sample_rate": sr, "container": "none"},
                   json={"text": text}, headers={"Authorization": f"Token {k}"}, timeout=30)
    return np.frombuffer(r.content, dtype=np.int16).astype(np.float32)
iris = tts("The weather in Davao is partly cloudy, thirty one degrees, with a chance of rain later this afternoon.", 24000, "aura-2-pandora-en")
user = tts("Hey, stop. Turn the lights blue instead.", 16000, "aura-2-orion-en")
# echo at the mic: Iris resampled to 16k, delayed 80 ms, at 0.5 gain, plus a little room smear
iris16 = np.interp(np.linspace(0, len(iris) - 1, int(len(iris) * 16000 / 24000)), np.arange(len(iris)), iris)
echo = np.zeros(len(iris16) + 16000 * 2); d = int(0.08 * 16000)
echo[d:d + len(iris16)] += 0.5 * iris16; echo[d + 160:d + 160 + len(iris16)] += 0.15 * iris16
mic = echo.copy(); u0 = int(2.0 * 16000); mic[u0:u0 + len(user)] += user[: len(mic) - u0]
mic = np.clip(mic, -32768, 32767).astype(np.int16)
aec = EchoCanceller(16000, delay_ms=80)
out = bytearray(); spk = iris.astype(np.int16).tobytes(); step_spk = 480 * 2; step_mic = 320 * 2   # 20 ms chunks
mb = mic.tobytes()
for i in range(0, len(mb), step_mic):
    j = i // step_mic
    aec.far_end(spk[j * step_spk:(j + 1) * step_spk], 24000)
    out += aec.near_end(mb[i:i + step_mic])
o = np.frombuffer(bytes(out), dtype=np.int16).astype(np.float32)
def rms(x): return float(np.sqrt(np.mean(x ** 2)) + 1e-9)
seg = slice(int(0.6 * 16000), u0)                  # echo-only window (after AEC has converged a bit)
useg = slice(u0 + 1600, min(u0 + len(user), len(o)))
print(f"echo-only window: mic {rms(mic[seg].astype(np.float32)):.0f} -> out {rms(o[seg]):.0f}  "
      f"({20*np.log10(rms(o[seg])/rms(mic[seg].astype(np.float32))):.1f} dB)")
print(f"user window:      mic {rms(mic[useg].astype(np.float32)):.0f} -> out {rms(o[useg]):.0f}")
import wave
for name, data in (("logs/aec_mic.wav", mic.tobytes()), ("logs/aec_out.wav", bytes(out))):
    with wave.open(name, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(data)
