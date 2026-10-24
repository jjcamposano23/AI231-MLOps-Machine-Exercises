# Raspberry Pi 4 — VCM demo setup (end to end)

Headless setup (no monitor/keyboard — driven from your laptop over SSH).
Hostname used below: **`vcm`**. Replace `<user>` with the username you set.

---

## 0. Gear checklist
- [x] Raspberry Pi 4 (8 GB) + heatsink
- [x] 32 GB microSD card
- [x] USB microphone
- [ ] 5 V **3 A** USB-C power supply (a 2 A phone charger may cause undervoltage/crashes)
- [ ] Jumper wires + breadboard + LEDs + 330 Ω resistors (from your Arduino kit)
- [ ] KY-038 sound sensor
- Files to copy (7): `slu.onnx`, `pi_assistant.py`, `config.py`, `features.py`,
  `pi_gpio.py`, `pi_wake.py`, `setup_pi.sh`

---

## 1. Flash the microSD (on your laptop)
1. Install **Raspberry Pi Imager** (raspberrypi.com/software).
2. **Device:** Raspberry Pi 4. **OS:** Raspberry Pi OS (64-bit). **Storage:** your 32 GB card.
3. Click **Next → Edit Settings** (the ⚙ gear). This is the critical headless step:
   - Set **hostname**: `vcm`
   - Set **username + password** (remember these — this is `<user>`)
   - **Configure wireless LAN**: your WiFi SSID + password, country `PH`
   - **Set locale**: timezone `Asia/Manila`
   - Tab **Services → Enable SSH → Use password authentication**
4. **Save → Write.** Wait for write + verify, then eject.

> Without SSH + WiFi preset here, a headless Pi has no way to be reached. Double-check them.

---

## 2. First boot + connect
1. Insert the card, plug in the USB mic, connect power. First boot takes ~1–2 min.
2. From your laptop (same WiFi):
   ```
   ssh <user>@vcm.local
   ```
   Accept the fingerprint, enter your password.
3. If `vcm.local` won't resolve, find the Pi's IP (your router's device list, or `ping vcm.local`) and use `ssh <user>@<IP>`.
4. Check power health:
   ```
   vcgencmd get_throttled
   ```
   `throttled=0x0` = good. Anything else = undervoltage → use a 3 A supply.

---

## 3. Copy the files to the Pi (from your laptop)
In your laptop's ME2 folder:
```
scp slu.onnx pi_assistant.py config.py features.py pi_gpio.py pi_wake.py setup_pi.sh <user>@vcm.local:~/
```

---

## 4. Install everything (on the Pi)
```
bash setup_pi.sh
source ~/vcm/bin/activate
```
The script makes a venv at `~/vcm`, installs deps, prints your audio devices, and
verifies `slu.onnx` loads. **Note the USB mic's index** in the device list.

> Every new SSH session: run `source ~/vcm/bin/activate` first.

---

## 5. Test recognition (mic only, no wiring yet)
```
python pi_assistant.py --onnx slu.onnx --student <studentID> --no-wake --threshold 0.4
```
Say: "turn on the lights", "set a timer for 30 seconds", "change the color to red".
Watch the JSON decisions. Tune `--threshold` (0.3–0.5): lower = accepts more, higher = rejects more.

**If it hears nothing** (everything OUT_OF_SCOPE or no output): the USB mic isn't the
default input. Check `arecord -l`, and tell me the device index — I'll add a `--device` flag.

---

## 6. Wire + test the LEDs (standalone actuator)
Power off first (`sudo poweroff`) before wiring. Pins are **BCM → physical**:

| role | BCM | physical pin |
|---|---|---|
| ack LED | GPIO17 | 11 |
| lamp LED | GPIO27 | 13 |
| red | GPIO22 | 15 |
| green | GPIO23 | 16 |
| blue | GPIO24 | 18 |
| reject LED | GPIO25 | 22 |

Each LED: GPIO pin → **330 Ω resistor** → LED anode (long leg); LED cathode → any GND
(pins 6, 9, 14, 20, 25, 30, 34, 39). Power back on, then:
```
python pi_gpio.py        # self-test: cycles lamp / brightness / colors / reject
```
Then run with LEDs live:
```
python pi_assistant.py --onnx slu.onnx --student <studentID> --no-wake --threshold 0.4 --gpio
```

> If you see `No access to /dev/gpiomem`: run `sudo usermod -aG gpio <user>` then re-login,
> or prefix the command with `sudo ~/vcm/bin/python ...`.

---

## 7. Wire + test the KY-038 wake board
| KY-038 | Pi |
|---|---|
| VCC | **3V3** (pin 1) — keeps D0 at 3.3 V, safe for the Pi |
| GND | GND |
| D0 | **GPIO6** (physical pin 31) |

Test the sensor alone first:
```
python pi_wake.py 6
```
Clap → prints `sound!`. Turn the onboard pot until it fires on a clap/voice but stays
quiet in silence. Then the full wake demo:
```
python pi_assistant.py --onnx slu.onnx --student <studentID> --wake-gpio 6 --gpio --threshold 0.4
```
Flow: clap/"hey" → `[wake] listening...` → say a command → LEDs react + logged.

---

## 8. (Optional) ESP32 second display
Plug the ESP32 by USB, find its port (`ls /dev/ttyUSB* /dev/ttyACM*`), add `--port`:
```
python pi_assistant.py --onnx slu.onnx --student <studentID> --wake-gpio 6 --gpio --port /dev/ttyUSB0 --threshold 0.4
```

---

## 9. Run the class benchmark
```
cd ~ && git clone https://github.com/airimonda/vcm-benchmark
```
Read its README and follow the live-Pi instructions (paste it to me for exact commands).
Decisions are already logged by `pi_assistant.py` to `~/vcm_benchmark/*.log` in the
format `{"intent","slot","infer_ms","audio_ms"}`.

---

## 10. Demo-day quick reference
```
ssh <user>@vcm.local
source ~/vcm/bin/activate
python pi_assistant.py --onnx slu.onnx --student <studentID> --wake-gpio 6 --gpio --threshold 0.4
```
Speak prompts from your phone/laptop at the Pi's mic (the "two-device" trick).

### Troubleshooting
- **Can't SSH / no `vcm.local`** → use the IP from your router; confirm WiFi was set in Imager.
- **Random freezes / rainbow square** → undervoltage; use a real 5 V 3 A USB-C supply.
- **`onnxruntime` won't install** → confirm 64-bit OS: `uname -m` must say `aarch64`.
- **No audio / mic ignored** → `arecord -l`; make the USB mic the default, or ask for `--device`.
- **GPIO permission error** → `sudo usermod -aG gpio <user>`, re-login.
- **Commands misread** → speak close to the mic; nudge `--threshold` down to 0.3.
