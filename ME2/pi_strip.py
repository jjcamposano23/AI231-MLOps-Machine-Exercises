"""WS2812 strip actuator for the Pi demo (replaces the plain-LED panel / broken ring).

Maps light commands to a short addressable strip:
  LIGHT_ON = white, LIGHT_OFF = off, BRIGHTNESS = dim white, COLOR = red/green/blue,
  OUT_OF_SCOPE = brief red flash, any other intent = brief blue "ack".

Needs rpi_ws281x and ROOT (it uses PWM/DMA):   pip install rpi_ws281x
Run pi_assistant with the venv's python under sudo, e.g.:
  sudo ~/vcm/bin/python pi_assistant.py --onnx slu.onnx --student <id> \
       --wake-gpio 6 --oled --strip 18 --logdir /home/<user>/vcm_benchmark

Wiring:
  strip DIN  -> GPIO18 (pin 12)     strip 5V -> MB102 5V rail     strip GND -> MB102 GND
  tie MB102 GND to a Pi GND pin (COMMON GROUND -- essential).
Do NOT power the strip from the Pi's 5V pin.
POWER: this strip has 88 LEDs; at full white that is ~5 A, but the MB102 only does
~0.5-0.7 A. So we light only the first N_USED LEDs at reduced brightness (current-
limited). To light more, power the strip from a dedicated 5 V/3 A supply (common gnd).
IMPORTANT: GPIO18 PWM clashes with the Pi's onboard audio -- disable it (we use a USB
mic anyway):  add 'dtparam=audio=off' to /boot/firmware/config.txt and reboot.
"""
import time

from config import OOS

N_LEDS = 88      # physical length of the strip
N_USED = 88      # only light this many -- 88 @ full white would brown out the MB102
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
BRIGHT = {"20 percent": 0.2, "60 percent": 0.6, "100 percent": 1.0}


class StripPanel:
    def __init__(self, pin=18, n=N_LEDS, used=N_USED, brightness=90):
        from rpi_ws281x import PixelStrip, Color
        self._Color = Color
        self.n = n
        self.used = min(used, n)          # only illuminate this many (power budget)
        self.strip = PixelStrip(n, pin, 800000, 10, False, brightness, 0)
        self.strip.begin()
        self._fill(0, 0, 0)

    def _fill(self, r, g, b):
        on = self._Color(int(r), int(g), int(b))
        off = self._Color(0, 0, 0)
        for i in range(self.n):
            self.strip.setPixelColor(i, on if i < self.used else off)
        self.strip.show()

    def handle(self, intent, slot, conf=1.0):
        if intent == OOS:
            self._fill(120, 0, 0); time.sleep(0.3); self._fill(0, 0, 0); return
        if intent == "LIGHT_ON":
            self._fill(255, 255, 255)
        elif intent == "LIGHT_OFF":
            self._fill(0, 0, 0)
        elif intent == "BRIGHTNESS":
            v = int(255 * BRIGHT.get(slot, 0.6)); self._fill(v, v, v)
        elif intent == "COLOR":
            self._fill(*COLORS.get(slot, (255, 255, 255)))
        else:
            self._fill(0, 0, 120); time.sleep(0.15); self._fill(0, 0, 0)   # blue ack

    def close(self):
        try:
            self._fill(0, 0, 0)
        except Exception:
            pass


if __name__ == "__main__":          # sudo ~/vcm/bin/python pi_strip.py  (cycle test)
    p = StripPanel()
    for intent, slot in [("LIGHT_ON", ""), ("BRIGHTNESS", "20 percent"),
                         ("COLOR", "red"), ("COLOR", "green"), ("COLOR", "blue"),
                         ("LIGHT_OFF", ""), (OOS, "")]:
        print(intent, slot); p.handle(intent, slot); time.sleep(0.7)
    p.close()
