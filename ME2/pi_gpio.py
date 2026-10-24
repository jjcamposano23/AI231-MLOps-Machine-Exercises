"""Stand-alone Raspberry Pi GPIO actuator for the VCM demo -- 3 plain LEDs.

No root, no addressable-timing, no 5V level issues. Maps commands to 3 LEDs:
  LIGHT_ON       -> yellow 'lamp' on
  LIGHT_OFF      -> all off
  BRIGHTNESS     -> yellow lamp dimmed (PWM: 20 / 60 / 100 %)
  COLOR red/blue -> red / blue LED  (green -> yellow lamp, no green LED)
  OUT_OF_SCOPE   -> quick red flash (rejected)
  any other intent -> brief yellow blink (acknowledge)

Wiring -- BCM pins. Each LED:  GPIO -> 330 ohm -> LED long leg (anode);
LED short leg (cathode, flat side) -> GND rail.
  GPIO27 (pin 13) -> YELLOW  (lamp)
  GPIO17 (pin 11) -> RED
  GPIO22 (pin 15) -> BLUE
"""
import time

from config import OOS

PINS = dict(lamp=27, red=17, blue=22)
BRIGHT = {"20 percent": 0.2, "60 percent": 0.6, "100 percent": 1.0}


class Panel:
    def __init__(self):
        from gpiozero import PWMLED, LED
        self.lamp = PWMLED(PINS["lamp"])          # yellow, dimmable
        self.red = LED(PINS["red"])
        self.blue = LED(PINS["blue"])

    def _all_off(self):
        self.lamp.value = 0.0
        self.red.off()
        self.blue.off()

    def handle(self, intent, slot, conf=1.0):
        if intent == OOS:
            self.red.on(); time.sleep(0.3); self.red.off()
            return
        if intent == "LIGHT_ON":
            self._all_off(); self.lamp.value = 1.0
        elif intent == "LIGHT_OFF":
            self._all_off()
        elif intent == "BRIGHTNESS":
            self._all_off(); self.lamp.value = BRIGHT.get(slot, 0.6)
        elif intent == "COLOR":
            self._all_off()
            if slot == "red":
                self.red.on()
            elif slot == "blue":
                self.blue.on()
            else:                                  # green -> yellow lamp (no green LED)
                self.lamp.value = 1.0
        else:
            self.lamp.value = 1.0; time.sleep(0.12); self.lamp.value = 0.0   # ack blink

    def close(self):
        for led in (self.lamp, self.red, self.blue):
            try:
                led.close()
            except Exception:
                pass


if __name__ == "__main__":          # python pi_gpio.py  (cycle test)
    p = Panel()
    for intent, slot in [("LIGHT_ON", ""), ("BRIGHTNESS", "20 percent"),
                         ("COLOR", "red"), ("COLOR", "blue"), ("LIGHT_OFF", ""),
                         (OOS, "")]:
        print(intent, slot); p.handle(intent, slot); time.sleep(0.8)
    p.close()
