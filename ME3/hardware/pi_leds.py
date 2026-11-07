"""Three plain LED bulbs (reused from ME2's pi_gpio.py): yellow 'lamp', red and blue, all dimmable (gpiozero PWM).

Wiring (BCM): GPIOxx -> 330 ohm -> LED anode (long leg); cathode -> GND.
  GPIO27 (pin 13) -> YELLOW     GPIO17 (pin 11) -> RED     GPIO22 (pin 15) -> BLUE
No root needed (user in the gpio group).
"""
from __future__ import annotations

# colour name -> (yellow, red, blue) levels 0..1
MIX = {"white": (1, 1, 1), "warm white": (1, 0, 0), "yellow": (1, 0, 0), "orange": (1, 0.6, 0), "green": (1, 0, 0),
       "red": (0, 1, 0), "blue": (0, 0, 1), "cyan": (0, 0, 1), "purple": (0, 1, 1), "magenta": (0, 1, 1),
       "pink": (0.3, 1, 0.3)}


def mix_for(color: str, rgb) -> tuple[float, float, float]:
    if color in MIX:
        return MIX[color]
    r, g, b = (v / 255 for v in rgb)                 # hex colours: nearest mix of the three bulbs
    return (min(r, g) if g > 0.4 else 0.0, r if g < 0.6 else max(0.0, r - g), b)


class LedPanel:
    def __init__(self, yellow: int = 27, red: int = 17, blue: int = 22):
        from gpiozero import PWMLED
        self.leds = [PWMLED(yellow), PWMLED(red), PWMLED(blue)]

    def set(self, on: bool, color: str, rgb, brightness_pct: int) -> None:
        k = brightness_pct / 100 if on else 0.0
        for led, level in zip(self.leds, mix_for(color, rgb)):
            led.value = max(0.0, min(1.0, level * k))

    def close(self) -> None:
        for led in self.leds:
            led.close()
