"""SSD1306 OLED (I2C) display for the VCM demo -- shows the recognized command.

This is the best actuator for showing the VCM covers all 10 command categories:
it prints the recognized intent (+ slot + confidence) for EVERY command, on-device.

Enable I2C once on the Pi:   sudo raspi-config nonint do_i2c 0   (then reboot)
Install the lib:             pip install luma.oled
Wiring (SSD1306 -> Pi):      VCC->3V3 (pin1)  GND->GND  SDA->GPIO2 (pin3)  SCL->GPIO3 (pin5)
Default I2C address 0x3C (0x3D on some boards).
"""
from config import OOS, response_text


def _wrap2(text, width=16):
    """Split text into two display lines of <= width chars (on word boundaries)."""
    words, l1 = text.split(), ""
    i = 0
    while i < len(words) and len((l1 + " " + words[i]).strip()) <= width:
        l1 = (l1 + " " + words[i]).strip(); i += 1
    return (l1 or text[:width]), " ".join(words[i:])[:width]


class OLED:
    def __init__(self, addr=0x3C, port=1):
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306
        self.dev = ssd1306(i2c(port=port, address=addr))
        self._font = None
        try:
            from PIL import ImageFont
            self._font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            pass
        self.show("VCM ready", "listening...")

    def show(self, line1, line2=""):
        from luma.core.render import canvas
        with canvas(self.dev) as draw:
            draw.text((2, 4), str(line1)[:18], fill="white", font=self._font)
            if line2:
                draw.text((2, 34), str(line2)[:18], fill="white", font=self._font)

    def handle(self, intent, slot, conf):
        l1, l2 = _wrap2(response_text(intent, slot))
        self.show(l1, l2)

    def close(self):
        try:
            self.dev.cleanup()
        except Exception:
            pass
