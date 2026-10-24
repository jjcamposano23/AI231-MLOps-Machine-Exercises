"""TM1637 4-digit 7-segment display for the VCM demo -- shows the NUMERIC slot value
(timer seconds, alarm time, temperature, brightness %). Complements the OLED's text.

Bit-banged over two GPIO lines with lgpio -- no extra library, no root.
Wiring (TM1637 -> Pi):  VCC->3V3  GND->GND  CLK->GPIO20 (pin38)  DIO->GPIO21 (pin40)
Power at 3V3 (not 5V) so the DIO line stays 3.3 V-safe for the Pi.
Use with pi_assistant:  --tm1637 20,21
"""
import threading
import time

from config import OOS

_SEG = [0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f]   # digits 0-9
_BLANK = 0x00
_DELAY = 1e-5


class TM1637:
    def __init__(self, clk=20, dio=21, brightness=7):
        import lgpio
        self._lg = lgpio
        self.h = lgpio.gpiochip_open(0)
        self.clk, self.dio = clk, dio
        lgpio.gpio_claim_output(self.h, clk, 1)
        lgpio.gpio_claim_output(self.h, dio, 1)
        self.brightness = brightness & 0x7
        self._lock = threading.Lock()
        self._timer_thread = None
        self._timer_stop = threading.Event()
        self.clear()

    def _w(self, pin, val):
        self._lg.gpio_write(self.h, pin, val)
        time.sleep(_DELAY)

    def _start(self):
        self._w(self.dio, 0); self._w(self.clk, 0)

    def _stop(self):
        self._w(self.clk, 0); self._w(self.dio, 0)
        self._w(self.clk, 1); self._w(self.dio, 1)

    def _write_byte(self, b):
        for _ in range(8):
            self._w(self.clk, 0)
            self._w(self.dio, b & 1)
            b >>= 1
            self._w(self.clk, 1)
        self._w(self.clk, 0)         # 9th clock = ACK (DIO released high)
        self._w(self.dio, 1)
        self._w(self.clk, 1)
        self._w(self.clk, 0)

    def _show(self, segs, colon=False):
        segs = list(segs)
        if colon and len(segs) >= 2:
            segs[1] |= 0x80
        with self._lock:                            # serialize main-thread + countdown-thread writes
            self._start(); self._write_byte(0x40); self._stop()             # auto-increment
            self._start(); self._write_byte(0xC0)                           # address 0
            for s in segs:
                self._write_byte(s)
            self._stop()
            self._start(); self._write_byte(0x88 | self.brightness); self._stop()  # display on

    def clear(self):
        self._show([_BLANK] * 4)

    def number(self, n, colon=False):
        n = max(0, min(9999, int(n)))
        d = [(n // 1000) % 10, (n // 100) % 10, (n // 10) % 10, n % 10]
        segs = [_SEG[x] for x in d]
        lead = True                                   # blank leading zeros
        for i in range(3):
            if lead and d[i] == 0:
                segs[i] = _BLANK
            else:
                lead = False
        self._show(segs, colon=colon)

    def _show_mmss(self, total):
        total = max(0, min(5999, int(total)))
        m, s = divmod(total, 60)
        v = (m % 100) * 100 + s
        d = [(v // 1000) % 10, (v // 100) % 10, (v // 10) % 10, v % 10]
        self._show([_SEG[x] for x in d], colon=True)         # MM:SS

    def _run_countdown(self, seconds, stop):
        for t in range(seconds, -1, -1):
            if stop.is_set():
                return
            self._show_mmss(t)
            for _ in range(10):                              # ~1 s, but check stop every 0.1 s
                if stop.is_set():
                    return
                time.sleep(0.1)
        for _ in range(6):                                   # done -> flash 00:00
            if stop.is_set():
                return
            self.clear(); time.sleep(0.3)
            self._show_mmss(0); time.sleep(0.3)

    def start_timer(self, seconds):
        self.stop_timer()
        self._timer_stop = threading.Event()
        self._timer_thread = threading.Thread(
            target=self._run_countdown, args=(seconds, self._timer_stop), daemon=True)
        self._timer_thread.start()

    def stop_timer(self):
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_stop.set()
            self._timer_thread.join(timeout=0.3)

    def handle(self, intent, slot, conf=1.0):
        self.stop_timer()                            # a new command cancels a running countdown
        if intent == "TIMER":
            secs = _timer_seconds(slot)
            if secs:
                self.start_timer(secs)               # live countdown in the background
                return
        n, colon = _slot_number(intent, slot)
        if n is None:
            self.clear()
        else:
            self.number(n, colon=colon)

    def close(self):
        self.stop_timer()
        try:
            self.clear()
            self._lg.gpiochip_close(self.h)
        except Exception:
            pass


def _timer_seconds(slot):
    """Duration in seconds for a TIMER slot ('10 seconds' / '30 seconds' / '1 minute')."""
    s = (slot or "").lower()
    if "10" in s:
        return 10
    if "30" in s:
        return 30
    if "minute" in s:
        return 60
    return None


def _slot_number(intent, slot):
    """(intent, slot) -> (4-digit number, show-colon) for the display, or (None, _)."""
    s = (slot or "").lower()
    if intent == "TIMER":
        if "10" in s: return 10, False
        if "30" in s: return 30, False
        if "minute" in s: return 100, True            # 1:00
    elif intent == "ALARM":
        if "6" in s: return 600, True                 # 6:00
        if "8" in s: return 800, True
        if "9" in s: return 900, True
    elif intent == "TEMPERATURE":
        for v in (18, 22, 26):
            if str(v) in s: return v, False
    elif intent == "BRIGHTNESS":
        for v in (20, 60, 100):
            if str(v) in s: return v, False
    return None, False


if __name__ == "__main__":          # python pi_tm1637.py  (cycle test)
    t = TM1637()
    for intent, slot in [("TIMER", "30 seconds"), ("TIMER", "1 minute"),
                         ("ALARM", "6 AM"), ("TEMPERATURE", "22 degrees"),
                         ("BRIGHTNESS", "100 percent")]:
        print(intent, slot); t.handle(intent, slot); time.sleep(1.0)
    t.close()
