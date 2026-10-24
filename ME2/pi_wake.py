"""KY-038 sound-sensor 'wake board' for the Pi demo (option B).

The KY-038's D0 line flips when sound crosses its potentiometer-set threshold.
We watch that line on a Pi GPIO pin; when it fires, pi_assistant opens a short
listening window and classifies the next spoken utterance as the command. This
is the hardware wake Dr. Atienza described -- no pretrained wake-word model, so
it keeps the 'from scratch' rule clean (the wake is a sensor, not an ML model).

Wiring (KY-038 -> Raspberry Pi):
    VCC -> 3V3  (pin 1)      so D0 swings 0-3.3 V and is safe for the Pi GPIO
    GND -> GND  (pin 6/9/14/...)
    D0  -> the --wake-gpio pin (BCM numbering, default GPIO6 = physical pin 31)
    A0  -> not used
Turn the onboard potentiometer until the D0 LED just flips when you clap / speak.
Quick check on the Pi:  python pi_wake.py 6   (prints 'sound!' on each trigger)
"""
import time


class WakeGate:
    def __init__(self, pin=6):
        from gpiozero import DigitalInputDevice
        # KY-038 D0 is a driven (push-pull) comparator output, so no pull needed;
        # we treat ANY level transition as 'sound crossed threshold' -> polarity
        # doesn't matter whether this particular board idles high or low.
        self.dev = DigitalInputDevice(pin, pull_up=None, active_state=True)

    def wait_for_sound(self, timeout=None):
        """Block until the sensor line transitions (sound). True on trigger, False on timeout."""
        start = time.time()
        last = self.dev.value
        while True:
            v = self.dev.value
            if v != last:
                return True
            last = v
            if timeout is not None and time.time() - start > timeout:
                return False
            time.sleep(0.005)

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


if __name__ == "__main__":          # standalone test: python pi_wake.py [pin]
    import sys
    g = WakeGate(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    print("listening for sound (Ctrl-C to stop)...")
    try:
        while True:
            if g.wait_for_sound():
                print("sound!", time.strftime("%H:%M:%S"))
                time.sleep(0.3)     # debounce the print
    except KeyboardInterrupt:
        pass
    finally:
        g.close()
