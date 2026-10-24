/*
 * ME2 VCM — Phase 0 hardware bring-up test (ESP32-S3)
 *
 * Serial menu @ 115200 — type a letter + Enter:
 *   o = OLED    t = TM1637    n = STRIP thermometer   r = STRIP lamp
 *   b = buzzer  k = KY-038    a = run all
 *
 * Only ONE WS2812 strip is used (ring is out of commission). It plays both
 * roles: thermometer sweep (n) and lamp on/dim/color (r). In the real demo the
 * strip can be split into a thermometer zone + a lamp zone on this one data pin.
 *
 * Libraries: Adafruit SSD1306, Adafruit GFX, TM1637Display, Adafruit NeoPixel
 *
 * Pinout:  OLED SDA=8 SCL=9   TM1637 CLK=5 DIO=4   STRIP DIN=18
 *          Buzzer=7           KY-038 D0=6
 */
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TM1637Display.h>
#include <Adafruit_NeoPixel.h>

#define PIN_SDA     8
#define PIN_SCL     9
#define PIN_TM_CLK  5
#define PIN_TM_DIO  4
#define PIN_STRIP   18
#define PIN_BUZZER  7
#define PIN_KY038   6

#define STRIP_COUNT 30         // <-- set to your 0.5m strip's actual LED count

Adafruit_SSD1306 oled(128, 64, &Wire, -1);
TM1637Display    tm(PIN_TM_CLK, PIN_TM_DIO);
Adafruit_NeoPixel strip(STRIP_COUNT, PIN_STRIP, NEO_GRB + NEO_KHZ800);

bool oledOK = false;

void fillAll(uint32_t c) { for (int i=0;i<STRIP_COUNT;i++) strip.setPixelColor(i,c); strip.show(); }

void testOLED() {
  Serial.println("[OLED] drawing...");
  if (!oledOK) { Serial.println("  OLED not found at 0x3C — check SDA=8/SCL=9, or it's the SPI model"); return; }
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(2); oled.setCursor(0, 0);  oled.println("VCM OK");
  oled.setTextSize(1); oled.setCursor(0, 24); oled.println("intent: TIMER");
  oled.setCursor(0, 36); oled.println("slot:   30 seconds");
  oled.display();
  Serial.println("  -> screen should show 'VCM OK' + text");
}

void testTM1637() {
  Serial.println("[TM1637] 12:34 then a 5s countdown...");
  tm.setBrightness(7);
  tm.showNumberDecEx(1234, 0b01000000, true);
  delay(1200);
  for (int s = 5; s >= 0; s--) { tm.showNumberDecEx(s, 0b01000000, true); delay(1000); }
  Serial.println("  -> digits show 12:34 then count 5->0");
}

void testStrip() {                              // thermometer: fill blue->green->red
  Serial.println("[STRIP] thermometer sweep...");
  strip.setBrightness(30);
  strip.clear(); strip.show();
  for (int i = 0; i < STRIP_COUNT; i++) {
    uint32_t c = (i < STRIP_COUNT/3)   ? strip.Color(0,0,255)
               : (i < 2*STRIP_COUNT/3) ? strip.Color(0,255,0)
                                       : strip.Color(255,0,0);
    strip.setPixelColor(i, c); strip.show(); delay(120);
  }
  delay(800); strip.clear(); strip.show();
  Serial.println("  -> LEDs fill upward blue->green->red, then clear");
}

void testLamp() {                               // light control on the strip
  Serial.println("[LAMP] strip as lamp: white -> dim -> red/green/blue...");
  strip.setBrightness(70); fillAll(strip.Color(255,255,255)); delay(900);  // ON
  strip.setBrightness(15); strip.show();                       delay(900);  // DIM
  strip.setBrightness(70);
  fillAll(strip.Color(255,0,0)); delay(700);                                // red
  fillAll(strip.Color(0,255,0)); delay(700);                                // green
  fillAll(strip.Color(0,0,255)); delay(700);                                // blue
  strip.clear(); strip.show();
  Serial.println("  -> strip: white, dims, then red/green/blue, then off");
}

void testBuzzer() {
  Serial.println("[Buzzer] 3 beeps...");
  for (int i = 0; i < 3; i++) { tone(PIN_BUZZER, 2000, 150); delay(280); }
  noTone(PIN_BUZZER);
  Serial.println("  -> you should hear 3 beeps");
}

void testKY038() {
  Serial.println("[KY-038] reading D0 for 5s — clap/talk near it...");
  unsigned long t0 = millis(); int hits = 0;
  while (millis() - t0 < 5000) {
    if (digitalRead(PIN_KY038) == HIGH) { hits++; Serial.println("  sound!"); delay(120); }
  }
  Serial.printf("  -> %d sound events (turn the pot if 0 or always-on)\n", hits);
}

void setup() {
  Serial.begin(115200); delay(400);
  Wire.begin(PIN_SDA, PIN_SCL);
  oledOK = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  strip.begin(); strip.clear(); strip.show();
  pinMode(PIN_KY038, INPUT);
  Serial.println("\n=== ME2 hardware test ===");
  Serial.println("o=OLED t=TM1637 n=STRIP-thermo r=STRIP-lamp b=buzzer k=KY038 a=all");
}

void loop() {
  if (!Serial.available()) return;
  char c = Serial.read();
  switch (c) {
    case 'o': testOLED();   break;
    case 't': testTM1637(); break;
    case 'n': testStrip();  break;
    case 'r': testLamp();   break;
    case 'b': testBuzzer(); break;
    case 'k': testKY038();  break;
    case 'a': testOLED(); testTM1637(); testStrip(); testLamp(); testBuzzer(); testKY038(); break;
    default: break;
  }
}
