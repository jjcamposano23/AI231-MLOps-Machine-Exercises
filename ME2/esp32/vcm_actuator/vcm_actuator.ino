/*
 * ME2 VCM — Phase 1 serial actuator (ESP32-S3)
 *
 * The laptop runs the VCM and sends one line per command over USB serial:
 *     INTENT|slot_value        e.g.  TEMPERATURE|26 degrees   COLOR|red
 *                                    LIGHT_ON|   TIMER|30 seconds
 *
 * The single WS2812 strip shows ONE thing at a time (mode switch): the LAMP,
 * the THERMOMETER, or OFF — so LIGHT_OFF clears it and TEMPERATURE transitions
 * it from lamp to thermometer. Only N_USED LEDs near the power end are used.
 *
 * Test WITHOUT the laptop: Serial Monitor @115200 (line ending = Newline), type:
 *   LIGHT_ON|   COLOR|blue   BRIGHTNESS|20 percent   LIGHT_OFF|
 *   TEMPERATURE|26 degrees   TIMER|10 seconds   ALARM|6 AM
 *
 * Pinout: OLED SDA=8 SCL=9  TM1637 CLK=5 DIO=4  STRIP DIN=18  Buzzer=7
 */
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TM1637Display.h>
#include <Adafruit_NeoPixel.h>

#define PIN_SDA 8
#define PIN_SCL 9
#define PIN_TM_CLK 5
#define PIN_TM_DIO 4
#define PIN_STRIP 18
#define PIN_BUZZER 7

#define STRIP_COUNT 80                 // strip's physical LED count
#define N_USED      16                 // only LEDs 0..15 are used (rest stay dark)

Adafruit_SSD1306 oled(128, 64, &Wire, -1);
TM1637Display    tm(PIN_TM_CLK, PIN_TM_DIO);
Adafruit_NeoPixel strip(STRIP_COUNT, PIN_STRIP, NEO_GRB + NEO_KHZ800);

// ---- panel state ----
enum Mode { MODE_OFF, MODE_LAMP, MODE_THERMO };
Mode mode = MODE_OFF;
int  thermoDeg = 0;
uint8_t lampR = 255, lampG = 255, lampB = 255, lampPct = 100;
#define MAX_REM 5
String reminders[MAX_REM]; int remCount = 0;
String buf;

uint8_t sc(uint8_t v, uint8_t pct) { return (uint16_t)v * pct / 100; }

void renderStrip() {
  strip.setBrightness(255);
  strip.clear();
  if (mode == MODE_LAMP) {
    uint32_t c = strip.Color(sc(lampR,lampPct), sc(lampG,lampPct), sc(lampB,lampPct));
    for (int i = 0; i < N_USED; i++) strip.setPixelColor(i, c);
  } else if (mode == MODE_THERMO) {
    int level = (thermoDeg <= 18) ? 1 : (thermoDeg <= 22) ? 2 : 3;
    int on = N_USED * level / 3;
    for (int i = 0; i < on; i++) {
      uint32_t c = (i < N_USED/3)   ? strip.Color(0,0,150)
                 : (i < 2*N_USED/3) ? strip.Color(0,150,0)
                                    : strip.Color(180,0,0);
      strip.setPixelColor(i, c);
    }
  }
  strip.show();
}

void oledShow(const String& intent, const String& slot) {
  oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1); oled.setCursor(0, 0);  oled.println("VCM heard:");
  oled.setTextSize(intent.length() > 9 ? 1 : 2); oled.setCursor(0, 16); oled.println(intent);
  oled.setTextSize(1); oled.setCursor(0, 48);
  if (slot.length() && slot != "none") { oled.print("slot: "); oled.println(slot); }
  oled.display();
}

int leadingInt(const String& s) {
  int v = 0; bool any = false;
  for (unsigned i = 0; i < s.length(); i++) {
    char c = s[i];
    if (c >= '0' && c <= '9') { v = v*10 + (c-'0'); any = true; }
    else if (any) break;
  }
  return any ? v : -1;
}

void beep(int f, int ms) { tone(PIN_BUZZER, f, ms); delay(ms + 30); noTone(PIN_BUZZER); }

void doTimer(const String& slot) {
  int secs = (slot.indexOf("minute") >= 0) ? max(1, leadingInt(slot)) * 60
                                           : max(1, leadingInt(slot));
  oledShow("TIMER", slot);
  for (int s = secs; s >= 0; s--) {
    tm.showNumberDecEx((s/60)*100 + (s%60), 0b01000000, true);   // MM:SS
    delay(1000);
  }
  for (int i = 0; i < 3; i++) beep(2000, 160);                   // ring at 0
}

void doAlarm(const String& slot) {
  int h = leadingInt(slot); if (h <= 0) h = 6;
  if (slot.indexOf("PM") >= 0 && h < 12) h += 12;
  tm.showNumberDecEx(h * 100, 0b01000000, true);
  oledShow("ALARM set", slot);
}

void handle(String line) {
  line.trim(); if (!line.length()) return;
  int bar = line.indexOf('|');
  String intent = (bar >= 0) ? line.substring(0, bar) : line;
  String slot   = (bar >= 0) ? line.substring(bar + 1) : "";
  intent.trim(); slot.trim();
  Serial.print("recv: "); Serial.print(intent); Serial.print(" / "); Serial.println(slot);

  if (intent == "TIMER")       { doTimer(slot); return; }
  if (intent == "ALARM")       { doAlarm(slot); return; }
  if (intent == "TIME")        {
    int c = slot.indexOf(':');
    int hhmm = (c > 0) ? slot.substring(0,c).toInt()*100 + slot.substring(c+1).toInt() : 0;
    tm.showNumberDecEx(hhmm, 0b01000000, true); oledShow("TIME", slot); return;
  }
  if (intent == "TEMPERATURE") { thermoDeg = leadingInt(slot); mode = MODE_THERMO; renderStrip(); oledShow("TEMPERATURE", slot); return; }
  if (intent == "LIGHT_ON")    { mode = MODE_LAMP; renderStrip(); oledShow("LIGHT ON", ""); return; }
  if (intent == "LIGHT_OFF")   { mode = MODE_OFF;  renderStrip(); oledShow("LIGHT OFF", ""); return; }
  if (intent == "BRIGHTNESS")  { lampPct = constrain(leadingInt(slot),0,100); mode = MODE_LAMP; renderStrip(); oledShow("BRIGHTNESS", slot); return; }
  if (intent == "COLOR") {
    if      (slot.indexOf("red")   >= 0) { lampR=255; lampG=0;   lampB=0;   }
    else if (slot.indexOf("green") >= 0) { lampR=0;   lampG=255; lampB=0;   }
    else if (slot.indexOf("blue")  >= 0) { lampR=0;   lampG=0;   lampB=255; }
    mode = MODE_LAMP; renderStrip(); oledShow("COLOR", slot); return;
  }
  if (intent == "CREATE_REMINDER") { if (remCount < MAX_REM) reminders[remCount++] = slot; oledShow("Reminder +", slot); beep(1800,100); return; }
  if (intent == "LIST_REMINDERS") {
    oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE); oled.setTextSize(1);
    oled.setCursor(0,0); oled.println("Reminders:");
    for (int i = 0; i < remCount; i++) { oled.setCursor(0, 12 + i*10); oled.print(i+1); oled.print(". "); oled.println(reminders[i]); }
    if (remCount == 0) { oled.setCursor(0,12); oled.println("(none)"); }
    oled.display(); return;
  }
  if (intent == "CALL")    { oledShow("Calling...", slot); for (int i=0;i<3;i++){ beep(1200,180); beep(1000,180); delay(200);} return; }
  if (intent == "MESSAGE") { oledShow("Message sent", ""); beep(1800,120); return; }

  // PLAY_MUSIC, WEATHER, PAUSE, STOP, NEXT, VOLUME_UP, VOLUME_DOWN -> OLED only
  oledShow(intent, slot);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(PIN_SDA, PIN_SCL);
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  tm.setBrightness(7); tm.clear();
  strip.begin(); strip.clear(); strip.show();
  oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(2); oled.setCursor(0, 20); oled.println(" VCM ready"); oled.display();
  Serial.println("VCM actuator ready. Send: INTENT|slot");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handle(buf); buf = ""; }
    else if (c != '\r') buf += c;
  }
}
