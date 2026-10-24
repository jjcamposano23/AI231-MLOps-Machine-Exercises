/*
 * ME2 VCM actuator -- Arduino Uno
 *
 * Receives one command per line (newline-terminated) from the laptop VCM over
 * USB serial at 115200 baud and drives simple actuators so each of the 10
 * voice commands has a visible / audible reaction.
 *
 * Wiring (all via resistors as needed):
 *   LIGHT_PIN  (12) -> LED  ("turn on/off lights")            digital
 *   DIM_PIN    ( 9) -> LED  ("dim lights to X%")   PWM-capable pin
 *   BUZZER_PIN ( 8) -> piezo buzzer  (timer / alarm)
 *   ACK_PIN    (13) -> onboard LED   (blinks to acknowledge any command)
 *
 * Protocol (from infer.py):
 *   LIGHTS_TOGGLE | LIGHTS_ON | LIGHTS_OFF
 *   DIM:<0-100>
 *   TIMER:<seconds>
 *   ALARM
 *   THERMO:<degrees>
 *   MUSIC | MEDIA | QUERY | REMINDER | CALL | UNKNOWN
 */

const int LIGHT_PIN  = 12;
const int DIM_PIN    = 9;
const int BUZZER_PIN = 8;
const int ACK_PIN    = 13;

bool lightsOn = false;

// Non-blocking timer state.
bool timerActive = false;
unsigned long timerFireAt = 0;

String buf = "";

void ackBlink() {
  digitalWrite(ACK_PIN, HIGH); delay(60); digitalWrite(ACK_PIN, LOW);
}

void beep(int ms) {
  tone(BUZZER_PIN, 2000); delay(ms); noTone(BUZZER_PIN);
}

void alarmPattern() {
  for (int i = 0; i < 3; i++) { beep(150); delay(120); }
}

void handle(String cmd) {
  cmd.trim();
  ackBlink();

  if (cmd == "LIGHTS_TOGGLE") {
    lightsOn = !lightsOn;
    digitalWrite(LIGHT_PIN, lightsOn ? HIGH : LOW);
  } else if (cmd == "LIGHTS_ON") {
    lightsOn = true;  digitalWrite(LIGHT_PIN, HIGH);
  } else if (cmd == "LIGHTS_OFF") {
    lightsOn = false; digitalWrite(LIGHT_PIN, LOW);
  } else if (cmd.startsWith("DIM:")) {
    int pct = cmd.substring(4).toInt();          // 0..100
    pct = constrain(pct, 0, 100);
    analogWrite(DIM_PIN, map(pct, 0, 100, 0, 255));
  } else if (cmd.startsWith("TIMER:")) {
    long secs = cmd.substring(6).toInt();
    timerActive = true;
    timerFireAt = millis() + (unsigned long)secs * 1000UL;
  } else if (cmd == "ALARM") {
    alarmPattern();
  } else if (cmd.startsWith("THERMO:")) {
    // Placeholder: represent temperature however you like (servo, display...).
    beep(80);
  } else {
    // MUSIC / MEDIA / QUERY / REMINDER / CALL / UNKNOWN -> just acknowledge.
    beep(40);
  }

  Serial.print("ok:"); Serial.println(cmd);      // echo back for debugging
}

void setup() {
  pinMode(LIGHT_PIN, OUTPUT);
  pinMode(DIM_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(ACK_PIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  // Read serial into a line buffer.
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') { handle(buf); buf = ""; }
    else if (c != '\r') { buf += c; }
  }

  // Fire the timer when it elapses.
  if (timerActive && (long)(millis() - timerFireAt) >= 0) {
    timerActive = false;
    alarmPattern();
  }
}
