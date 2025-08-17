#include <Servo.h>

const int NUM_SERVOS = 8;
unsigned long lastUpdate = millis();

// Servo objects
Servo servos[NUM_SERVOS];

// Servo pins
const int servoPins[NUM_SERVOS] = {12, 11, 10, 9, 4, 5, 6, 7};

// Reverse direction flags (true = reversed)
bool servoReverse[NUM_SERVOS] = {true, true, false, false, false, true, true, false};

// Minimum and maximum limits for each servo
int servoMin[NUM_SERVOS] = {0, 0, 0, 0, 0, 0, 0, 0};
int servoMax[NUM_SERVOS] = {180, 180, 180, 180, 180, 180, 180, 180};

// Input buffer
String inputString = "";

void setup() {
  Serial.begin(115200);

  // Attach servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(90); // Neutral default
  }

  Serial.println("Servo Controller Ready!");
  Serial.println("Send commands: S<index>:<angle> (multiple allowed)");
  Serial.println("Example: S0:120 S3:45 S7:90");
}

void loop() {
  unsigned long now = millis();
  float dt = (now - lastUpdate) / 1000.0;  // elapsed time in seconds
  lastUpdate = now;
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      if (inputString.length() > 0) {
        processCommandLine(inputString);
        inputString = "";
      }
    } else {
      inputString += inChar;
    }
  }
}

void processCommandLine(String line) {
  // Split commands by space
  int start = 0;
  while (start < line.length()) {
    int spaceIndex = line.indexOf(' ', start);
    if (spaceIndex == -1) spaceIndex = line.length();

    String command = line.substring(start, spaceIndex);
    processCommand(command);

    start = spaceIndex + 1;
  }
}

void processCommand(String command) {
  // Expected format: S<servoIndex>:<angle>
  if (command.length() < 4) return; // Too short to be valid

  if (command.charAt(0) == 'S') {
    int colonIndex = command.indexOf(':');
    if (colonIndex > 1) {
      int servoIndex = command.substring(1, colonIndex).toInt();
      int angle = command.substring(colonIndex + 1).toInt();

      if (servoIndex >= 0 && servoIndex < NUM_SERVOS) {
        // Apply limits
        if (angle < servoMin[servoIndex]) angle = servoMin[servoIndex];
        if (angle > servoMax[servoIndex]) angle = servoMax[servoIndex];

        // Apply reversal
        if (servoReverse[servoIndex]) {
          angle = 180 - angle;
        }

        servos[servoIndex].write(angle);

        Serial.print("Servo ");
        Serial.print(servoIndex);
        Serial.print(" -> ");
        Serial.println(angle);
      } else {
        Serial.println("Invalid servo index.");
      }
    } else {
      Serial.println("Invalid format. Use S<index>:<angle>");
    }
  }
}
