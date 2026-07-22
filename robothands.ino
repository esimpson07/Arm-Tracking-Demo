#include <Servo.h>

const int NUM_SERVOS = 8;
unsigned long lastUpdate = 0;

// Servo objects
Servo servos[NUM_SERVOS];

// Servo pins
const int servoPins[NUM_SERVOS] = {12, 11, 10, 9, 4, 5, 6, 7};

// Reverse direction flags (true = reversed)
bool servoReverse[NUM_SERVOS] = {false, false, true, false, true, false, false, false};

// Minimum and maximum limits for each servo
int servoMin[NUM_SERVOS] = {0, 0, 90, 95, 0, 0, 90, 95};
int servoMax[NUM_SERVOS] = {180, 180, 180, 163, 180, 180, 180, 163};

// ---------------------------------------------------------------------------
// Motion smoothing (slew-rate + acceleration limiting)
// ---------------------------------------------------------------------------
//
// Incoming serial values are now treated as TARGETS, not immediate positions.
// Every loop we ease each servo's current angle toward its target under a
// capped speed and acceleration. This removes the "aggressive snap": the servo
// can no longer jump straight to a new target - it accelerates, cruises, and
// decelerates into place. Because this loop runs far faster than the ~30
// targets/sec arriving over serial, it also interpolates smoothly between them
// instead of stepping.
//
// Tune these against the real rig:
const float MAX_SPEED = 150.0;  // deg/sec   - top servo travel speed (lower = gentler)
const float MAX_ACCEL = 500.0;  // deg/sec^2 - how fast speed can change (lower = softer starts/stops)
const float DEADBAND  = 0.75;   // deg       - ignore targets this close (kills resting buzz)

float currentAngle[NUM_SERVOS];  // where each servo actually is (smoothed)
float targetAngle[NUM_SERVOS];   // where the latest serial data wants it
float currentVel[NUM_SERVOS];    // current angular velocity (deg/sec)

// Input buffer
String inputString = "";

void setup() {
  Serial.begin(115200);

  // Attach servos and initialise the smoothing state to neutral.
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(90); // Neutral default
    currentAngle[i] = 90.0;
    targetAngle[i]  = 90.0;
    currentVel[i]   = 0.0;
  }

  lastUpdate = millis(); // avoid a huge first-frame dt
}

void loop() {
  unsigned long now = millis();
  float dt = (now - lastUpdate) / 1000.0;  // elapsed time in seconds
  lastUpdate = now;

  // Clamp dt so a stall (or the very first frame) can't cause a jump.
  if (dt <= 0.0) return;
  if (dt > 0.05) dt = 0.05;

  // Read any available serial and update TARGET angles.
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

  // Ease every servo toward its target under speed/accel limits.
  updateServos(dt);
}

// ---------------------------------------------------------------------------
// Smoothly drive each servo toward its target
// ---------------------------------------------------------------------------
void updateServos(float dt) {
  for (int i = 0; i < NUM_SERVOS; i++) {
    float error = targetAngle[i] - currentAngle[i];

    if (fabs(error) <= DEADBAND) {
      // Close enough - settle exactly and stop, so the servo doesn't buzz.
      currentAngle[i] = targetAngle[i];
      currentVel[i]   = 0.0;
    } else {
      // Fastest velocity that still reaches the target this step, capped by
      // MAX_SPEED. As the error shrinks this naturally eases out.
      float desiredVel = error / dt;
      desiredVel = constrain(desiredVel, -MAX_SPEED, MAX_SPEED);

      // Limit how fast the velocity itself may change -> soft starts/stops.
      float maxDeltaV = MAX_ACCEL * dt;
      float dv = constrain(desiredVel - currentVel[i], -maxDeltaV, maxDeltaV);
      currentVel[i] += dv;

      currentAngle[i] += currentVel[i] * dt;
    }

    // Safety clamp to the physical servo range and write.
    float out = constrain(currentAngle[i], 0.0, 180.0);
    servos[i].write((int)round(out));
  }
}

void processCommandLine(String line) {
  // Split commands by ';'
  int start = 0;
  while (start < line.length()) {
    int spaceIndex = line.indexOf(';', start);
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
        if(servoIndex == 2 || servoIndex == 6) {
          angle = 90 + angle;
        }
        // Apply limits
        if (angle < servoMin[servoIndex]) angle = servoMin[servoIndex];
        if (angle > servoMax[servoIndex]) angle = servoMax[servoIndex];

        // Apply reversal
        if (servoReverse[servoIndex]) {
          angle = 180 - angle;
        }

        // Store as the new TARGET - motion smoothing happens in updateServos().
        targetAngle[servoIndex] = angle;
      }
    }
  }
}
