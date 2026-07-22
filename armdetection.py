import cv2
import mediapipe as mp
import math
import time
import argparse
import numpy as np
import serial

"""

    Find more about this project on my github @ esimpson07

    or at https://github.com/esimpson07/Arm-Tracking-Demo/blob/main/README.md

    USAGE:

    In order to run this program, you should probably run it from the Command Prompt or
    the Terminal. The argument specifications are:

    python armdetection.py --port COM1 --baudrate 115200

    or running with the default baudrate (115200):

    python armdetection.py --port COM1

    or just running on the laptop without serial communication:

    python armdetection.py

    Optionally, tune the smoothing (One Euro filter) with:

    python armdetection.py --port COM1 --min-cutoff 1.0 --beta 0.02

    Lower --min-cutoff = steadier when still (less jitter); raise --beta if
    fast motions start to lag.



    This program can run independently of the robotic hands, but for the demonstration
    it is necessary to have the robotic hands connected via serial.



    DEPENDENCIES:

    pip install opencv-python
    pip install mediapipe
    pip install numpy
    pip install pyserial



    This program is made to be a program that detects the position of the arms and hands,
    and finds the angles between the different joints and saves them. Using the MediaPipe
    landmarks, the 3d points will be found and the angles between the landmarks will be
    saved to be used in accordance to a set of robotic hands from TMH for a Make-A-Wish
    demo.

    This program using MediaPipe and OpenCV to find and estimate the position of the arms
    and hands, and the data that I will use will be:
    - External shoulder rotation
    - Forward shoulder rotation
    - Elbow rotation
    - Hand state (open / closed)

    ///

    Landmark Associations (XYZ): LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_ELBOW: 13,
    RIGHT_ELBOW: 14, LEFT_WRIST: 15, RIGHT_WRIST: 16

"""


# ---------------------------------------------------------------------------
# One Euro Filter (adaptive smoothing)
# ---------------------------------------------------------------------------
#
# Replaces the old fixed-alpha EMA. A plain EMA forces a single trade-off:
# smooth-but-laggy OR responsive-but-jittery. The One Euro filter is adaptive -
# it smooths heavily when the signal is nearly still (kills servo jitter at
# rest) and automatically backs off when you move fast (keeps lag low). It is
# the standard filter for noisy human-motion tracking (MediaPipe / Kinect).
#
# Tuning (both exposed on the command line):
#   min_cutoff : lower  -> more smoothing when still (less jitter, more lag)
#   beta       : higher -> less lag when moving fast (but more jitter)
# Workflow: set beta = 0, lower min_cutoff until jitter at rest looks good,
# then raise beta until fast moves stop lagging.

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        # First sample: seed the state and pass the value straight through.
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0.0:
            dt = 1e-3  # guard against zero / non-monotonic timestamps

        # Filter the derivative, then let its magnitude drive the cutoff.
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroSmoother:
    """Keeps an independent One Euro filter per named signal."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filters = {}

    def smooth(self, key, new_value, t):
        f = self.filters.get(key)
        if f is None:
            f = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
            self.filters[key] = f
        return f.filter(new_value, t)


# ---------------------------------------------------------------------------
# Serial Handler
# ---------------------------------------------------------------------------

class SerialHandler:
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connected = False
        self.connect()

    def connect(self):
        if self.port is None:
            print("[Serial] No COM port specified. Running in simulation mode.")
            return
        try:
            self.ser = serial.Serial(
                self.port, self.baudrate, timeout=1, write_timeout=0.02
            )
            self.connected = True
            print(f"[Serial] Connected to {self.port} at {self.baudrate} baudrate.")
        except serial.SerialException:
            print(
                f"[Serial] Could not connect to {self.port}. Running in simulation mode."
            )

    def send(self, message):
        if self.connected and self.ser:
            try:
                self.ser.write((message + "\n").encode())
            except serial.SerialException:
                print("[Serial] Write failed. Switching to simulation mode.")
                self.connected = False
        else:
            print(f"[Serial-Sim] {message}")

    def close(self):
        if self.connected and self.ser:
            self.ser.close()
            print("[Serial] Connection closed.")


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def to_3d_point(landmark, w, h, scale_z=1.0):
    return (landmark.x * w, landmark.y * h, landmark.z * scale_z)


def vector(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def vector_magnitude(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def normalize(v):
    mag = vector_magnitude(v)
    if mag == 0:
        return (0, 0, 0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def angle_between_vectors(v1, v2):
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot_product(v1, v2) / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def cross_product(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def distance_3d(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


def average_point_2d(landmarks, landmark_ids, image_width, image_height):
    points = np.array(
        [
            [landmarks[i].x * image_width, landmarks[i].y * image_height]
            for i in landmark_ids
        ]
    )
    return np.mean(points, axis=0)


def average_point_3d(landmarks, landmark_ids):
    points = np.array(
        [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in landmark_ids]
    )
    return np.mean(points, axis=0)


# ---------------------------------------------------------------------------
# Serial message formatter
# ---------------------------------------------------------------------------

def format_arm_data(
    left_shoulder_forward,
    left_shoulder_side,
    left_elbow_angle,
    right_shoulder_forward,
    right_shoulder_side,
    right_elbow_angle,
    hand_states,
):
    """
    Combine arm joint angles + hand states into a single formatted string.
    hand_states should be a dict: {"Left": ratio_str, "Right": ratio_str}
    """
    return (
        f"S0:{left_shoulder_forward:.1f};"
        f"S1:{left_shoulder_side:.1f};"
        f"S2:{left_elbow_angle:.1f};"
        f"S3:{hand_states['Left']};"
        f"S4:{right_shoulder_forward:.1f};"
        f"S5:{right_shoulder_side:.1f};"
        f"S6:{right_elbow_angle:.1f};"
        f"S7:{hand_states['Right']};"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Arm Detection With Serial")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port (e.g. COM7 on Windows or /dev/ttyUSB0 on Linux).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Baud rate (default: 115200).",
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=1.0,
        help=(
            "One Euro min cutoff frequency (Hz). "
            "Lower = steadier when still (less jitter) but more lag. "
            "Default: 1.0"
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help=(
            "One Euro speed coefficient. "
            "Higher = less lag on fast moves (but more jitter). "
            "Default: 0.02"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_detector = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

args = parse_args()
serial_handler = SerialHandler(port=args.port, baudrate=args.baudrate)
smoother = OneEuroSmoother(min_cutoff=args.min_cutoff, beta=args.beta)
print(f"[Smoothing] One Euro filter: min_cutoff={args.min_cutoff}, beta={args.beta}")

# Fallback values used before the first valid detection
left_shoulder_forward  = 90.0
left_shoulder_side     = 90.0
left_elbow_angle       = 90.0

right_shoulder_forward = 90.0
right_shoulder_side    = 90.0
right_elbow_angle      = 90.0

hand_states = {"Left": "90.00", "Right": "90.00"}

cap = cv2.VideoCapture(0)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()   # single timestamp for all filters this frame

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results   = pose.process(rgb_frame)
    hands_results  = hands_detector.process(rgb_frame)

    # ------------------------------------------------------------------
    # Pose / arm detection
    # ------------------------------------------------------------------
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        # Key body points
        left_shoulder  = to_3d_point(lm[11], w, h)
        right_shoulder = to_3d_point(lm[12], w, h)
        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_shoulder[2] + right_shoulder[2]) / 2,
        )

        # Torso reference axes
        forward   = normalize(vector(shoulder_mid, to_3d_point(lm[0], w, h)))
        left_vec  = normalize(vector(right_shoulder, left_shoulder))
        right_vec = normalize(vector(left_shoulder, right_shoulder))
        up_vec    = normalize(cross_product(forward, right_vec))  # noqa: F841

        # Left arm
        left_elbow   = to_3d_point(lm[13], w, h)
        left_wrist   = to_3d_point(lm[15], w, h)
        left_upper_arm = vector(left_shoulder, left_elbow)
        left_forearm   = vector(left_elbow, left_wrist)

        # Right arm
        right_elbow   = to_3d_point(lm[14], w, h)
        right_wrist   = to_3d_point(lm[16], w, h)
        right_upper_arm = vector(right_shoulder, right_elbow)
        right_forearm   = vector(right_elbow, right_wrist)

        # Raw angle calculations → smoothed immediately
        left_shoulder_forward  = smoother.smooth(
            "lsf", angle_between_vectors(left_upper_arm, forward), now
        )
        left_shoulder_side     = smoother.smooth(
            "lss", angle_between_vectors(left_upper_arm, left_vec), now
        )
        left_elbow_angle       = smoother.smooth(
            "le",  angle_between_vectors(left_upper_arm, left_forearm), now
        )

        right_shoulder_forward = smoother.smooth(
            "rsf", angle_between_vectors(right_upper_arm, forward), now
        )
        right_shoulder_side    = smoother.smooth(
            "rss", angle_between_vectors(right_upper_arm, right_vec), now
        )
        right_elbow_angle      = smoother.smooth(
            "re",  angle_between_vectors(right_upper_arm, right_forearm), now
        )

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Overlay angle values
        cv2.putText(frame, f"lsf: {int(left_shoulder_forward)}",  (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"lss: {int(left_shoulder_side)}",     (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"le:  {int(left_elbow_angle)}",       (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"rsf: {int(right_shoulder_forward)}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"rss: {int(right_shoulder_side)}",    (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"re:  {int(right_elbow_angle)}",      (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ------------------------------------------------------------------
    # Hand detection
    # ------------------------------------------------------------------
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hand_landmarks, hand_label in zip(
            hands_results.multi_hand_landmarks, hands_results.multi_handedness
        ):
            label = hand_label.classification[0].label  # "Left" or "Right"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger tips and bases
            finger_tip_ids  = [8, 12, 16, 20]
            finger_base_ids = [5,  9, 13, 17]

            avg_finger_tip_3d  = average_point_3d(hand_landmarks.landmark, finger_tip_ids)
            avg_finger_tip_2d  = average_point_2d(hand_landmarks.landmark, finger_tip_ids, w, h)
            avg_finger_base_3d = average_point_3d(hand_landmarks.landmark, finger_base_ids)
            avg_finger_base_2d = average_point_2d(hand_landmarks.landmark, finger_base_ids, w, h)

            cv2.circle(frame, (int(avg_finger_tip_2d[0]),  int(avg_finger_tip_2d[1])),  5, (0, 255, 0), -1)
            cv2.circle(frame, (int(avg_finger_base_2d[0]), int(avg_finger_base_2d[1])), 5, (0, 255, 0), -1)

            wrist_base = (
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
                hand_landmarks.landmark[0].z,
            )
            cv2.circle(frame, (int(wrist_base[0] * w), int(wrist_base[1] * h)), 5, (0, 255, 0), -1)

            tip_dist  = distance_3d(wrist_base, avg_finger_tip_3d)
            base_dist = distance_3d(wrist_base, avg_finger_base_3d)

            raw_ratio    = tip_dist / base_dist if base_dist > 0 else 1.0
            clamped      = min(max(raw_ratio, 0.6), 1.8)
            scaled       = (clamped - 0.6) * 150.0

            # Smooth the hand ratio with its own One Euro filter per hand
            smoothed_scaled = smoother.smooth(f"hand_{label}", scaled, now)
            open_ratio = f"{smoothed_scaled:4.2f}"

            hand_states[label] = open_ratio

            if label == "Left":
                cv2.putText(frame, f"LH: {open_ratio}", (10, 120),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(frame, f"RH: {open_ratio}", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ------------------------------------------------------------------
    # Send over serial
    # ------------------------------------------------------------------
    message = format_arm_data(
        left_shoulder_forward,
        left_shoulder_side,
        left_elbow_angle,
        right_shoulder_forward,
        right_shoulder_side,
        right_elbow_angle,
        hand_states,
    )
    serial_handler.send(message)

    cv2.imshow("Angles", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cap.release()
serial_handler.close()
cv2.destroyAllWindows()
