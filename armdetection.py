import cv2
import mediapipe as mp
import math

# =========================
# Helper functions
# =========================
def to_3d_point(landmark, w, h, scale_z=1.0):
    return (landmark.x*w, landmark.y*h, landmark.z*scale_z)

def vector(a, b):
    return (b[0]-a[0], b[1]-a[1], b[2]-a[2])

def vector_magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def normalize(v):
    mag = vector_magnitude(v)
    if mag == 0:
        return (0,0,0)
    return (v[0]/mag, v[1]/mag, v[2]/mag)

def dot_product(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def angle_between_vectors(v1, v2):
    """Returns angle in degrees between two 3D vectors"""
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot_product(v1,v2)/(mag1*mag2)))
    return math.degrees(math.acos(cos_angle))

def hand_open_ratio(hand_landmarks, w, h):
    tips = [4,8,12,16,20]
    bases = [2,5,9,13,17]
    total = 0
    for tip, base in zip(tips,bases):
        tip_pos = (hand_landmarks.landmark[tip].x*w, hand_landmarks.landmark[tip].y*h)
        base_pos = (hand_landmarks.landmark[base].x*w, hand_landmarks.landmark[base].y*h)
        total += math.dist(tip_pos, base_pos)
    return total / len(tips)

def cross_product(a,b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])

#init MP
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)


    #Hand detection
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            ratio = hand_open_ratio(hand_landmarks, w, h)
            state = "Open" if ratio > 50 else "Closed"
            
        cv2.putText(frame, f"Hand: {state}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        #Shoulder points
        left_shoulder = to_3d_point(lm[11], w, h)
        right_shoulder = to_3d_point(lm[12], w, h)
        shoulder_mid = ((left_shoulder[0]+right_shoulder[0])/2,
                        (left_shoulder[1]+right_shoulder[1])/2,
                        (left_shoulder[2]+right_shoulder[2])/2)

        #Torso reference axes
        forward = normalize(vector(shoulder_mid, to_3d_point(lm[0], w, h))) # midpoint to nose
        right_vec = normalize(vector(left_shoulder, right_shoulder))
        up_vec = normalize(cross_product(forward, right_vec))

        #Right arm vectors relative to shoulder
        shoulder = right_shoulder
        elbow = to_3d_point(lm[14], w, h)
        wrist = to_3d_point(lm[16], w, h)
        upper_arm = vector(shoulder, elbow)
        forearm = vector(elbow, wrist)

        #Recorded angles for the robotic arms
        shoulder_forward = angle_between_vectors(upper_arm, forward)
        shoulder_side    = angle_between_vectors(upper_arm, right_vec)
        elbow_angle      = angle_between_vectors(upper_arm, forearm)

        #Draw on image
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #Display angles on screen
        cv2.putText(frame, f"Shoulder Forward: {int(shoulder_forward)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        cv2.putText(frame, f"Shoulder Side: {int(shoulder_side)}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)


    cv2.imshow("Arm & Hand Angles Relative to Shoulders", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
