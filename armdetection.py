import cv2
import mediapipe as mp
import math
import numpy as np
import serialmanager

"""

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

#methods
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
    #returns angle in degrees for easy display
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot_product(v1,v2)/(mag1*mag2)))
    return math.degrees(math.acos(cos_angle))

def hand_open_ratio(hand_landmarks, w, h):
    #tells the distance between the tips of the fingers and hands
    tips = [4,8,12,16,20] #landmark #s
    bases = [2,5,9,13,17] #landmark #s
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

def distance_3d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2)

def distance_3d_lm(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2 +
            (p1.z - p2.z) ** 2)

def distance_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2)

def average_point_2d(landmarks, landmark_ids, image_width, image_height):
    """
    Compute the average (x, y) point of a set of landmarks.

    Args:
        landmarks: list of MediaPipe landmarks
        landmark_ids: list of landmark indices to average
        image_width: width of the image (for pixel coordinates)
        image_height: height of the image (for pixel coordinates)

    Returns:
        np.array([avg_x, avg_y])
    """
    points = np.array([
        [landmarks[i].x * image_width, landmarks[i].y * image_height]
        for i in landmark_ids
    ])
    return np.mean(points, axis=0)

def average_point_3d(landmarks, landmark_ids):
    """
    Compute the average (x, y) point of a set of landmarks.

    Args:
        landmarks: list of MediaPipe landmarks
        landmark_ids: list of landmark indices to average
        image_width: width of the image (for pixel coordinates)
        image_height: height of the image (for pixel coordinates)

    Returns:
        np.array([avg_x, avg_y])
    """
    points = np.array([
        [landmarks[i].x, landmarks[i].y, landmarks[i].z]
        for i in landmark_ids
    ])
    return np.mean(points, axis=0)

#init MediaPose positions and find estimates of body
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#initialize serial
ser = serialmanager.SerialManager(port="COM7", baudrate=115200)

#initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

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
        left_vec = normalize(vector(right_shoulder, left_shoulder))
        right_vec = normalize(vector(left_shoulder, right_shoulder))
        up_vec = normalize(cross_product(forward, right_vec))

        #Left arm vectors relative to torso
        left_elbow = to_3d_point(lm[13], w, h)
        left_wrist = to_3d_point(lm[15], w, h)
        left_upper_arm = vector(left_shoulder, left_elbow)
        left_forearm = vector(left_elbow, left_wrist)
        
        #Right arm vectors relative to torso
        right_elbow = to_3d_point(lm[14], w, h)
        right_wrist = to_3d_point(lm[16], w, h)
        right_upper_arm = vector(right_shoulder, right_elbow)
        right_forearm = vector(right_elbow, right_wrist)
        
        #Recorded angles for the robot arms: will be send over serial
        left_shoulder_forward = angle_between_vectors(left_upper_arm, forward)
        left_shoulder_side    = angle_between_vectors(left_upper_arm, left_vec)
        left_elbow_angle      = angle_between_vectors(left_upper_arm, left_forearm)
        
        right_shoulder_forward = angle_between_vectors(right_upper_arm, forward)
        right_shoulder_side    = angle_between_vectors(right_upper_arm, right_vec)
        right_elbow_angle      = angle_between_vectors(right_upper_arm, right_forearm)


        
        #Draw on image
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #Draw angles on the image
        cv2.putText(frame, f"lsf: {int(left_shoulder_forward)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        cv2.putText(frame, f"lss: {int(left_shoulder_side)}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        cv2.putText(frame, f"le: {int(left_elbow_angle)}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)

        
        cv2.putText(frame, f"rsf: {int(right_shoulder_forward)}", (100,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        cv2.putText(frame, f"rss: {int(right_shoulder_side)}", (100,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        cv2.putText(frame, f"re: {int(right_elbow_angle)}", (100,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)

        

    #Hand detection
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #The tip of the fingers: used to check if the hand is open or closed
            finger_tip_ids = [8, 12, 16, 20]
            avg_finger_tip_3d = average_point_3d(hand_landmarks.landmark, finger_tip_ids)
            avg_finger_tip_2d = average_point_2d(hand_landmarks.landmark, finger_tip_ids, w, h)
            cv2.circle(frame, (int(avg_finger_tip_2d[0]), int(avg_finger_tip_2d[1])), 5, (0, 255, 0), -1)

            #The base of the fingers: used as a reference to determine the state of the hand
            finger_base_ids = [5, 9, 13, 17]
            avg_finger_base_3d = average_point_3d(hand_landmarks.landmark, finger_base_ids)
            avg_finger_base_2d = average_point_2d(hand_landmarks.landmark, finger_base_ids, w, h)
            cv2.circle(frame, (int(avg_finger_base_2d[0]), int(avg_finger_base_2d[1])), 5, (0, 255, 0), -1)
            
            #The base of the palm: used to compare the distance of the two parts of the hand
            wrist_base = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z)
            tip_dist = distance_3d(wrist_base, avg_finger_tip_3d)
            base_dist = distance_3d(wrist_base, avg_finger_base_3d)
            cv2.circle(frame, (int(wrist_base[0] * w), int(wrist_base[1] * h)), 5, (0, 255, 0), -1)
            
            state = "Open" if tip_dist > base_dist else "Closed"
            
            cv2.putText(frame, f"Hand: {state}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    

    cv2.imshow("Angles", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
