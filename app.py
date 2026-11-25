import cv2
import numpy as np
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download Model File
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    if os.path.exists(model_path):
        os.remove(model_path)
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download Complete!")

# 2. Setup Detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

# 3. Load Images (ඔයා එවපු Image filenames)
img_ayubowan = cv2.imread("ayubowan.jpeg")
img_winpos = cv2.imread("winpos.jpeg")
img_angry = cv2.imread("angry.jpg")

img_size = (450, 550)
img_ayubowan = cv2.resize(img_ayubowan, img_size) if img_ayubowan is not None else np.zeros((550, 450, 3), dtype=np.uint8)
img_winpos = cv2.resize(img_winpos, img_size) if img_winpos is not None else np.zeros((550, 450, 3), dtype=np.uint8)
img_angry = cv2.resize(img_angry, img_size) if img_angry is not None else np.zeros((550, 450, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    
    mr_photo = img_winpos # Default image
    pose_name = "WIN POS (Default)"

    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        landmarks = detection_result.pose_landmarks[0]

        # Keypoints: Wrists, Shoulders, Ears
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_ear = landmarks[7]
        right_ear = landmarks[8]

        # Draw green points on wrists
        cv2.circle(frame, (int(left_wrist.x * w), int(left_wrist.y * h)), 8, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_wrist.x * w), int(right_wrist.y * h)), 8, (0, 255, 0), -1)

        # Distances & Positional logic
        hand_dist = np.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
        
        # Distance to ears (for Angry / Call pose)
        dist_left_to_ear = np.sqrt((left_wrist.x - left_ear.x)**2 + (left_wrist.y - left_ear.y)**2)
        dist_right_to_ear = np.sqrt((right_wrist.x - right_ear.x)**2 + (right_wrist.y - right_ear.y)**2)

        is_hand_near_ear = dist_left_to_ear < 0.25 or dist_right_to_ear < 0.25
        is_hand_above_shoulder = left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y

        # MATCHING LOGIC
        if hand_dist < 0.35:
            mr_photo = img_ayubowan
            pose_name = "AYUBOWAN"
        elif is_hand_near_ear:
            mr_photo = img_angry
            pose_name = "ANGRY / ON CALL"
        elif is_hand_above_shoulder:
            mr_photo = img_winpos
            pose_name = "WIN POS"

        cv2.putText(frame, f"Dist: {round(hand_dist, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, f"Pose: {pose_name}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Mahinda Rajapaksa Match", mr_photo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()