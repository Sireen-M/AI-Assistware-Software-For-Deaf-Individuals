import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd

# Paths
PROCESSED_DIR = 'processed_videos'
OUTPUT_DIR = 'landmarks_csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# Extract landmarks from video
for filename in os.listdir(PROCESSED_DIR):
    if not filename.endswith('.mp4'):
        continue

    video_path = os.path.join(PROCESSED_DIR, filename)
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        frame_landmarks = []
        hand_data = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                hand_data.append(landmarks)

            # Ensure two-hand format (dominant + non-dominant)
            if len(hand_data) == 1:
                hand_data.append([0] * 63)
            elif len(hand_data) > 2:
                hand_data = hand_data[:2]

            frame_landmarks = hand_data[0] + hand_data[1]
        else:
            frame_landmarks = [0] * 126

        landmarks_list.append(frame_landmarks)

    cap.release()

    if landmarks_list:
        df = pd.DataFrame(landmarks_list)
        csv_name = os.path.splitext(filename)[0] + '.csv'
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved landmarks: {csv_name}")
    else:
        print(f"⚠️ No landmarks found in: {filename}")
