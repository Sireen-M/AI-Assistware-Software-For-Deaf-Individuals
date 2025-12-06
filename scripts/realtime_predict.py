import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from tensorflow.keras.models import load_model
from tcn import TCN
from keras.utils import custom_object_scope
from collections import deque

# Load the trained TCN model
with custom_object_scope({'TCN': TCN}):
    model = load_model('models/sign_model_tcn.h5')

LABELS = ['#BACK', 'AHEAD', 'ALCOHOL', 'ALLOW', 'APPLY', 'APPOINTMENT', 'ARRIVE', 'LOOK',
          'BAR', 'BELONG', 'BETTER', 'BADGE', 'BLOW-NOSE', 'EAT+NOON', 'GO', 'LINE',
          'LIST', 'LIVE', 'BLANKET', 'MISUNDERSTAND']

# TTS setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Prediction smoothing
prediction_queue = deque(maxlen=10)
last_spoken_word = ''
speak_cooldown = 0

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Video stream
cap = cv2.VideoCapture(0)
sequence = []

print("[INFO] Real-time sign prediction running...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

        # Pad with zeros if fewer than 126 landmarks
        if len(landmarks) < 126:
            landmarks += [0.0] * (126 - len(landmarks))

        if len(landmarks) == 126:
            sequence.append(landmarks)
            if len(sequence) > 30:
                sequence.pop(0)

            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0)  # shape: (1, 30, 126)
                prediction = model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(prediction)
                predicted_label = LABELS[predicted_index]
                confidence = prediction[predicted_index]

                prediction_queue.append(predicted_label)
                most_common_pred = max(set(prediction_queue), key=prediction_queue.count)

                # Display prediction on screen
                cv2.putText(frame, f'{most_common_pred} ({confidence:.2f})',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 255), 2)

                # Voice output if stable + confidence > 0.3
                if (most_common_pred != last_spoken_word and
                    prediction_queue.count(most_common_pred) > 6 and
                    confidence > 0.3 and
                    speak_cooldown == 0):

                    tts_engine.say(most_common_pred)
                    tts_engine.runAndWait()
                    last_spoken_word = most_common_pred
                    speak_cooldown = 30  # delay for next voice

        # Decrease cooldown
        if speak_cooldown > 0:
            speak_cooldown -= 1

        cv2.imshow("Sign Prediction (TCN)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
