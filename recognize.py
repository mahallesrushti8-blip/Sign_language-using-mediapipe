"""
STEP 3: LIVE RECOGNITION
-------------------------------------------------
Opens your webcam, tracks your hand with MediaPipe, and uses the
model you trained (model.pkl) to predict which sign you're making
in real time.

Run: python recognize.py
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter

MODEL_PATH = "model.pkl"

model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Smooths predictions over the last N frames so the label doesn't
# flicker wildly between two similar signs frame to frame.
recent_predictions = deque(maxlen=10)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label_text = "No hand detected"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
            row = np.array(row).reshape(1, -1)

            prediction = model.predict(row)[0]
            # confidence = how many of the forest's trees agree
            probs = model.predict_proba(row)[0]
            confidence = max(probs)

            recent_predictions.append(prediction)
            # majority vote over recent frames = smoothed label
            smoothed = Counter(recent_predictions).most_common(1)[0][0]

            label_text = f"{smoothed} ({confidence*100:.0f}%)"
        else:
            recent_predictions.clear()

        cv2.putText(frame, label_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
