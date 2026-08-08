"""
STEP 1: DATA COLLECTION
-------------------------------------------------
This script opens your webcam, uses MediaPipe to find your hand,
and lets you record example "frames" of each sign you want to teach
the model. Every frame is turned into a row of 63 numbers
(21 landmark points x, y, z) and saved into a CSV file.

HOW TO USE:
1. Edit the GESTURES list below with the names of YOUR signs
   (e.g. ["hello", "thanks", "yes", "no"])
2. Run: python collect_data.py
3. A window opens. Press the NUMBER key shown on screen for the
   gesture you're currently making, and it will save that frame's
   landmarks. Hold the sign steady and press the key repeatedly
   (aim for 150-300 samples per gesture, from slightly different
   angles/distances for a more robust model).
4. Press 'q' to quit and save everything to landmarks.csv
"""

import cv2
import mediapipe as mp
import csv
import os

# --- EDIT THIS: the names of the signs you want to teach the model ---
GESTURES = ["hello", "thanks", "yes"]
# -----------------------------------------------------------------

CSV_PATH = "landmarks.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create the CSV file with a header row if it doesn't exist yet
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

cap = cv2.VideoCapture(0)

counts = {g: 0 for g in GESTURES}

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as hands:

    print("Press the number key for the gesture you're making.")
    for i, g in enumerate(GESTURES):
        print(f"  [{i}] = {g}")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels more natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks_row = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks_row = []
            for lm in hand_landmarks.landmark:
                landmarks_row += [lm.x, lm.y, lm.z]

        # Draw on-screen instructions
        y0 = 30
        for i, g in enumerate(GESTURES):
            text = f"[{i}] {g}: {counts[g]} samples"
            cv2.putText(frame, text, (10, y0 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, y0 + len(GESTURES) * 25 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Check if a number key matching a gesture index was pressed
        if landmarks_row is not None:
            for i, g in enumerate(GESTURES):
                if key == ord(str(i)):
                    with open(CSV_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([g] + landmarks_row)
                    counts[g] += 1
                    print(f"Saved sample for '{g}' (total: {counts[g]})")

cap.release()
cv2.destroyAllWindows()
print("\nDone. Data saved to", CSV_PATH)
print("Sample counts:", counts)
