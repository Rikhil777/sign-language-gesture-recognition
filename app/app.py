import cv2
import numpy as np
import pickle
import os
import sys

# Allow import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_detection.hand_landmark import get_landmarks, draw_landmarks

# Load trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

print("🚀 App started. Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    landmarks = get_landmarks(frame)

    if landmarks:
        features = []

        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

        cv2.putText(
            frame,
            f"Sign: {prediction}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        draw_landmarks(frame, landmarks)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()