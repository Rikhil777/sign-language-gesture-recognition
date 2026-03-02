import cv2
import os
import csv
import sys

# Allow import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_detection.hand_landmark import get_landmarks

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "features.csv")

# Prepare CSV header
header = []
for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])
header.append("label")

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # Loop through gesture folders
    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Processing {label}...")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            landmarks = get_landmarks(image)

            if landmarks:
                row = []
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)
                writer.writerow(row)

print("✅ Feature extraction completed. Saved to features.csv")