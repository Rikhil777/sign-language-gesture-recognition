import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize once
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def get_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0]
    return None

def draw_landmarks(image, landmarks):
    mp_draw.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)