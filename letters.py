import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import json
import time
from PIL import ImageFont, ImageDraw, Image  # Pillow for Unicode text

# === Normalization function ===
def normalize_landmarks(lm):
    lm -= lm[0]
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0:
        lm /= max_dist

    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]
    lm[4] = (thumb_tip - index_tip) * 1.5
    lm[3] = (lm[3] - middle_tip) * 1.2
    return lm

# === Paths ===
model_path = r"C:\CAST\trained_models\asl_letter_final.h5"
label_map_file = r"C:\CAST\CAST_preprocessed\aSL_preprocessed\merged_dataset\label_map.json"

# Load label map (with fixed ordering)
with open(label_map_file, "r", encoding="utf-8") as f:
    label_map = json.load(f)

class_labels = [label_map[str(i)] for i in sorted(map(int, label_map.keys()))]
print("✅ Ordered class labels:", class_labels)

# Settings
confidence_threshold = 0.5
sequence_length = 30
smoothing_window = 5
prediction_interval = 0.25  # seconds between predictions

# Load model
model = load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Buffers
sequence = deque(maxlen=sequence_length)
predictions = deque(maxlen=smoothing_window)

# Typing variables
accumulated_text = ""
last_typed_char = ""   # prevents spamming
last_prediction_time = 0

# Pause detection variables
last_hand_time = time.time()
pause_threshold = 2.0  # seconds without hand → insert space

# Load Unicode font
try:
    font_path = "C:/Windows/Fonts/arial.ttf"
    font_large = ImageFont.truetype(font_path, 36)
    font_medium = ImageFont.truetype(font_path, 28)
except Exception as e:
    print(f"⚠️ Font not found, using default. Error: {e}")
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()

def draw_unicode_text(frame, texts, font, colors, positions):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, pos, color in zip(texts, positions, colors):
        draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🚀 Starting sign language detection. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        accumulated_text = ""  # clear text

    # Canvas for video + text
    h, w, _ = frame.shape
    text_height = 150
    canvas = np.zeros((h + text_height, w, 3), dtype=np.uint8)
    canvas[:h, :, :] = frame

    status_texts, status_colors, status_positions = [], [], []

    if result.multi_hand_landmarks:
        last_hand_time = time.time()  # reset pause timer

        hand_landmarks = result.multi_hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark])
        lm = normalize_landmarks(lm)
        sequence.append(lm.flatten())
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(sequence) == sequence_length:
            if (time.time() - last_prediction_time) >= prediction_interval:
                seq_array = np.expand_dims(np.array(sequence), 0)
                pred = model.predict(seq_array, verbose=0)[0]  # shape: (num_classes,)
                pred_prob = np.max(pred)
                pred_class = class_labels[np.argmax(pred)]

                predictions.append(pred_class if pred_prob >= confidence_threshold else "unknown")
                last_prediction_time = time.time()

                # === Print class probabilities in terminal ===
                print("\n🔎 Prediction probabilities:")
                for cls, prob in sorted(zip(class_labels, pred), key=lambda x: x[1], reverse=True):
                    print(f"  {cls}: {prob:.4f}")

            if len(predictions) == smoothing_window:
                most_common, count = Counter(predictions).most_common(1)[0]
                if most_common != "unknown" and count >= (smoothing_window // 2):
                    if most_common != last_typed_char:
                        if most_common.lower() == "backspace":
                            if len(accumulated_text) > 0:
                                accumulated_text = accumulated_text[:-1]
                                print("⌫ Backspace detected -> removed last char")
                        elif most_common.isalnum() or most_common in ["Ñ", "ñ"]:
                            accumulated_text += most_common
                        last_typed_char = most_common

                    status_texts.append(f"Predicted: {most_common}")
                    status_colors.append((0, 255, 0))
                    status_positions.append((10, 40))
                else:
                    status_texts.append("Unknown sign language")
                    status_colors.append((255, 165, 0))
                    status_positions.append((10, 40))
            else:
                status_texts.append("Collecting predictions...")
                status_colors.append((255, 255, 0))
                status_positions.append((10, 40))
        else:
            status_texts.append(f"Collecting frames: {len(sequence)}/{sequence_length}")
            status_colors.append((255, 255, 0))
            status_positions.append((10, 40))
    else:
        sequence.clear()
        predictions.clear()
        last_typed_char = ""
        status_texts.append("No hand detected")
        status_colors.append((255, 0, 0))
        status_positions.append((10, 40))

        # Auto-space
        if (time.time() - last_hand_time) >= pause_threshold:
            if accumulated_text != "" and not accumulated_text.endswith(" "):
                accumulated_text += " "
                print("␣ Auto-space inserted (pause detected)")
            last_hand_time = time.time()

    # Draw status text on screen
    canvas = draw_unicode_text(canvas, status_texts, font_large, status_colors, status_positions)

    # Draw accumulated text with word wrapping
    y_offset = h + 50
    max_width = w - 40
    words = accumulated_text.split(" ")
    line = ""

    for word in words:
        test_line = line + word + " "
        test_img = Image.new("RGB", (1000, 100))
        test_draw = ImageDraw.Draw(test_img)
        bbox = test_draw.textbbox((0, 0), test_line, font=font_large)
        text_width = bbox[2] - bbox[0]

        if text_width > max_width:
            canvas = draw_unicode_text(canvas, [line], font_large, [(255, 255, 255)], [(20, y_offset)])
            y_offset += 40
            line = word + " "
        else:
            line = test_line

    canvas = draw_unicode_text(canvas, [line], font_large, [(255, 255, 255)], [(20, y_offset)])

    cv2.imshow("Sign Language to Text", canvas)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Sign language detection ended.")
