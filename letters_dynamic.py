import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import ImageFont, ImageDraw, Image  # Pillow for Unicode text

# === SETTINGS ===
letter = "J"  
frames_per_sequence = 30
target_real_sequences = 250
base_dir = r"C:\CAST\FSL\Dynamic"
save_dir = os.path.join(base_dir, letter)
os.makedirs(save_dir, exist_ok=True)

# Delay between sequences in auto mode (seconds)
use_delay = True
delay_seconds = 1.0

# === Mediapipe Hands setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

# === Load a font that supports Ñ ===
font_path = "arial.ttf"  # Change if needed
font_large = ImageFont.truetype(font_path, 32)
font_medium = ImageFont.truetype(font_path, 28)

# === Helper: draw Unicode text ===
def draw_unicode_text(frame, text, position, font, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# === Helper: augment a sequence ===
def augment_sequence(seq):
    augmented = []

    # Gaussian noise
    noise = seq + np.random.normal(0, 0.01, seq.shape)
    augmented.append(noise)

    # Slight scaling
    center = np.mean(seq[:, :, :2], axis=1, keepdims=True)
    scaled = (seq[:, :, :2] - center) * 1.05 + center
    scaled_seq = seq.copy()
    scaled_seq[:, :, :2] = scaled
    augmented.append(scaled_seq)

    # Mirror
    mirrored = seq.copy()
    mirrored[:, :, 0] = 1 - mirrored[:, :, 0]
    augmented.append(mirrored)

    return augmented

# === MAIN LOOP ===
sequence_count = len([f for f in os.listdir(save_dir)
                      if f.endswith(".npy") and "_aug" not in f])
print(f"Starting collection for {letter}. Already have {sequence_count} real sequences.")

collecting = False
auto_mode = False
frames = []
frame_counter = 0
last_save_time = 0  # timer for auto mode delay

while cap.isOpened() and sequence_count < target_real_sequences:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay text
    frame = draw_unicode_text(frame, f"Letter: {letter}", (10, 40), font_large, (0, 255, 0))
    frame = draw_unicode_text(frame, f"Sequences: {sequence_count}/{target_real_sequences}", (10, 80), font_large, (0, 255, 0))
    if auto_mode:
        frame = draw_unicode_text(frame, "AUTO MODE (press 'a' to stop)", (10, 120), font_medium, (255, 165, 0))
    if collecting or (auto_mode and frame_counter > 0):
        frame = draw_unicode_text(frame, f"Recording frame: {frame_counter}/{frames_per_sequence}", (10, 160), font_medium, (255, 0, 0))

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # === Keyboard controls ===
    if key == ord('s') and not collecting:
        frames, frame_counter = [], 0
        collecting = True
        print("\n▶ Recording new sequence (manual)...")

    if key == ord('a'):
        auto_mode = not auto_mode
        if auto_mode:
            print("\n⚡ AUTO MODE ON: recording continuously...")
            last_save_time = 0  # reset delay timer
        else:
            print("\n⏹ AUTO MODE OFF")

    if key == ord('q'):
        break

    # === Recording logic ===
    if collecting or auto_mode:
        if auto_mode and use_delay:
            # enforce non-blocking delay between auto captures
            if time.time() - last_save_time < delay_seconds and frame_counter == 0:
                continue

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
        else:
            landmarks = [[0, 0, 0] for _ in range(21)]

        frames.append(landmarks)
        frame_counter += 1

        if frame_counter == frames_per_sequence:
            seq_array = np.array(frames)

            # Save original
            seq_id = f"{letter}_{sequence_count}"
            np.save(os.path.join(save_dir, f"{seq_id}.npy"), seq_array)

            # Save augmentations
            aug_sequences = augment_sequence(seq_array)
            for j, aug in enumerate(aug_sequences):
                aug_id = f"{letter}_{sequence_count}_aug{j}"
                np.save(os.path.join(save_dir, f"{aug_id}.npy"), aug)

            sequence_count += 1
            last_save_time = time.time()  # update delay timer
            print(f"✔ Saved sequence {sequence_count} with augmentations.")

            frames, frame_counter = [], 0
            collecting = False  # stop manual recording in manual mode

cap.release()
cv2.destroyAllWindows()
