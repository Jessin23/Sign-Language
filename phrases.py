import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import os
import time

# === SETTINGS ===
actions = ['kumusta', 'salamat', 'mahal kita', 'none', 'paalam', 'oo', 'patawad']
sequence_length = 30
threshold = 0.5
smoothing_window = 10
required_consistency = 8
prediction_interval = 0.25  # seconds between predictions

MODEL_PATH = r"C:\CAST\trained_models\fsl_phrase_final.h5"

# === LOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# === MEDIA PIPE SETUP ===
mp_holistic = mp.solutions.holistic

# === NORMALIZATION FUNCTION ===
def normalize_keypoints(keypoints):
    keypoints = keypoints.copy()
    if len(keypoints) >= 33*4:
        torso_x = keypoints[0]  # nose x
        torso_y = keypoints[1]  # nose y
        keypoints[0::4] -= torso_x
        keypoints[1::4] -= torso_y
    max_val = np.max(np.abs(keypoints))
    if max_val > 0:
        keypoints /= max_val
    return keypoints

# === KEYPOINT EXTRACTION ===
def extract_keypoints(results):
    if not (results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks):
        return None  # No landmarks detected
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    keypoints = np.concatenate([pose, face, lh, rh])
    return normalize_keypoints(keypoints)

# === HELPER FUNCTION ===
def mediapipe_detection(image, holistic_model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic_model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

# === LIVE DETECTION ===
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=sequence_length)
predictions_prob = deque(maxlen=smoothing_window)
current_class = None
switch_count = 0
last_prediction_time = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = mediapipe_detection(frame, holistic)

        # Extract keypoints & build sequence
        keypoints = extract_keypoints(results)

        # If no landmarks, reset sequence and predict "none"
        if keypoints is None:
            sequence.clear()
            predictions_prob.clear()
            current_class = "none"
            avg_res = np.zeros(len(actions))
            avg_res[actions.index("none")] = 1.0
        else:
            sequence.append(keypoints)
            avg_res = np.zeros(len(actions))
            if len(sequence) == sequence_length:
                if (time.time() - last_prediction_time) >= prediction_interval:
                    seq_array = np.expand_dims(np.array(sequence), axis=0)
                    res = model.predict(seq_array, verbose=0)[0]
                    predictions_prob.append(res)
                    last_prediction_time = time.time()

                if len(predictions_prob) > 0:
                    avg_res = np.mean(predictions_prob, axis=0)
                    max_idx = np.argmax(avg_res)

                    # Class switching logic
                    if avg_res[max_idx] > threshold:
                        if current_class != actions[max_idx]:
                            switch_count += 1
                            if switch_count >= required_consistency:
                                current_class = actions[max_idx]
                                switch_count = 0
                        else:
                            switch_count = 0
                    else:
                        current_class = None
                        switch_count = 0

        # Display predictions only
        y0, dy = 50, 30
        for i, (action_name, prob) in enumerate(zip(actions, avg_res)):
            cv2.putText(frame, f"{action_name}: {prob*100:.1f}%", (10, y0 + i*dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        display_text = f"Predicted: {current_class}" if current_class else "Predicted: None"
        cv2.rectangle(frame, (0,0), (640,40), (245,117,16), -1)
        cv2.putText(frame, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("ASL Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
