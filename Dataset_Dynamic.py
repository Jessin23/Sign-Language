import os
import cv2
import numpy as np
import mediapipe as mp
import time

# === SETTINGS ===
DATA_PATH = r"C:\CAST\ASL\Dynamic_3"
actions = ['I love you']  
total_sequences_per_action = 200
sequence_length = 30
checkpoint_interval = 50   
pre_sequence_delay = 1     

# === MEDIA PIPE INIT ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# === KEYPOINT EXTRACTION ===
def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# === DATA COLLECTION ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opening")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        print(f"\n📌 Collecting data for action: '{action}'")

        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path, exist_ok=True)
        existing_dirs = [int(i) for i in os.listdir(action_path) if i.isdigit()]
        dirmax = max(existing_dirs) if existing_dirs else 0

        seq_counter = 1
        while seq_counter <= total_sequences_per_action:

            # === Wait for key press before starting batch (or after checkpoint) ===
            if (seq_counter == 1) or ((seq_counter-1) % checkpoint_interval == 0):
                while True:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Press any key to start next batch of sequences for '{action}'", 
                                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    cv2.imshow('ASL Data Collector', frame)
                    key = cv2.waitKey(1)
                    if key != -1:
                        break

            # 1-second delay before recording each sequence
            start_time = time.time()
            while time.time() - start_time < pre_sequence_delay:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f'Starting Seq {seq_counter} in {pre_sequence_delay - int(time.time() - start_time)}s', 
                            (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.imshow('ASL Data Collector', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Record frames for this sequence
            seq_dir = os.path.join(action_path, str(dirmax + seq_counter))
            os.makedirs(seq_dir, exist_ok=True)

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cv2.putText(frame, f'Collecting {action} Seq {seq_counter} Frame {frame_num}', 
                            (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imshow('ASL Data Collector', frame)

                # Save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(seq_dir, f'{frame_num}.npy')
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            seq_counter += 1

            # === Checkpoint pause ===
            if seq_counter % checkpoint_interval == 1 and seq_counter <= total_sequences_per_action:
                print(f"\n⏸️  Checkpoint reached at sequence {seq_counter-1}. Adjust camera angle now.")
                while True:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Adjust camera angle and press any key to continue", 
                                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    cv2.imshow('ASL Data Collector', frame)
                    key = cv2.waitKey(1)
                    if key != -1:
                        break

cap.release()
cv2.destroyAllWindows()
print("✅ Data collection completed for all actions!")
