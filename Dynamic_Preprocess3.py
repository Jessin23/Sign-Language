import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm  # progress bar

# === SETTINGS ===
DATA_PATH = r"C:\CAST\FSL\Dynamic_3"
OUTPUT_PATH = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\processed_dynamic_3"
actions = ['kumusta', 'salamat', 'mahal kita', 'none', 'paalam', 'oo', 'patawad']
sequence_length = 30
use_mirror = False  # optional mirrored augmentation

# === MIRROR FUNCTION ===
def mirror_keypoints(keypoints):
    mirrored = keypoints.copy()
    # Pose (33*4)
    mirrored[:33*4:4] = 1 - mirrored[:33*4:4]
    # Face (468*3)
    start = 33*4
    mirrored[start:start+468*3:3] = 1 - mirrored[start:start+468*3:3]
    # Left hand (21*3)
    start += 468*3
    mirrored[start:start+21*3:3] = 1 - mirrored[start:start+21*3:3]
    # Right hand (21*3)
    start += 21*3
    mirrored[start:start+21*3:3] = 1 - mirrored[start:start+21*3:3]
    return mirrored

# === NORMALIZATION FUNCTION ===
def normalize_keypoints(keypoints):
    keypoints = keypoints.copy()
    # Center by torso (pose landmark 0 = nose)
    if len(keypoints) >= 33*4:
        torso_x = keypoints[0]  # nose x
        torso_y = keypoints[1]  # nose y
        keypoints[0::4] -= torso_x
        keypoints[1::4] -= torso_y
    # Scale by max absolute value
    max_val = np.max(np.abs(keypoints))
    if max_val > 0:
        keypoints /= max_val
    return keypoints

# === PROCESS DATA ===
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"Warning: Folder not found: {action_path}")
        continue

    sequence_dirs = sorted(os.listdir(action_path), key=lambda x: int(x) if x.isdigit() else -1)
    for sequence in tqdm(sequence_dirs, desc=f"Processing '{action}' sequences", unit="seq"):
        seq_path = os.path.join(action_path, sequence)
        if not os.path.isdir(seq_path):
            continue

        window = []
        for frame_num in range(sequence_length):
            npy_file = os.path.join(seq_path, f"{frame_num}.npy")
            if not os.path.exists(npy_file):
                print(f"Missing file: {npy_file}")
                continue
            keypoints = np.load(npy_file)
            keypoints = normalize_keypoints(keypoints)
            window.append(keypoints)

        if len(window) != sequence_length:
            print(f"Incomplete sequence skipped: {seq_path}")
            continue

        sequences.append(window)
        labels.append(action)

        # mirrored version
        if use_mirror:
            mirrored_window = [normalize_keypoints(mirror_keypoints(frame)) for frame in window]
            sequences.append(mirrored_window)
            labels.append(action)

# === CONVERT TO ARRAYS ===
label_map = {label: num for num, label in enumerate(actions)}
y = np.array([label_map[label] for label in labels])
X = np.array(sequences)
y = to_categorical(y).astype(int)

# === SPLIT TRAIN/TEST ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42, stratify=y
)

# === SAVE ===
os.makedirs(OUTPUT_PATH, exist_ok=True)
np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'), y_test)

print(f"Label map: {label_map}")
print(f"Total sequences: {len(sequences)}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Preprocessed data saved to {OUTPUT_PATH}")
