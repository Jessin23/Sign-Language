import os
import numpy as np
from tqdm import tqdm

# === SETTINGS ===
DATA_PATH = r"C:\CAST\ASL\Dynamic_3"   # folder with original sequences
actions = ['I love you']
sequence_length = 30
num_augments = 3                        # number of augmented copies per sequence

# === AUGMENTATION FUNCTIONS ===
def add_noise(seq, noise_std=0.02):
    return seq + np.random.normal(0, noise_std, seq.shape)

def scale_keypoints(seq, scale_range=(0.9, 1.1)):
    factor = np.random.uniform(*scale_range)
    return seq * factor

def rotate_keypoints(seq, angle_range=(-10, 10)):
    angle_deg = np.random.uniform(*angle_range)
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad),  np.cos(angle_rad)]])
    rotated_seq = np.zeros_like(seq)
    for i in range(seq.shape[0]):
        frame = seq[i].reshape(-1, 3)  # x,y,z
        xy = frame[:, :2].dot(rotation_matrix.T)
        frame[:, :2] = xy
        rotated_seq[i] = frame.flatten()
    return rotated_seq

def augment_pipeline(seq):
    seq = add_noise(seq)
    seq = scale_keypoints(seq)
    seq = rotate_keypoints(seq)
    return seq

# === PROCESSING ===
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequence_dirs = [d for d in os.listdir(action_path) if d.isdigit()]
    sequence_dirs.sort(key=lambda x: int(x))

    for seq_dir in tqdm(sequence_dirs, desc=f"Augmenting action: '{action}'"):
        seq_path = os.path.join(action_path, seq_dir)
        frames = []
        for frame_num in range(sequence_length):
            frame_file = os.path.join(seq_path, f"{frame_num}.npy")
            frame = np.load(frame_file)
            frames.append(frame)
        seq_array = np.array(frames)

        # === Generate multiple augmentations ===
        for aug_idx in range(1, num_augments + 1):
            aug_seq = augment_pipeline(seq_array)
            aug_seq_dir = f"{seq_path}_aug{aug_idx}"
            os.makedirs(aug_seq_dir, exist_ok=True)
            for i, frame in enumerate(aug_seq):
                np.save(os.path.join(aug_seq_dir, f"{i}.npy"), frame)

print("✅ Augmentation completed.")
