import os
import numpy as np

# === SETTINGS ===
dynamic_dir = r"C:\CAST\FSL\Dynamic"  # folder containing subfolders for each letter
output_base_dir = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\processed_dynamic"
os.makedirs(output_base_dir, exist_ok=True)
y
# === Ask user whether to overwrite all or skip processed ===
overwrite_all = input("Do you want to overwrite already processed letters? (y/n): ").strip().lower() == "y"

# === Normalization function ===
def normalize_landmarks(lm):
    # Translate so wrist (landmark 0) is at origin
    lm -= lm[0]

    # Scale normalization (divide by max distance from wrist)
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0:
        lm /= max_dist

    # Extra: thumb-relative tweak
    # Make thumb position more distinctive relative to index finger
    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]
    lm[4] = (thumb_tip - index_tip) * 1.5  # exaggerate relative thumb-index difference
    lm[3] = (lm[3] - middle_tip) * 1.2     # adjust thumb base a little too

    return lm

# Get all dynamic letters (subfolders)
letters = [d for d in os.listdir(dynamic_dir) if os.path.isdir(os.path.join(dynamic_dir, d))]

for letter in letters:
    output_letter_dir = os.path.join(output_base_dir, letter)
    output_file = os.path.join(output_letter_dir, f"{letter}_all_sequences.npy")

    # Skip if already processed, unless overwrite_all = True
    if os.path.exists(output_file) and not overwrite_all:
        print(f"⚠️ Letter {letter}: already processed, skipping.")
        continue

    letter_dir = os.path.join(dynamic_dir, letter)
    all_sequences = []

    # Recursively walk through the letter folder and all subfolders
    for root, dirs, files in os.walk(letter_dir):
        for file in files:
            if file.endswith(".npy"):
                filepath = os.path.join(root, file)
                seq = np.load(filepath)  # shape: (frames, 21, 3)

                # --- Normalize each frame ---
                seq_norm = np.array([normalize_landmarks(frame) for frame in seq])

                # --- Flatten frames ---
                seq_flat = seq_norm.reshape(seq_norm.shape[0], -1)  # (frames, 63)

                all_sequences.append(seq_flat)

    # --- Combine all sequences for this letter ---
    if all_sequences:
        all_sequences_array = np.stack(all_sequences)  # (num_sequences, frames, 63)

        # --- Create folder for the letter ---
        os.makedirs(output_letter_dir, exist_ok=True)

        # --- Save combined file inside the letter folder ONLY ---
        np.save(output_file, all_sequences_array)
        print(f"✅ Letter {letter}: saved {all_sequences_array.shape} in -> {output_file}")
    else:
        print(f"⚠️ Letter {letter}: no .npy files found")
