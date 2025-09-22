import os
import numpy as np
import json
from sklearn.model_selection import train_test_split

frames = 30
dynamic_base = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\processed_dynamic"
output_dir = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\merged_dataset"
label_map_file = os.path.join(output_dir, "label_map.json")

os.makedirs(output_dir, exist_ok=True)

# Load previous label map if exists
if os.path.exists(label_map_file):
    with open(label_map_file, "r") as f:
        label_map = json.load(f)
else:
    label_map = {}  # empty dict

X, y = [], []
current_index = max([int(i) for i in label_map.keys()], default=-1) + 1

# Loop through all subfolders
for folder_name in sorted(os.listdir(dynamic_base)):
    folder_path = os.path.join(dynamic_base, folder_name)
    if not os.path.isdir(folder_path):
        continue

    npy_file = os.path.join(folder_path, f"{folder_name}_all_sequences.npy")
    if not os.path.exists(npy_file):
        continue

    data = np.load(npy_file)  # shape: (num_seq, frames, 63)

    # Assign label index
    if folder_name in label_map.values():
        # Find existing key
        label_index = next(k for k, v in label_map.items() if v == folder_name)
    else:
        label_index = str(current_index)
        label_map[label_index] = folder_name
        current_index += 1

    X.extend(data)
    y.extend([int(label_index)] * len(data))

    print(f"✔ Loaded {folder_name} -> label {label_index}, samples: {len(data)}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save datasets
np.savez(os.path.join(output_dir, "fsl_train_dataset.npz"), X=X_train, y=y_train)
np.savez(os.path.join(output_dir, "fsl_val_dataset.npz"), X=X_val, y=y_val)
np.savez(os.path.join(output_dir, "fsl_test_dataset.npz"), X=X_test, y=y_test)

# Save updated label map
with open(label_map_file, "w") as f:
    json.dump(label_map, f, indent=2)

print("✅ Merge complete!")
print("Training set - X:", X_train.shape, "y:", y_train.shape)
print("Validation set - X:", X_val.shape, "y:", y_val.shape)
print("Testing set - X:", X_test.shape, "y:", y_test.shape)
print("Label map saved to", label_map_file)
