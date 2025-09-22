import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Conv1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === Paths ===
train_data_path = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\merged_dataset\fsl_train_dataset.npz"
val_data_path = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\merged_dataset\fsl_val_dataset.npz"
test_data_path = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\merged_dataset\fsl_test_dataset.npz"
label_map_file = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\merged_dataset\label_map.json"
output_model_dir = r"C:\CAST\trained_models"

os.makedirs(output_model_dir, exist_ok=True)

# === Load label map ===
with open(label_map_file, "r") as f:
    label_map = json.load(f)

num_classes = len(label_map)
print(f"✅ Loaded label map with {num_classes} classes")

# === Load datasets ===
train_loaded = np.load(train_data_path, allow_pickle=True)
X_train, y_train = train_loaded["X"], train_loaded["y"]

val_loaded = np.load(val_data_path, allow_pickle=True)
X_val, y_val = val_loaded["X"], val_loaded["y"]

test_loaded = np.load(test_data_path, allow_pickle=True)
X_test, y_test = test_loaded["X"], test_loaded["y"]

print(f"✅ Dataset shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape},   y_val: {y_val.shape}")
print(f"  X_test:  {X_test.shape},  y_test: {y_test.shape}")

# === One-hot encode labels ===
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# === Build model ===
model = Sequential([
    Masking(mask_value=0.0, input_shape=(30, 63)),
    Conv1D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# === Callbacks ===
checkpoint_path = os.path.join(output_model_dir, "fsl_letter_best.h5")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# === Train ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

# === Evaluate & Save final model ===
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}")

final_model_path = os.path.join(output_model_dir, "fsl_letter_final.h5")
model.save(final_model_path)
print(f"✅ Final model saved to {final_model_path}")
