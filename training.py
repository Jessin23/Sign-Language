import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# === SETTINGS ===
DATA_PATH = r"C:\CAST\CAST_preprocessed\FSL_preprocessed\processed_dynamic_3"
MODEL_PATH = r"C:\CAST\trained_models"
actions = ['kumusta', 'salamat', 'mahal kita', 'none', 'paalam', 'oo', 'patawad']
sequence_length = 30
batch_size = 16
epochs = 200
learning_rate = 0.001

# === LOAD DATA ===
X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# === MODEL CALLBACKS ===
os.makedirs(MODEL_PATH, exist_ok=True)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

checkpoint_path = os.path.join(MODEL_PATH, "fsl_phrase_final.h5")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_categorical_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop_callback = EarlyStopping(
    monitor="val_categorical_accuracy",
    patience=30,
    restore_best_weights=True,
    mode="max",
    verbose=1
)

# === CNN-LSTM MODEL ===
model = Sequential()

# 1D Convolution layers (temporal feature extraction)
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# LSTM layers (temporal sequence learning)
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))

# Dense layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# === TRAIN MODEL ===
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, checkpoint_callback, early_stop_callback],
    verbose=1
)

# === SAVE FINAL MODEL ===
final_model_path = os.path.join(MODEL_PATH, 'fsl_phrase_final.h5')
model.save(final_model_path)
print(f"✅ Training complete. Best model saved at {checkpoint_path}, final model saved at {final_model_path}")
