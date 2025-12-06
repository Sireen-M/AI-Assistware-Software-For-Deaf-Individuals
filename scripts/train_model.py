import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tcn import TCN
import tensorflow as tf

# Load data
X = np.load('scripts/X.npy')
y = np.load('scripts/y.npy')
classes = np.load('scripts/classes.npy')

print(f"🔹 Loaded data shapes: X={X.shape}, y={y.shape}, num_classes={len(classes)}")

# Train/Val split WITHOUT stratify
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Sequential([
    TCN(input_shape=(X.shape[1], X.shape[2])),  # TCN expects (timesteps, features)
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint('models/sign_model_tcn.h5', monitor='val_accuracy', save_best_only=True)

# Train
print("\n🚀 Training TCN model...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[checkpoint]
)

print("✅ Training complete. Best model saved to: models/sign_model_tcn.h5")
