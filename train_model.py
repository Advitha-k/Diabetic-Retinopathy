import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

CSV_PATH = "train.csv"
IMAGE_DIR = "train_images"
IMG_SIZE = 224
SAMPLE_SIZE = 800
NUM_CLASSES = 5

df = pd.read_csv(CSV_PATH)
df = df.sample(SAMPLE_SIZE, random_state=42)

def load_image(image_id):
    img_path = os.path.join(IMAGE_DIR, image_id + ".png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

X, y = [], []

for _, row in df.iterrows():
    X.append(load_image(row["id_code"]))
    y.append(row["diagnosis"])

X = np.array(X)
y = to_categorical(np.array(y), NUM_CLASSES)

# Train / Validation split (manual, avoids sklearn)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16
)

model.save("dr_model.h5")
print("Model saved as dr_model.h5")
