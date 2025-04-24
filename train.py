import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

import test

# ==== 1. Load và tiền xử lý dữ liệu ====
data = []
labels = []
NUM_CLASSES = 43
IMAGE_SIZE = 30
train_size = 0.8

cur_path = os.getcwd()
for i in range(NUM_CLASSES):
    path = os.path.join(cur_path, 'train', str(i))
    for img_name in os.listdir(path):
        try:
            img = Image.open(os.path.join(path, img_name))
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            data.append(np.array(img))
            labels.append(i)
        except:
            print(f"Lỗi ảnh ở lớp {i}: {img_name}")

data = np.array(data) / 255.0
labels = np.array(labels)
# shuffle
data, labels = shuffle(data, labels, random_state=42)
# One-hot và chia tập
split_index = int(len(data) * 0.8)

x_train_full = data[:split_index]
x_test = data[split_index:]

y_train_full = labels[:split_index]
y_test = labels[split_index:]

val_split_index = int(len(x_train_full) * 0.2)

x_val = x_train_full[:val_split_index]
y_val = y_train_full[:val_split_index]

x_train = x_train_full[val_split_index:]
y_train = y_train_full[val_split_index:]


y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)

print("y_test sample:", labels[len(y_train):10 + len(y_train)])
print("NUM_CLASSES:", NUM_CLASSES)
model = Sequential([
    # Layer 1: Convolution + ReLU + MaxPooling
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(30, 30, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # Layer 2: Convolution + ReLU + MaxPooling
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    # Layer 3: Convolution + ReLU + MaxPooling
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten + Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # chống overfitting

    # Output layer: softmax cho 43 lớp
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
model.save("my_model.keras")
# Accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy over epochs')
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()
loss, acc = model.evaluate(x_test, y_test)
print(f"Accuracy trên tập test: {acc * 100:.2f}% và loss: {loss* 100:.2f}%")


