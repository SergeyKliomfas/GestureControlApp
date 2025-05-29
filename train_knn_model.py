import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# Путь к датасету
dataset_dir = "gesture_dataset"

# Загрузка данных
X = []
y = []
gestures = [g for g in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, g))]

for i, gesture in enumerate(gestures):
    gesture_path = os.path.join(dataset_dir, gesture)
    for file in os.listdir(gesture_path):
        sequence = np.load(os.path.join(gesture_path, file))
        X.append(sequence)
        y.append(i)


X = np.array(X)
y = np.array(y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Сохранение модели
model.save('gesture_recognition_model.h5')
