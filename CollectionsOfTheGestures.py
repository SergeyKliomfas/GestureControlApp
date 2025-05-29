import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

dataset_dir = "gesture_dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

gestures = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist"]

while True:
    gesture_name = input("Введите название жеста (или 'exit' для выхода): ")
    if gesture_name == "exit":
        break
    if gesture_name not in gestures:
        print("Некорректное название жеста")
        continue

    gesture_path = os.path.join(dataset_dir, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    sample_count = int(input("Сколько образцов записать для этого жеста?: "))

    for sample_index in range(sample_count):
        print(f"\n▶ Подготовьтесь... запись {sample_index + 1}/{sample_count} начнется через 3 секунды")
        time.sleep(3)

        sequence = []
        captured_frames = 0

        while captured_frames < 30:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                sequence.append(np.array(landmarks).flatten())
                captured_frames += 1

                # Отрисовка руки
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"Запись: кадр {captured_frames}/30", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Рука не обнаружена", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Сбор жеста", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(sequence) == 30:
            np.save(os.path.join(gesture_path, f"{len(os.listdir(gesture_path))}.npy"), np.array(sequence))
            print("✔ Образец сохранен.")
        else:
            print("✖ Не удалось собрать 30 кадров. Повторите.")

cap.release()
cv2.destroyAllWindows()

