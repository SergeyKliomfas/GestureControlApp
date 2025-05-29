import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv


load_dotenv()

scope = "user-modify-playback-state"

scope = "user-modify-playback-state"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope=scope
))


model = load_model('gesture_recognition_model.h5')
gestures = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "fist"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Gesture Music Controller")
root.geometry("400x250")

status_label = tk.Label(root, text="Status: Waiting", font=("Arial", 16))
status_label.pack(pady=10)


gesture_info = """
🖐️ Для переключения следующего трека - проведите рукой против часовой стрелки

🖐️ Для переключения предыдущего трека - проведите рукой по часовой стрелки

⬆️ Для увеличения громкости - проведите ладонью сверху вниз

⬇️ Для понижения громкости - проведите ладонью снизу вверх

✊ Для паузы - сожмите ладонь в кулак
"""

def show_gesture_help():
    messagebox.showinfo("Поддерживаемые жесты", gesture_info)

help_button = tk.Button(root, text="ℹ️ Помощь", command=show_gesture_help)
help_button.pack()


buffer = []


def update_status(message):
    status_label.config(text=f"Status: {message}")

def gesture_loop():
    ret, frame = cap.read()
    if not ret:
        root.after(10, gesture_loop)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        buffer.append(np.array(landmarks).flatten())
    else:
        buffer.append(np.zeros(63))

    if len(buffer) > 30:
        buffer.pop(0)

    if len(buffer) == 30:
        real_count = sum(1 for item in buffer if np.count_nonzero(item) > 0)
        if real_count >= 20:
            sequence = np.array(buffer).reshape(1, 30, 63)
            prediction = model.predict(sequence, verbose=0)
            gesture_index = np.argmax(prediction)
            recognized_gesture = gestures[gesture_index]
            print(f"[INFO] Gesture: {recognized_gesture} ({real_count}/30 valid frames)")

            try:
                if recognized_gesture == "swipe_left":
                    sp.previous_track()
                    update_status("Previous Track")
                elif recognized_gesture == "swipe_right":
                    sp.next_track()
                    update_status("Next Track")
                elif recognized_gesture == "swipe_up":
                    volume = sp.current_playback()['device']['volume_percent']
                    sp.volume(min(volume + 10, 100))
                    update_status("Volume Up")
                elif recognized_gesture == "swipe_down":
                    volume = sp.current_playback()['device']['volume_percent']
                    sp.volume(max(volume - 10, 0))
                    update_status("Volume Down")
                elif recognized_gesture == "fist":
                    playback = sp.current_playback()
                    if playback and playback['is_playing']:
                        sp.pause_playback()
                        update_status("Paused")
                    else:
                        sp.start_playback()
                        update_status("Playing")
            except Exception as e:
                print(f"[ERROR] {e}")
        else:
            print(f"[INFO] Not enough hand frames ({real_count}/30) — skipping prediction")

        buffer.clear()

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.destroy()
        return

    root.after(1, gesture_loop)

# --- Запуск ---
root.after(0, gesture_loop)
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), cv2.destroyAllWindows(), root.destroy()))
root.mainloop()












