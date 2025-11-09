import os
import time
import json
import cv2
import math
import numpy as np
import mediapipe as mp
import winsound
from tensorflow.keras.models import load_model


MODEL_PATH = os.path.join("models", "drowsiness_mobilenetv2.keras")
LABELS_PATH = os.path.join("models", "class_indices.json")
IMG_SIZE = 224


def load_labels(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        # Reverse mapping: index -> class_name
        index_to_class = {v: k for k, v in class_indices.items()}
        return index_to_class
    except Exception:
        return {0: "Non Drowsy", 1: "Drowsy"}


def preprocess_bgr(image_bgr):
    image = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def crop_face_from_landmarks(frame, landmarks, frame_w, frame_h, margin=20):
    xs = [int(lm.x * frame_w) for lm in landmarks]
    ys = [int(lm.y * frame_h) for lm in landmarks]
    x_min, x_max = max(min(xs) - margin, 0), min(max(xs) + margin, frame_w)
    y_min, y_max = max(min(ys) - margin, 0), min(max(ys) + margin, frame_h)
    return frame[y_min:y_max, x_min:x_max]


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first with 'train_drowsiness_model.py'.")
        return

    model = load_model(MODEL_PATH)
    index_to_class = load_labels(LABELS_PATH)

    # Determine which index corresponds to Drowsy
    # Heuristic: look for key containing 'drowsy' case-insensitive
    drowsy_index = None
    for idx, name in index_to_class.items():
        if "drowsy" in name.lower():
            drowsy_index = idx
            break
    # Fallback to 1 if not found
    if drowsy_index is None:
        drowsy_index = 1

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    # Temporal smoothing
    drowsy_threshold = 0.6
    required_consecutive = 8
    cooldown_seconds = 2.0

    consecutive_drowsy = 0
    last_alert_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        status_text = "No face"
        color = (0, 255, 0)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            face_roi = crop_face_from_landmarks(frame, lm, w, h, margin=28)
            if face_roi.size == 0:
                consecutive_drowsy = 0
            else:
                inp = preprocess_bgr(face_roi)
                pred = model.predict(inp, verbose=0)[0][0]
                # Model is binary sigmoid: probability of class index 1 by convention
                # If dataset mapping puts Drowsy at index 0, invert
                prob_drowsy = pred if drowsy_index == 1 else (1.0 - pred)

                if prob_drowsy >= drowsy_threshold:
                    consecutive_drowsy += 1
                else:
                    consecutive_drowsy = 0

                is_drowsy = consecutive_drowsy >= required_consecutive
                status_text = f"Drowsy p={prob_drowsy:.2f}  streak={consecutive_drowsy}"
                color = (0, 0, 255) if is_drowsy else (0, 255, 0)

                now = time.time()
                if is_drowsy and (now - last_alert_time) > cooldown_seconds:
                    winsound.Beep(1200, 300)
                    winsound.Beep(900, 300)
                    last_alert_time = now

        cv2.putText(
            frame,
            status_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.imshow("Model Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


