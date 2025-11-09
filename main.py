import time
import cv2
import numpy as np
import mediapipe as mp
import platform

try:
    import winsound  # Windows beep
except Exception:
    winsound = None

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark indices for EAR & MAR (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [13, 14, 78, 308]  # upper, lower, left corner, right corner

def calculate_ear(landmarks, eye_indices, image_w, image_h):
    pts = [landmarks.landmark[i] for i in eye_indices]
    coords = np.array([(p.x * image_w, p.y * image_h) for p in pts], dtype=np.float32)
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, mouth_indices, image_w, image_h):
    pts = [landmarks.landmark[i] for i in mouth_indices]
    coords = np.array([(p.x * image_w, p.y * image_h) for p in pts], dtype=np.float32)
    vertical = np.linalg.norm(coords[0] - coords[1])
    horizontal = np.linalg.norm(coords[2] - coords[3])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

# Detection parameters (tune as needed)
EAR_THRESHOLD = 0.20         # Eye ratio threshold (higher = more sensitive)
EAR_CONSEC_FRAMES = 8        # Reduced: detect eye closure faster
MAR_THRESHOLD = 0.5          # Mouth ratio threshold for yawning
MAR_CONSEC_FRAMES = 10       # Frames mouth must be open
ALERT_COOLDOWN = 2           # Reduced: allow more frequent alerts

def play_beep():
    # Cross-platform simple beep: use winsound on Windows, otherwise fallback to printing
    if winsound and platform.system() == 'Windows':
        for _ in range(2):  # Shorter beep sequence
            winsound.Beep(2000, 200)  # Shorter beep duration
            time.sleep(0.1)
    else:
        # Try using terminal bell as fallback (may be quiet). For production on Pi, replace with GPIO buzzer.
        for _ in range(2):
            print('\a', end='', flush=True)
            time.sleep(0.1)

def main():
    cap = cv2.VideoCapture(0)
    eye_closed_counter = 0
    mar_counter = 0
    last_alert_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        info_text = 'No face'
        trigger_alert = False

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            ear_l = calculate_ear(lm, LEFT_EYE, w, h)
            ear_r = calculate_ear(lm, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2.0
            mar = calculate_mar(lm, MOUTH, w, h)

            # update counters
            if ear < EAR_THRESHOLD:
                eye_closed_counter += 1
            else:
                eye_closed_counter = 0

            if mar > MAR_THRESHOLD:
                mar_counter += 1
            else:
                mar_counter = 0

            info_text = f'EAR={ear:.2f}  MAR={mar:.2f}'

            now = time.time()
            if (eye_closed_counter >= EAR_CONSEC_FRAMES) or (mar_counter >= MAR_CONSEC_FRAMES):
                if now - last_alert_time > ALERT_COOLDOWN:
                    trigger_alert = True
                    last_alert_time = now

            # draw landmarks
            mp_draw.draw_landmarks(frame, lm, mp_face_mesh.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=None,
                                   connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
            mp_draw.draw_landmarks(frame, lm, mp_face_mesh.FACEMESH_CONTOURS,
                                   landmark_drawing_spec=None,
                                   connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())

        if trigger_alert:
            cv2.putText(frame, 'DROWSINESS ALERT - WAKE UP!', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            play_beep()
        else:
            cv2.putText(frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Smart Helmet - Drowsiness Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Load MobileNet models (ensure these are trained for your use-case)
eye_model = load_model('mobilenet_eye.h5')     # open/closed classifier
mouth_model = load_model('mobilenet_mouth.h5') # yawn/no yawn classifier

# Landmark indices for cropping
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [13, 14, 78, 308, 82, 312]

def crop_region(frame, landmarks, indices, w, h, margin=10):
    coords = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices])
    x, y, w_, h_ = cv2.boundingRect(coords)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w_ = w_ + 2 * margin
    h_ = h_ + 2 * margin
    return frame[y:y+h_, x:x+w_]

def preprocess(img):
    if img is None or img.size == 0:
        # Return a blank image if crop failed
        return np.zeros((224, 224, 3), dtype=np.float32)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = ""
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Crop left and right eye regions, combine for better accuracy
            left_eye_img = crop_region(frame, face_landmarks, LEFT_EYE, w, h)
            right_eye_img = crop_region(frame, face_landmarks, RIGHT_EYE, w, h)
            mouth_img = crop_region(frame, face_landmarks, MOUTH, w, h)

            # Preprocess for MobileNet
            left_eye_input = preprocess(left_eye_img)
            right_eye_input = preprocess(right_eye_img)
            mouth_input = preprocess(mouth_img)

            try:
                # Predict eye state for both eyes, average the result
                left_eye_pred = eye_model.predict(left_eye_input)[0][0]
                right_eye_pred = eye_model.predict(right_eye_input)[0][0]
                eye_score = (left_eye_pred + right_eye_pred) / 2.0
                eye_state = "Closed" if eye_score > 0.5 else "Open"

                # Predict mouth state
                mouth_pred = mouth_model.predict(mouth_input)[0][0]
                mouth_state = "Yawning" if mouth_pred > 0.5 else ""

                status = f"Eyes {eye_state} | {mouth_state}"
            except Exception as e:
                status = "Detection Error"

            # Draw status and landmarks
            cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255) if "Closed" in status or "Yawning" in status else (0,255,0), 2)

            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )

    cv2.imshow('MobileNet Drowsiness Detection', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
