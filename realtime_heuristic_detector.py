import time
import math
import winsound
import cv2
import numpy as np
import mediapipe as mp


# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


# Landmark index groups
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]
UPPER_LIP = 13
LOWER_LIP = 14


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    # Based on dlib EAR concept
    pts = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
    # vertical distances
    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    # horizontal distance
    h = euclidean_distance(pts[0], pts[3])
    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear


def compute_mouth_aspect_ratio(landmarks, frame_w, frame_h):
    upper = (int(landmarks[UPPER_LIP].x * frame_w), int(landmarks[UPPER_LIP].y * frame_h))
    lower = (int(landmarks[LOWER_LIP].x * frame_w), int(landmarks[LOWER_LIP].y * frame_h))
    mar = euclidean_distance(upper, lower) / (frame_h + 1e-6)
    return mar


def main():
    cap = cv2.VideoCapture(0)

    # Thresholds and counters
    eye_closed_threshold = 0.21
    mouth_open_threshold = 0.06
    consecutive_closed_required = 10  # ~10 frames ~ 0.33s at ~30fps
    yawn_frames_required = 12

    closed_counter = 0
    yawn_counter = 0
    last_beep_time = 0.0
    beep_cooldown = 2.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        status_text = ""  # shown on screen
        drowsy_flag = False

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            # EAR
            left_ear = compute_eye_aspect_ratio(face.landmark, LEFT_EYE_LANDMARKS, w, h)
            right_ear = compute_eye_aspect_ratio(face.landmark, RIGHT_EYE_LANDMARKS, w, h)
            ear = (left_ear + right_ear) / 2.0

            # MAR
            mar = compute_mouth_aspect_ratio(face.landmark, w, h)

            if ear < eye_closed_threshold:
                closed_counter += 1
            else:
                closed_counter = 0

            if mar > mouth_open_threshold:
                yawn_counter += 1
            else:
                yawn_counter = 0

            is_blink_or_closed = closed_counter >= consecutive_closed_required
            is_yawning = yawn_counter >= yawn_frames_required

            if is_blink_or_closed or is_yawning:
                drowsy_flag = True

            status_parts = [
                f"EAR: {ear:.3f}",
                f"MAR: {mar:.3f}",
                "EYES CLOSED" if is_blink_or_closed else "EYES OK",
                "YAWN" if is_yawning else "MOUTH OK",
            ]
            status_text = " | ".join(status_parts)

        # Alerts
        if drowsy_flag and (time.time() - last_beep_time) > beep_cooldown:
            winsound.Beep(1200, 300)
            winsound.Beep(900, 300)
            last_beep_time = time.time()

        cv2.putText(
            frame,
            status_text if status_text else "No face",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if drowsy_flag else (0, 255, 0),
            2,
        )
        cv2.imshow("Heuristic Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


