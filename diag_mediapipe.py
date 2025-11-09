import mediapipe as mp
print("mediapipe imported, version:", mp.__version__)
mpf = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
print("FaceMesh created OK")
mpf.close()