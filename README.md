# Real-time Drowsiness Detection System

A computer vision system that monitors user alertness by detecting eye closure and yawning in real-time, providing audio alerts to prevent drowsiness-related accidents.
The system uses pre-trained model ie Mediapipe Face Mesh. NOT trained using any dataset.

## Features

- Real-time face detection and landmark tracking using MediaPipe
- Eye closure detection using Eye Aspect Ratio (EAR)
- Yawning detection using Mouth Aspect Ratio (MAR)
- Audio alerts when drowsiness detected
- Simple and efficient implementation with no external hardware required

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- (Windows only) winsound

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlexTittoZach/Driver-Drowsiness-Detection-using-MediaPipe-n-OpenCV.git
cd Driver-Drowsiness-Detection-using-MediaPipe-n-OpenCV
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the drowsiness detection:
```bash
python main.py
```

Controls:
- Press 'ESC' to exit
- Camera feed shows EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) values
- Red alert appears when drowsiness detected

## Configuration

Adjust detection parameters in `main.py`:
```python
EAR_THRESHOLD = 0.20      # Eye closure sensitivity
EAR_CONSEC_FRAMES = 8     # Frames for eye closure detection
MAR_THRESHOLD = 0.5       # Yawning detection threshold
ALERT_COOLDOWN = 2        # Seconds between alerts
```

## How it Works

1. **Face Detection**: Uses MediaPipe FaceMesh to detect facial landmarks
2. **Eye Monitoring**: Calculates Eye Aspect Ratio (EAR) to detect eye closure
3. **Yawn Detection**: Uses Mouth Aspect Ratio (MAR) to detect yawning
4. **Alert System**: Triggers audio alert when drowsiness patterns detected

## License

[Add your chosen license]

## Contributing

Contributions welcome! Please feel free to submit pull requests.
