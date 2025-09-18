# Face Detection and Characteristics Display

This project contains multiple Python scripts for detecting human faces using your webcam, highlighting them with rectangles, and displaying various facial characteristics.

## Face Detection Implementations

The project includes three different implementations for face detection:

1. **Basic Face Detector** (`face_detector.py`): Uses OpenCV and face_recognition library
2. **Fast Face Detector** (`fast_face_detector.py`): Optimized implementation with performance metrics
3. **YOLO Face Detector** (`yolo_face_detector.py`): High-performance implementation using YOLOv8 with GPU acceleration

## Features

### Common Features
- Real-time face detection
- Drawing bounding boxes around detected faces
- Showing face characteristics (width, height, aspect ratio)
- Counting the number of faces detected in the frame
- Performance metrics display

### Implementation-Specific Features
- **Basic**: Facial landmark detection (eyes, eyebrows, nose, mouth, chin)
- **Fast**: Optimized for CPU with performance metrics and eye detection
- **YOLO**: GPU-accelerated detection with high accuracy and detailed performance metrics

## Requirements

### Basic Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- face_recognition library

### YOLO Implementation
- Ultralytics YOLOv8 (`pip install ultralytics`)
- PyTorch (installed automatically with Ultralytics)
- CUDA toolkit (optional, for GPU acceleration)

## Installation

1. Ensure you have Python installed
2. Install the required packages based on which implementation you want to use:

   **Basic Face Detector:**
   ```
   pip install opencv-python face_recognition
   ```
   Note: The `face_recognition` library depends on `dlib`, which might require additional setup on some systems.

   **Fast Face Detector:**
   ```
   pip install opencv-python face_recognition
   ```

   **YOLO Face Detector:**
   ```
   pip install ultralytics opencv-python
   ```
   Note: For GPU acceleration, ensure you have a compatible NVIDIA GPU with CUDA installed.

## Usage

### Basic Face Detector
```
python face_detector.py
```

### Fast Face Detector (Optimized)
```
python fast_face_detector.py
```

### YOLO Face Detector (GPU-accelerated)
```
python yolo_face_detector.py
```

The webcam will activate and display the video feed with face detection. Press 'q' to quit the application.

## Controls

### Common Controls
- **q**: Quit the application

### Fast Face Detector Additional Controls
- **+**: Increase detection sensitivity
- **-**: Decrease detection sensitivity
- **n**: Change minNeighbors parameter

## Performance Comparison

| Implementation | Performance | Accuracy | Hardware Acceleration |
|----------------|-------------|----------|----------------------|
| Basic          | Slow        | Good     | None                 |
| Fast           | Medium      | Good     | Limited (OpenCV)     |
| YOLO           | Fast        | Best     | Full GPU (CUDA)      |

## Notes

- All scripts use your default webcam (index 0)
- YOLO implementation provides the best performance, especially with GPU acceleration
- The YOLO model will be downloaded automatically on first run (yolov8n-face.pt)
- For optimal performance with YOLO, use a computer with a CUDA-capable NVIDIA GPU