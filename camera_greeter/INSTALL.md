# Camera Greeter Installation Guide

This guide explains how to set up the Camera Greeter project on your system.

## Prerequisites

- Python 3.9 or higher
- Webcam
- VLC media player (for audio playback)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/barcesat/geekcon25.git
   cd geekcon25/camera_greeter
   ```

2. Create and activate a virtual environment:
   
   **Windows:**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **Linux/macOS:**
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 face model (if not already included):
   The model should be located at `yolov8-face-landmarks-opencv-dnn/weights/yolov8n-face.onnx`.
   If it's missing, you can download it from the official repository.

## Running the Application

### Starting the Face Detection API Server

```
python run_face_detector_api.py --smile_threshold 0.61 --serial_port COM4 --cam_number 0 --api_port 8000
```

Options:
- `--cam_number`: Camera device number (default: 0)
- `--smile_threshold`: Threshold for smile detection (0.0-1.0)
- `--smile_duration`: Duration in seconds for smile detection to trigger signal (default: 5.0)
- `--serial_port`: Serial port for external devices (e.g., COM4, /dev/ttyUSB0)
- `--api_host`: Host for the API server (default: localhost)
- `--api_port`: Port for the API server (default: 8000)

### Running the Face Tracking Client

```
python face_tracking_client.py --port 8000
```

Options:
- `--port`: Port number to connect to (default: 8000)

### Demo Client

For a simpler demonstration:

```
python face_tracking_client_demo.py
```

## Troubleshooting

1. **Missing dependencies**: If you get import errors, ensure you've installed all dependencies with `pip install -r requirements.txt`

2. **Camera access**: Make sure your webcam is connected and not in use by another application

3. **VLC Installation**: Ensure VLC media player is installed on your system for audio playback

4. **CUDA issues**: If using GPU acceleration, ensure CUDA and appropriate PyTorch versions are installed

## Using Multiple Cameras

You can run multiple instances of the face detector API with different camera numbers and ports:

```
# First camera on port 8000
python run_face_detector_api.py --cam_number 0 --api_port 8000

# Second camera on port 8001
python run_face_detector_api.py --cam_number 1 --api_port 8001
```

Then connect clients to each API instance:

```
python face_tracking_client.py --port 8000
python face_tracking_client.py --port 8001
```