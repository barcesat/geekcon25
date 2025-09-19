# Face Tracking API

This API service allows other Python scripts to recognize when a big face is being tracked by the camera. It provides information about face detection, tracking duration, and smile status through a simple HTTP API.

## Files

- `face_tracking_api.py` - The API server implementation
- `run_face_detector_api.py` - Modified face detector that includes the API service
- `face_tracking_client.py` - Example client to use the API

## How to Use

### Starting the API Server

Run the face detector with the API server enabled:

```bash
python run_face_detector_api.py --smile_threshold 0.6 --serial_port COM8
```

Optional arguments:
- `--api_host` - Host for the API server (default: localhost)
- `--api_port` - Port for the API server (default: 8000)
- `--smile_threshold` - Threshold for smile detection (default: 0.6)
- `--smile_duration` - Duration in seconds for smile detection (default: 5.0)
- `--serial_port` - Serial port for sending signals (default: COM4)

### API Endpoints

- `GET /api/face-tracking` - Get the current face tracking status

Example response:
```json
{
  "big_face_detected": true,
  "tracking_start_time": 1632150000.123,
  "tracking_duration": 3.5,
  "face_info": {
    "box": {
      "x1": 100,
      "y1": 150, 
      "x2": 300,
      "y2": 350
    },
    "confidence": 0.98,
    "width": 200,
    "height": 200,
    "area": 40000,
    "aspect_ratio": 1.0,
    "landmarks": {
      "point_0": {"x": 120, "y": 180},
      "point_1": {"x": 280, "y": 180},
      "point_2": {"x": 200, "y": 250},
      "point_3": {"x": 150, "y": 300},
      "point_4": {"x": 250, "y": 300}
    },
    "is_biggest": true,
    "is_smiling": true,
    "smile_confidence": 0.85,
    "smile_metrics": { "mouth_face_ratio": 0.6 }
  },
  "is_smiling": true,
  "smile_confidence": 0.85,
  "last_update_time": 1632150003.623
}
```

### Using the Client

For a simple way to use the API in your Python script:

```python
from face_tracking_client import FaceTrackingClient

# Create client
client = FaceTrackingClient("http://localhost:8000/api/face-tracking")

# Register callbacks for events
def on_face_detected(data):
    print(f"New face detected! Face area: {data['face_info']['area']} pixels")

def on_smile_started(data):
    print(f"Smile detected with confidence: {data['smile_confidence']:.2f}")

client.register_callback("on_face_detected", on_face_detected)
client.register_callback("on_smile_started", on_smile_started)

# Start monitoring
client.start_monitoring()
```

### Available Events

The client supports these events:
- `on_face_detected` - Called when a big face is first detected
- `on_face_lost` - Called when a big face is lost
- `on_smile_started` - Called when smile is detected
- `on_smile_ended` - Called when smile ends
- `on_tracking_update` - Called on every update

### Direct API Access

You can also access the API directly using HTTP requests:

```python
import requests

response = requests.get("http://localhost:8000/api/face-tracking")
face_data = response.json()

if face_data["big_face_detected"]:
    print(f"Face detected for {face_data['tracking_duration']:.1f} seconds")
    if face_data["is_smiling"]:
        print("Person is smiling!")
```

## Testing

1. Start the API server:
   ```bash
   python run_face_detector_api.py
   ```

2. In another terminal, run the example client:
   ```bash
   python face_tracking_client.py
   ```

3. Move in front of the camera and observe the client output as it detects your face and smile.