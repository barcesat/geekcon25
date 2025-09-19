# Readme

## Smile detection
```
geekcon25\camera_greeter\.venv\Scripts\Activate.ps1

py c:\Projects\geekcon25\camera_greeter\run_face_detector.py --mode smile
```

## API Smile detection

py run_face_detector_api.py --smile_threshold 0.61 --serial_port COM8 --cam_number 1 --api_port 8000

C:\Projects\geekcon25\camera_greeter> py .\face_tracking_client.py.py --port 8000

py run_face_detector_api.py --smile_threshold 0.61 --serial_port COM8 --cam_number 0 --api_port 8001

C:\Projects\geekcon25\camera_greeter> py .\face_tracking_client.py.py --port 8001