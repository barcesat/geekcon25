import cv2
import numpy as np
import time
import os
import argparse
from face_tracking_api import FaceTrackingAPIServer
from numpy_json_utils import convert_numpy_types

# Import the smile detector
from yolo_serial_smile_detector import YOLOSmileDetector

# --- Import printer and crop utilities ---
from crop import prepare_image_for_thermal_printer
from escpos.printer import Serial
import datetime
import threading
import shutil

def main():
    """Run the serial smile detector with API server enabled"""
    parser = argparse.ArgumentParser(description='Face Detection with API')
    parser.add_argument('--cam_number', type=int, default=0,
                        help='Camera device number (default: 0)')
    parser.add_argument('--mode', type=str, default='serial_smile_api', 
                        choices=['serial_smile_api'],
                        help='Detection mode with API server')
    parser.add_argument('--width', type=int, default=640, 
                        help='Camera width resolution')
    parser.add_argument('--height', type=int, default=480, 
                        help='Camera height resolution')
    parser.add_argument('--fps', type=int, default=30, 
                        help='Target camera FPS')
    parser.add_argument('--smile_threshold', type=float, default=0.6,
                        help='Threshold for smile detection (0.0-1.0)')
    parser.add_argument('--smile_duration', type=float, default=2.0,
                        help='Duration in seconds for smile detection to trigger signal')
    parser.add_argument('--serial_port', type=str, default='COM8',
                        help='Serial port to connect to (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)')
    parser.add_argument('--api_host', type=str, default='localhost',
                        help='Host for the API server')
    parser.add_argument('--api_port', type=int, default=8000,
                        help='Port for the API server')
    args = parser.parse_args()
    
    # Start API server
    api_server = FaceTrackingAPIServer(host=args.api_host, port=args.api_port)
    api_server.start()
    
    try:
        # Initialize the detector
        detector = YOLOSmileDetector(
            smile_threshold=args.smile_threshold,
            smile_duration=args.smile_duration,
            serial_port=args.serial_port
        )
        
        # Open the camera
        cap = cv2.VideoCapture(args.cam_number)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        
        print(f"Camera opened with resolution {args.width}x{args.height} @ {args.fps}fps")

        last_smile_state = False


        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            # Always set the original frame for smile event handling
            detector.set_last_original_frame(frame)

            faces, metrics = detector.detect_faces(frame)
            frame = detector.draw_faces(frame, faces, metrics)

            # --- Existing API and display logic ---
            # (same as before, but moved below photo logic)
            if faces:
                # Find the biggest face (should be marked with is_biggest)
                biggest_face = next((face for face in faces if face.get('is_biggest', False)), None)
                if biggest_face:
                    face_info = biggest_face.copy()
                    if 'box' in face_info:
                        x1, y1, x2, y2 = face_info['box']
                        face_info['box'] = {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2)
                        }
                    if 'landmarks' in face_info:
                        landmark_dict = {}
                        for i, (x, y) in enumerate(face_info['landmarks']):
                            landmark_dict[f'point_{i}'] = {'x': int(x), 'y': int(y)}
                        face_info['landmarks'] = landmark_dict
                else:
                    face_info = None
                face_data = {
                    "big_face_detected": biggest_face is not None,
                    "face_info": face_info,
                    "is_smiling": biggest_face.get('is_smiling', False) if biggest_face else False,
                    "smile_confidence": biggest_face.get('smile_confidence', 0.0) if biggest_face else 0.0
                }
            else:
                face_data = {
                    "big_face_detected": False,
                    "face_info": None,
                    "is_smiling": False,
                    "smile_confidence": 0.0
                }
            if face_data["big_face_detected"] and face_data["face_info"]:
                is_smiling = face_data["is_smiling"]
                smile_confidence = face_data["smile_confidence"]
                print(f"Biggest face - smile confidence: {smile_confidence:.2f}, smiling: {is_smiling}")
                if isinstance(is_smiling, np.bool_):
                    face_data["is_smiling"] = bool(is_smiling)
            face_data = convert_numpy_types(face_data)
            api_server.update_tracking_data(face_data)
            # Set the OpenCV window to fullscreen mode
            window_name = 'Emotional Damage Machine'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    finally:
        # Stop API server
        api_server.stop()

if __name__ == "__main__":
    main()