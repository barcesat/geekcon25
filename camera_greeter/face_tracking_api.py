import threading
import time
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class FaceTrackingData:
    """Stores and manages face tracking data to be shared via API"""
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "big_face_detected": False,
            "tracking_start_time": None,
            "tracking_duration": 0,
            "face_info": None,
            "is_smiling": False,
            "smile_confidence": 0.0,
            "last_update_time": time.time()
        }
        
    @staticmethod
    def _convert_numpy_types(obj):
        """Convert numpy types to standard Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: FaceTrackingData._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FaceTrackingData._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(FaceTrackingData._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def update_face_data(self, face_data: Dict[str, Any]):
        """Update the face tracking data with new information"""
        # Convert any numpy types to standard Python types
        face_data = self._convert_numpy_types(face_data)
        
        with self._lock:
            current_time = time.time()
            
            # Check if this is a new face being tracked
            if not self._data["big_face_detected"] and face_data["big_face_detected"]:
                self._data["tracking_start_time"] = current_time
            
            # Update tracking data
            self._data["big_face_detected"] = face_data["big_face_detected"]
            
            if face_data["big_face_detected"]:
                # Calculate tracking duration if we're tracking a face
                if self._data["tracking_start_time"] is not None:
                    self._data["tracking_duration"] = current_time - self._data["tracking_start_time"]
                
                # Update face info if provided
                if "face_info" in face_data and face_data["face_info"] is not None:
                    self._data["face_info"] = face_data["face_info"]
                
                # Update smile information if provided
                if "is_smiling" in face_data:
                    self._data["is_smiling"] = face_data["is_smiling"]
                if "smile_confidence" in face_data:
                    self._data["smile_confidence"] = face_data["smile_confidence"]
            else:
                # Reset tracking data when no face is detected
                self._data["tracking_duration"] = 0
                self._data["face_info"] = None
                self._data["is_smiling"] = False
                self._data["smile_confidence"] = 0.0
            
            self._data["last_update_time"] = current_time
    
    def get_data(self) -> Dict[str, Any]:
        """Get a copy of the current face tracking data"""
        with self._lock:
            # Return a copy with all numpy types converted to standard Python types
            return self._convert_numpy_types(dict(self._data))


class FaceTrackingRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for face tracking API"""
    tracking_data = None  # Will be set by the server
    
    def do_GET(self):
        """Handle GET requests to the API"""
        if self.path == '/api/face-tracking':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # CORS header for cross-origin requests
            self.end_headers()
            
            # Get current tracking data
            data = self.tracking_data.get_data() if self.tracking_data else {}
            
            # Extra safety check for bool_ types which might have been missed
            if "is_smiling" in data and isinstance(data["is_smiling"], np.bool_):
                data["is_smiling"] = bool(data["is_smiling"])
            
            if "big_face_detected" in data and isinstance(data["big_face_detected"], np.bool_):
                data["big_face_detected"] = bool(data["big_face_detected"])
            
            # Check face_info for any boolean values
            if data.get("face_info") and isinstance(data["face_info"], dict):
                for key, value in list(data["face_info"].items()):
                    if isinstance(value, np.bool_):
                        data["face_info"][key] = bool(value)
            
            # Send response with custom encoder to handle numpy types
            try:
                json_data = json.dumps(data, cls=NumpyEncoder)
                self.wfile.write(json_data.encode('utf-8'))
            except TypeError as e:
                print(f"JSON serialization error: {e}")
                # Send a simplified response as fallback
                simple_data = {
                    "big_face_detected": bool(data.get("big_face_detected", False)),
                    "tracking_duration": float(data.get("tracking_duration", 0)),
                    "is_smiling": bool(data.get("is_smiling", False)),
                    "smile_confidence": float(data.get("smile_confidence", 0)),
                    "error": f"Data serialization error: {str(e)}"
                }
                self.wfile.write(json.dumps(simple_data).encode('utf-8'))
        else:
            # Return 404 for any other paths
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Disable request logging for cleaner output"""
        return


class FaceTrackingAPIServer:
    """API server for exposing face tracking data"""
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.tracking_data = FaceTrackingData()
        self.server = None
        self.server_thread = None
        self.is_running = False
    
    def start(self):
        """Start the API server in a background thread"""
        if self.is_running:
            print("API server is already running")
            return
        
        # Set the tracking data in the request handler
        FaceTrackingRequestHandler.tracking_data = self.tracking_data
        
        # Create and start the server
        self.server = HTTPServer((self.host, self.port), FaceTrackingRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.is_running = True
        print(f"Face tracking API server started at http://{self.host}:{self.port}/api/face-tracking")
    
    def stop(self):
        """Stop the API server"""
        if self.server and self.is_running:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
            print("Face tracking API server stopped")
    
    def update_tracking_data(self, face_data):
        """Update the face tracking data"""
        self.tracking_data.update_face_data(face_data)


# Example of how to use this in another script:
if __name__ == "__main__":
    # This is just a demo of how the API server works
    api_server = FaceTrackingAPIServer(host='localhost', port=8000)
    api_server.start()
    
    try:
        # Simulate face detection updates
        for i in range(10):
            # Simulate face detection data
            face_data = {
                "big_face_detected": i > 2,  # Start detecting face after 3rd iteration
                "face_info": {
                    "width": 200,
                    "height": 250,
                    "area": 200 * 250,
                } if i > 2 else None,
                "is_smiling": i > 5,  # Start smiling after 6th iteration
                "smile_confidence": 0.7 if i > 5 else 0.0
            }
            
            # Update tracking data
            api_server.update_tracking_data(face_data)
            print(f"Updated face data: {face_data}")
            
            time.sleep(1)  # Wait 1 second between updates
    
    finally:
        api_server.stop()