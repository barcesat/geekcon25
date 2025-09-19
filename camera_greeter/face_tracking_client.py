import requests
import time
import json
import argparse

class FaceTrackingClient:
    """Client for the Face Tracking API"""

    def __init__(self, api_url="http://localhost:{port}/api/face-tracking", port=8000):
        api_url = api_url.format(port=port)
        self.api_url = api_url
        self.last_status = {
            "big_face_detected": False,
            "tracking_duration": 0,
            "is_smiling": False
        }
        self.callbacks = {
            "on_face_detected": [],       # Called when a big face is first detected
            "on_face_lost": [],           # Called when a big face is lost
            "on_smile_started": [],        # Called when smile is detected
            "on_smile_ended": [],          # Called when smile ends
            "on_tracking_update": []       # Called on every update
        }
    
    def register_callback(self, event_type, callback_function):
        """Register a callback function for a specific event"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback_function)
            return True
        return False
    
    def _trigger_callbacks(self, event_type, data):
        """Trigger all callbacks for a specific event"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Error in callback: {e}")
    
    def get_face_status(self):
        """Get the current face tracking status"""
        try:
            response = requests.get(self.api_url, timeout=1)
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as e:
                    print(f"Error parsing JSON response: {e}")
                    return None
            else:
                print(f"API returned status code: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Error connecting to face tracking API: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def start_monitoring(self, interval=0.1):
        """Start monitoring the face tracking API with callbacks for events"""
        print(f"Monitoring face tracking API at {self.api_url}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                status = self.get_face_status()
                if status:
                    # Check for face detection events
                    if not self.last_status.get("big_face_detected", False) and status.get("big_face_detected", False):
                        self._trigger_callbacks("on_face_detected", status)
                        print("\n🔍 Big face detected!")
                    elif self.last_status.get("big_face_detected", False) and not status.get("big_face_detected", False):
                        self._trigger_callbacks("on_face_lost", status)
                        print("\n👋 Face lost")
                    
                    # Check for smile events
                    if not self.last_status.get("is_smiling", False) and status.get("is_smiling", False):
                        self._trigger_callbacks("on_smile_started", status)
                        print("\n😊 Smile detected!")
                    elif self.last_status.get("is_smiling", False) and not status.get("is_smiling", False):
                        self._trigger_callbacks("on_smile_ended", status)
                        print("\n😐 Smile ended")
                    
                    # Trigger general update callback
                    self._trigger_callbacks("on_tracking_update", status)
                    
                    # Update last status
                    self.last_status = status
                    
                    # Print status info
                    if status["big_face_detected"]:
                        area = status['face_info'].get('area', 'N/A') if status['face_info'] else 'N/A'
                        print(f"\rTracking face for {status['tracking_duration']:.1f}s | Area: {area} | Smiling: {'✓' if status['is_smiling'] else '✗'} ({status['smile_confidence']:.2f})", end="", flush=True)
                    else:
                        print("\rNo face detected                                                             ", end="", flush=True)
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\nStopped monitoring")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Tracking Client')
    parser.add_argument('--api_url', type=str, default="http://localhost:{port}/api/face-tracking",
                        help='URL of the face tracking API (use {port} for port substitution)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port number of the face tracking API')
    args = parser.parse_args()
    client = FaceTrackingClient(args.api_url, args.port)
    
    # Register callbacks
    def on_face_detected(data):
        if data['face_info'] and 'area' in data['face_info']:
            print(f"\nNew face detected! Face area: {data['face_info']['area']} pixels")
        else:
            print("\nNew face detected!")
    
    def on_smile_started(data):
        print(f"\nSmile detected with confidence: {data['smile_confidence']:.2f}")
    
    client.register_callback("on_face_detected", on_face_detected)
    client.register_callback("on_smile_started", on_smile_started)
    
    # Start monitoring
    client.start_monitoring()