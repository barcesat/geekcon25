import requests
import time

def main():
    """Simple demo showing how to use the Face Tracking API from another project"""
    api_url = "http://localhost:8000/api/face-tracking"
    
    print("Face Tracking API Client Demo")
    print("============================")
    print(f"Connecting to API at: {api_url}")
    print("This demo will monitor for face detection and smile events.")
    print("Press Ctrl+C to exit")
    print()
    
    last_state = {
        "face_detected": False,
        "is_smiling": False
    }
    
    try:
        while True:
            try:
                # Fetch the current status from the API
                response = requests.get(api_url, timeout=1)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for state changes
                    if not last_state["face_detected"] and data.get("big_face_detected", False):
                        # Face newly detected
                        print("\n[EVENT] Face detected!")
                        
                        # Get face details
                        face_info = data.get("face_info", {})
                        if face_info:
                            width = face_info.get("width", "unknown")
                            height = face_info.get("height", "unknown")
                            area = face_info.get("area", "unknown")
                            print(f"Face dimensions: {width}x{height} pixels (area: {area})")
                    
                    elif last_state["face_detected"] and not data.get("big_face_detected", False):
                        # Face lost
                        print("\n[EVENT] Face lost")
                    
                    # Check for smile events
                    if not last_state["is_smiling"] and data.get("is_smiling", False):
                        # Started smiling
                        print(f"\n[EVENT] Started smiling! Confidence: {data.get('smile_confidence', 0):.2f}")
                    
                    elif last_state["is_smiling"] and not data.get("is_smiling", False):
                        # Stopped smiling
                        print("\n[EVENT] Stopped smiling")
                    
                    # Update the display
                    if data.get("big_face_detected", False):
                        duration = data.get("tracking_duration", 0)
                        is_smiling = "Yes" if data.get("is_smiling", False) else "No"
                        confidence = data.get("smile_confidence", 0)
                        print(f"\rTracking face for {duration:.1f}s | Smiling: {is_smiling} ({confidence:.2f})   ", end="", flush=True)
                    else:
                        print("\rWaiting for face...                                       ", end="", flush=True)
                    
                    # Update last state
                    last_state["face_detected"] = data.get("big_face_detected", False)
                    last_state["is_smiling"] = data.get("is_smiling", False)
                
                else:
                    print(f"\rAPI error: Status code {response.status_code}                ", end="", flush=True)
            
            except requests.exceptions.RequestException:
                print("\rCannot connect to API. Is the server running?        ", end="", flush=True)
            except Exception as e:
                print(f"\rError: {e}                                          ", end="", flush=True)
            
            # Wait before checking again
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nExiting demo")

if __name__ == "__main__":
    main()