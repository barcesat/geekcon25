import cv2
import face_recognition
import numpy as np
import time
from collections import deque
import platform
import os

# Set environment variables for performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if platform.system() == 'Windows':
    os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '1'

# Try to enable GPU acceleration if available
try:
    # Check if CUDA is available for OpenCV
    cv2_build_info = cv2.getBuildInformation()
    if "CUDA" in cv2_build_info and "YES" in cv2_build_info.split("CUDA")[1].split("\n")[0]:
        print("OpenCV with CUDA support is available!")
    else:
        print("OpenCV without CUDA support.")
except:
    print("Could not check OpenCV CUDA status.")

# Performance tracker class for measuring execution times
class PerformanceTracker:
    def __init__(self, max_samples=30):
        self.timers = {}
        self.max_samples = max_samples
    
    def start(self, name):
        if name not in self.timers:
            self.timers[name] = {'start': 0, 'times': deque(maxlen=self.max_samples)}
        self.timers[name]['start'] = time.time()
    
    def end(self, name):
        if name in self.timers and self.timers[name]['start'] > 0:
            elapsed = (time.time() - self.timers[name]['start']) * 1000  # ms
            self.timers[name]['times'].append(elapsed)
            return elapsed
        return 0
    
    def get_avg(self, name):
        if name in self.timers and len(self.timers[name]['times']) > 0:
            return sum(self.timers[name]['times']) / len(self.timers[name]['times'])
        return 0
    
    def get_all_stats(self):
        stats = {}
        for name, data in self.timers.items():
            if len(data['times']) > 0:
                stats[name] = self.get_avg(name)
        return stats

def main():
    # Initialize the webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try CAP_DSHOW on Windows for better performance
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    # Initialize performance tracker
    perf = PerformanceTracker()
    frame_count = 0
    last_fps_time = time.time()
    
    # Variables for processing every Nth frame
    process_this_frame = 0
    process_every_n_frames = 2  # Process every 2nd frame for face detection
    
    # For FPS calculation
    fps = 0
    
    while True:
        perf.start('full_frame')
        perf.start('frame_capture')
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_time = perf.end('frame_capture')
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Calculate FPS
        frame_count += 1
        if time.time() - last_fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_fps_time = time.time()
        
        # Process every Nth frame for face detection (skip frames to increase speed)
        process_faces = process_this_frame == 0
        process_this_frame = (process_this_frame + 1) % process_every_n_frames
        
        face_locations = []
        face_encodings = []
        
        if process_faces:
            perf.start('resize')
            # Resize frame for faster face recognition processing
            # Make it even smaller for better performance
            small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
            resize_time = perf.end('resize')
            
            perf.start('color_convert')
            # Convert the image from BGR (OpenCV ordering) to RGB (face_recognition ordering)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            color_time = perf.end('color_convert')
            
            perf.start('face_locations')
            # Find all face locations in the current frame
            # Use HOG method for CPU, or CNN for GPU if available (but it's slower without dedicated GPU)
            face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
            location_time = perf.end('face_locations')
            
            if face_locations:
                perf.start('face_encodings')
                # Only compute encodings if faces were found
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                encoding_time = perf.end('face_encodings')
        
        # Process each face in the frame
        perf.start('face_processing')
        for i, (face_loc, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = face_loc
            # Scale back up face locations since the frame we detected in was scaled to 0.2x size
            scale_factor = 5  # 1/0.2 = 5
            top *= scale_factor
            right *= scale_factor
            bottom *= scale_factor
            left *= scale_factor
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Extract landmarks only if we have faces and not too many (for performance)
            if i == 0 and len(face_locations) < 3:  # Only process landmarks for the first face if < 3 faces
                perf.start('landmarks')
                face_landmark_dict = face_recognition.face_landmarks(rgb_small_frame, [face_loc])
                landmarks_time = perf.end('landmarks')
                
                # Get facial characteristics
                if face_landmark_dict:
                    landmarks = face_landmark_dict[0]
                    # Display characteristics count
                    characteristics = {
                        "Eyes": len(landmarks.get('left_eye', [])) + len(landmarks.get('right_eye', [])),
                        "Eyebrows": len(landmarks.get('left_eyebrow', [])) + len(landmarks.get('right_eyebrow', [])),
                        "Nose": len(landmarks.get('nose_bridge', [])) + len(landmarks.get('nose_tip', [])),
                        "Mouth": len(landmarks.get('top_lip', [])) + len(landmarks.get('bottom_lip', [])),
                        "Chin": len(landmarks.get('chin', []))
                    }
                    
                    # Draw selected landmarks instead of all for better performance
                    landmark_step = 2  # Draw every 2nd landmark point
                    for feature_name, points in landmarks.items():
                        for j, point in enumerate(points):
                            if j % landmark_step == 0:  # Only draw every Nth point
                                # Scale point back to original size
                                px, py = point
                                px *= scale_factor
                                py *= scale_factor
                                cv2.circle(frame, (px, py), 2, (255, 0, 0), -1)
                
                    # Display characteristic information below the face
                    y_offset = bottom + 20
                    for feature, count in characteristics.items():
                        text = f"{feature}: {count} points"
                        cv2.putText(frame, text, (left, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20
                    
                    # Calculate and display additional metrics
                    # Face width and height
                    face_width = right - left
                    face_height = bottom - top
                    aspect_ratio = face_width / face_height if face_height != 0 else 0
                    
                    cv2.putText(frame, f"Face width: {face_width}px", (left, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20
                    cv2.putText(frame, f"Face height: {face_height}px", (left, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20
                    cv2.putText(frame, f"Aspect ratio: {aspect_ratio:.2f}", (left, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
        face_processing_time = perf.end('face_processing')
        
        # Display performance metrics on the frame
        perf.start('display')
        
        # Display the number of faces found
        cv2.putText(frame, f"Faces detected: {len(face_locations)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display performance metrics
        metrics_y = 90
        stats = perf.get_all_stats()
        
        # Sort metrics by time consumption (most expensive first)
        sorted_metrics = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        
        for name, avg_time in sorted_metrics:
            cv2.putText(frame, f"{name}: {avg_time:.1f}ms", 
                        (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            metrics_y += 25
            
        # Display processing mode
        status = "Processing" if process_faces else "Displaying"
        cv2.putText(frame, status, (frame.shape[1] - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
        display_time = perf.end('display')
        
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)
        
        total_frame_time = perf.end('full_frame')
        
        # Key handling - 'q' to quit, 's' to toggle processing speed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Toggle processing frequency
            process_every_n_frames = 4 if process_every_n_frames == 2 else 2
            print(f"Processing every {process_every_n_frames} frames")
        elif key == ord('d'):
            # Toggle detailed mode (more or less landmarks)
            landmark_step = 1 if landmark_step == 2 else 2
            print(f"Landmark detail level: {landmark_step}")
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final performance stats
    print("\nPerformance Statistics:")
    stats = perf.get_all_stats()
    for name, avg_time in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {avg_time:.2f}ms")
    print(f"Average FPS: {fps}")
    print("Webcam closed.")

# Add this function to attempt CUDA face detection if available
def try_cuda_face_detection():
    # This is experimental and may not work on all systems
    try:
        # Try to load face detection model with CUDA
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'CascadeClassifier_create'):
            # Create CUDA cascade classifier
            cuda_cascade = cv2.cuda.CascadeClassifier_create(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("CUDA face detection available!")
            return True
        else:
            print("CUDA face detection NOT available.")
            return False
    except Exception as e:
        print(f"Error initializing CUDA face detection: {e}")
        return False

if __name__ == "__main__":
    # Try CUDA face detection (optional)
    try_cuda_face_detection()
    
    # Run the main application
    main()