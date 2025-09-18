import cv2
import numpy as np
import time
from collections import deque
import platform
import os

# Set environment variables for performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if platform.system() == 'Windows':
    os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '1'

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
    # Initialize the webcam with DirectShow backend for better performance on Windows
    print("Starting webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Try to load CUDA version if available
    use_cuda = False
    try:
        if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'CascadeClassifier_create'):
            cuda_face_cascade = cv2.cuda.CascadeClassifier_create(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cuda_eye_cascade = cv2.cuda.CascadeClassifier_create(cv2.data.haarcascades + 'haarcascade_eye.xml')
            use_cuda = True
            print("Using CUDA acceleration for face detection!")
    except Exception as e:
        print(f"CUDA not available: {e}")
    
    # Initialize performance tracker
    perf = PerformanceTracker()
    
    # For FPS calculation
    frame_count = 0
    last_fps_time = time.time()
    fps = 0
    
    # Settings
    scaleFactor = 1.1
    minNeighbors = 5
    
    while True:
        perf.start('full_frame')
        
        # Capture frame-by-frame
        perf.start('frame_capture')
        ret, frame = cap.read()
        capture_time = perf.end('frame_capture')
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Calculate FPS
        frame_count += 1
        if time.time() - last_fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_fps_time = time.time()
        
        # Convert to grayscale for faster processing
        perf.start('preprocessing')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram to improve detection in varying lighting
        gray = cv2.equalizeHist(gray)
        preprocess_time = perf.end('preprocessing')
        
        # Detect faces
        perf.start('face_detection')
        if use_cuda:
            # Convert to CUDA GpuMat
            cuda_gray = cv2.cuda_GpuMat(gray)
            faces = cuda_face_cascade.detectMultiScale(cuda_gray, scaleFactor, minNeighbors)
            faces = faces.download()  # Download from GPU
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
        
        detection_time = perf.end('face_detection')
        
        # Process detected faces
        perf.start('face_processing')
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region for eye detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            perf.start('eye_detection')
            if use_cuda:
                cuda_roi = cv2.cuda_GpuMat(roi_gray)
                eyes = cuda_eye_cascade.detectMultiScale(cuda_roi)
                eyes = eyes.download()
            else:
                eyes = eye_cascade.detectMultiScale(roi_gray)
            eye_detection_time = perf.end('eye_detection')
            
            # Draw circles around eyes
            for (ex, ey, ew, eh) in eyes:
                center = (x + ex + ew//2, y + ey + eh//2)
                cv2.circle(frame, center, int(ew//2.5), (255, 0, 0), 2)
            
            # Calculate face characteristics
            face_width = w
            face_height = h
            aspect_ratio = face_width / face_height if face_height != 0 else 0
            
            # Display face metrics
            y_offset = y + h + 20
            cv2.putText(frame, f"Face width: {face_width}px", (x, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Face height: {face_height}px", (x, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Aspect ratio: {aspect_ratio:.2f}", (x, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Eyes detected: {len(eyes)}", (x, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        processing_time = perf.end('face_processing')
        
        # Display performance metrics
        perf.start('display')
        
        # Display the number of faces found
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display performance metrics
        metrics_y = 90
        stats = perf.get_all_stats()
        
        # Show only the most important metrics to avoid cluttering the display
        important_metrics = ['face_detection', 'eye_detection', 'preprocessing', 'full_frame']
        for name in important_metrics:
            if name in stats:
                cv2.putText(frame, f"{name}: {stats[name]:.1f}ms", 
                            (10, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                metrics_y += 25
        
        # Display acceleration mode
        mode = "CUDA" if use_cuda else "CPU"
        cv2.putText(frame, f"Mode: {mode}", (frame.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
        display_time = perf.end('display')
        
        # Display the resulting frame
        cv2.imshow('Fast Face Detection', frame)
        
        total_frame_time = perf.end('full_frame')
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            # Increase sensitivity
            scaleFactor = max(1.05, scaleFactor - 0.05)
            print(f"Scale factor: {scaleFactor}")
        elif key == ord('-'):
            # Decrease sensitivity
            scaleFactor = min(1.4, scaleFactor + 0.05)
            print(f"Scale factor: {scaleFactor}")
        elif key == ord('n'):
            # Change min neighbors
            minNeighbors = (minNeighbors % 10) + 1
            print(f"Min neighbors: {minNeighbors}")
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final performance stats
    print("\nPerformance Statistics:")
    for name, avg_time in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {avg_time:.2f}ms")
    print(f"Average FPS: {fps}")
    print("Webcam closed.")

if __name__ == "__main__":
    main()