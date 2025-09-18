import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import torch
import os

class YOLOFaceDetector:
    def __init__(self, model_path=None, conf_threshold=0.3):  # Lowered threshold from 0.5 to 0.3
        """Initialize the YOLO face detector"""
        print("Initializing YOLOv8 face detector...")
        
        # Set CUDA device if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        if self.device == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load YOLO model - either use pretrained or specific model
        if model_path:
            self.model = YOLO(model_path)
            print(f"Loaded custom model: {model_path}")
        else:
            # Use YOLOv8n by default, which is lightweight and fast
            self.model = YOLO('yolov8n.pt')
            print("Loaded default YOLOv8n model for face detection")
        
        # Model will run on GPU if available
        self.model.to(self.device)
        
        # Configuration
        self.conf_threshold = conf_threshold
        
        # Simple face tracking
        self.prev_faces = []
        self.face_history = deque(maxlen=5)  # Track last 5 detections for stability
        
        # Performance tracking
        self.perf_tracker = PerformanceTracker()
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def detect_faces(self, frame):
        """Detect faces in a frame using YOLOv8"""
        self.perf_tracker.start('full_process')
        
        # Track FPS
        self.frame_count += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = time.time()
        
        # Run inference
        self.perf_tracker.start('inference')
        results = self.model(frame, conf=self.conf_threshold, verbose=False, 
                            device=self.device, imgsz=320,   # Reduced image size for speed
                            classes=0)  # Only detect class 0 (person)
        inference_time = self.perf_tracker.end('inference')
        
        # Process results
        self.perf_tracker.start('process_results')
        faces = []
        
        for result in results:
            boxes = result.boxes
            
            # Extract face information
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get keypoints (landmarks) if available
                landmarks = []
                if hasattr(box, 'keypoints') and box.keypoints is not None:
                    try:
                        kpts = box.keypoints.xy[0].cpu().numpy()
                        for kp in kpts:
                            landmarks.append((int(kp[0]), int(kp[1])))
                    except (AttributeError, IndexError):
                        pass
                
                # Get face dimensions
                face_width = x2 - x1
                face_height = y2 - y1
                aspect_ratio = face_width / face_height if face_height > 0 else 0
                
                # Show all person detections (class 0) without additional filtering
                if class_id == 0:
                    
                    # Store face info
                    face_info = {
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'width': face_width,
                        'height': face_height,
                        'aspect_ratio': aspect_ratio,
                        'landmarks': landmarks
                    }
                    
                    faces.append(face_info)
        
        # Apply simple tracking for stability
        if len(faces) > 0:
            self.face_history.append(faces)
        elif len(self.face_history) > 0:
            # If no faces detected but we had some recently, use the last known faces
            # with slightly reduced confidence
            last_faces = self.face_history[-1]
            for face in last_faces:
                face['confidence'] *= 0.9  # Reduce confidence for historical faces
            faces = last_faces
        
        process_time = self.perf_tracker.end('process_results')
        total_time = self.perf_tracker.end('full_process')
        
        # Return faces and performance metrics
        return faces, {
            'fps': self.fps,
            'inference_time': inference_time,
            'process_time': process_time,
            'total_time': total_time
        }
    
    def draw_faces(self, frame, faces, metrics):
        """Draw face boxes and information on the frame"""
        # Start timing the drawing process
        self.perf_tracker.start('drawing')
        
        # Draw each detected face
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['box']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label with confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks if available
            if 'landmarks' in face and face['landmarks']:
                landmark_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                landmark_names = ['Right Eye', 'Left Eye', 'Nose', 'Right Mouth', 'Left Mouth']
                
                for j, landmark in enumerate(face['landmarks']):
                    color = landmark_colors[j % len(landmark_colors)]
                    cv2.circle(frame, landmark, 4, color, -1)
                    
                    # Optional: Draw landmark names
                    if j < len(landmark_names):
                        cv2.putText(frame, landmark_names[j], (landmark[0] + 5, landmark[1] - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Display face characteristics
            y_offset = y2 + 20
            
            # Display face width and height
            cv2.putText(frame, f"Width: {face['width']}px", (x1, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Height: {face['height']}px", (x1, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Aspect: {face['aspect_ratio']:.2f}", (x1, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw performance metrics on frame
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, f"FPS: {metrics['fps']}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Inference: {metrics['inference_time']:.1f}ms", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Process: {metrics['total_time']:.1f}ms", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Show hardware info
        device_info = "GPU" if self.device == 'cuda' else "CPU"
        cv2.putText(frame, f"Device: {device_info}", (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        self.perf_tracker.end('drawing')
        return frame

class PerformanceTracker:
    def __init__(self, max_samples=30):
        """Track performance metrics with rolling average"""
        self.timers = {}
        self.max_samples = max_samples
    
    def start(self, name):
        """Start timing an operation"""
        if name not in self.timers:
            self.timers[name] = {'start': 0, 'times': deque(maxlen=self.max_samples)}
        self.timers[name]['start'] = time.time()
    
    def end(self, name):
        """End timing and return elapsed time in milliseconds"""
        if name in self.timers and self.timers[name]['start'] > 0:
            elapsed = (time.time() - self.timers[name]['start']) * 1000  # ms
            self.timers[name]['times'].append(elapsed)
            return elapsed
        return 0
    
    def get_avg(self, name):
        """Get average time for an operation"""
        if name in self.timers and len(self.timers[name]['times']) > 0:
            return sum(self.timers[name]['times']) / len(self.timers[name]['times'])
        return 0
    
    def get_all_stats(self):
        """Get stats for all operations"""
        stats = {}
        for name, data in self.timers.items():
            if len(data['times']) > 0:
                stats[name] = self.get_avg(name)
        return stats

def download_model():
    """Download the YOLO model if not available"""
    try:
        if not os.path.exists('yolov8n.pt'):
            print("Downloading YOLOv8n model...")
            # Try to load it, which will trigger a download if not present
            YOLO('yolov8n.pt')
            print("Model downloaded successfully!")
        else:
            print("YOLOv8n model already exists")
    except Exception as e:
        print(f"Error downloading model: {e}")

def main():
    """Main function to run the face detector"""
    # Download model if needed
    download_model()
    
    # Initialize the webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW if cv2.__version__ >= '4.0.0' else 0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Set buffer size to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    # Initialize the face detector
    detector = YOLOFaceDetector()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect faces
        faces, metrics = detector.detect_faces(frame)
        
        # Draw faces on the frame
        frame = detector.draw_faces(frame, faces, metrics)
        
        # Display the resulting frame
        cv2.imshow('YOLO Face Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final performance stats
    stats = detector.perf_tracker.get_all_stats()
    print("\nPerformance Statistics:")
    for name, avg_time in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {avg_time:.2f}ms")
    print(f"Average FPS: {metrics['fps']}")
    print("Webcam closed.")

if __name__ == "__main__":
    main()