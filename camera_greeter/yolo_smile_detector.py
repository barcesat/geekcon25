import cv2
import numpy as np
import time
import math
from collections import deque
import os
import argparse
import copy

class YOLOSmileDetector:
    def __init__(self, model_path=None, conf_threshold=0.45, iou_threshold=0.5):
        """Initialize the YOLO face detector with landmarks and smile detection"""
        print("Initializing YOLOv8 Face Detector with smile detection...")
        
        # Set model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'yolov8-face-landmarks-opencv-dnn', 'weights', 'yolov8n-face.onnx')
        
        print(f"Using model: {model_path}")
        
        # Initialize OpenCV DNN
        self.net = cv2.dnn.readNet(model_path)
        
        # Check if CUDA is available and set preferred backend and target
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA is available! Using GPU acceleration")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.device = "GPU"
        else:
            print("CUDA is not available. Using CPU")
            self.device = "CPU"
        
        # Model parameters
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16
        self.class_names = ['face']
        
        # Initialize anchors for detection
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), 
                           math.ceil(self.input_width / self.strides[i])) 
                           for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)
        
        # Simple face tracking
        self.prev_faces = []
        self.face_history = deque(maxlen=5)  # Track last 5 detections for stability
        self.smile_history = deque(maxlen=10)  # Track smile detections for stability
        
        # Smile detection parameters
        self.smile_threshold = 0.7  # Higher threshold makes detection less sensitive
        
        # Performance tracking
        self.perf_tracker = PerformanceTracker()
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        """Apply softmax function."""
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def resize_image(self, srcimg, keep_ratio=True):
        """Resize image with padding to maintain aspect ratio if needed."""
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect_faces(self, frame):
        """Detect faces, landmarks and smiles in a frame."""
        self.perf_tracker.start('full_process')
        
        # Track FPS
        self.frame_count += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = time.time()
        
        # Preprocess image
        self.perf_tracker.start('preprocess')
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = frame.shape[0]/newh, frame.shape[1]/neww
        input_img = input_img.astype(np.float32) / 255.0
        preprocess_time = self.perf_tracker.end('preprocess')
        
        # Run inference
        self.perf_tracker.start('inference')
        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        inference_time = self.perf_tracker.end('inference')
        
        # Perform post-processing
        self.perf_tracker.start('postprocess')
        boxes, confidences, class_ids, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        postprocess_time = self.perf_tracker.end('postprocess')
        
        # Smile detection
        self.perf_tracker.start('smile_detection')
        faces = []
        for i, (box, confidence) in enumerate(zip(boxes, confidences)):
            x, y, w, h = box.astype(int)
            
            # Extract facial landmarks
            kp = landmarks[i]
            face_landmarks = []
            for j in range(5):
                lm_x, lm_y = int(kp[j * 3]), int(kp[j * 3 + 1])
                face_landmarks.append((lm_x, lm_y))
            
            # Detect smile based on landmarks
            is_smiling, smile_confidence = self.detect_smile(face_landmarks, frame)
            
            face_info = {
                'box': (x, y, x+w, y+h),  # x1, y1, x2, y2 format
                'confidence': float(confidence),
                'width': w,
                'height': h,
                'aspect_ratio': w/h if h > 0 else 0,
                'landmarks': face_landmarks,  # List of (x,y) tuples
                'is_smiling': is_smiling,
                'smile_confidence': smile_confidence
            }
            faces.append(face_info)
        
        # Apply simple tracking for stability
        if len(faces) > 0:
            self.face_history.append(faces)
        elif len(self.face_history) > 0:
            # If no faces detected but we had some recently, use the last known faces
            # with slightly reduced confidence
            last_faces = copy.deepcopy(self.face_history[-1])
            for face in last_faces:
                face['confidence'] *= 0.9  # Reduce confidence for historical faces
            faces = last_faces
        
        smile_detection_time = self.perf_tracker.end('smile_detection')
        total_time = self.perf_tracker.end('full_process')
        
        # Return faces and metrics
        metrics = {
            'fps': self.fps,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'smile_detection_time': smile_detection_time,
            'total_time': total_time
        }
        
        return faces, metrics
    
    def detect_smile(self, landmarks, frame=None):
        """Detect if a face is smiling based on landmarks."""
        # Need at least 5 landmarks (eyes, nose, mouth corners)
        if len(landmarks) < 5:
            return False, 0.0
            
        # Extract mouth landmarks (index 3 and 4 are mouth corners)
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        nose = landmarks[2]
        left_eye = landmarks[1]
        right_eye = landmarks[0]
        
        # Calculate mouth width
        mouth_width = np.sqrt((right_mouth[0] - left_mouth[0])**2 + (right_mouth[1] - left_mouth[1])**2)
        
        # Calculate face width based on eyes
        face_width = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        
        # Calculate vertical distance between mouth and nose
        mouth_y = (left_mouth[1] + right_mouth[1]) / 2
        nose_to_mouth = abs(nose[1] - mouth_y)
        
        # Calculate ratio of mouth width to face width
        mouth_face_ratio = mouth_width / face_width if face_width > 0 else 0
        
        # Check for mouth height - smiling usually means corners turn up
        # A smaller y-difference indicates a more horizontal mouth (less likely to be a smile)
        mouth_height_diff = abs(right_mouth[1] - left_mouth[1])
        mouth_height_ratio = mouth_height_diff / mouth_width if mouth_width > 0 else 1
        
        # Penalize for very horizontal mouth lines (reduce confidence if mouth is too horizontal)
        horizontal_penalty = 0.0
        if mouth_height_ratio < 0.1:  # Very horizontal mouth
            horizontal_penalty = 0.3  # Reduce confidence
        
        # A smiling face typically has:
        # 1. Wider mouth relative to face width
        # 2. Smaller vertical distance from nose to mouth
        # 3. Some curvature in the mouth (not perfectly horizontal)
        
        # Extract mouth region for potential further analysis
        if frame is not None:
            mouth_x1 = max(0, left_mouth[0] - int(mouth_width * 0.2))
            mouth_y1 = max(0, left_mouth[1] - int(nose_to_mouth * 0.5))
            mouth_x2 = min(frame.shape[1], right_mouth[0] + int(mouth_width * 0.2))
            mouth_y2 = min(frame.shape[0], max(left_mouth[1], right_mouth[1]) + int(nose_to_mouth * 0.5))
            
            # Optional: Extract mouth region for further analysis
            # mouth_region = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        
        # Calculate smile confidence based on mouth-face ratio and other factors
        base_confidence = self.get_smile_confidence(mouth_face_ratio)
        
        # Apply penalties for features that suggest not smiling
        adjusted_confidence = max(0, base_confidence - horizontal_penalty)
        
        # Check mouth position relative to eyes (smiling typically raises the cheeks)
        eye_y = (left_eye[1] + right_eye[1]) / 2
        mouth_eye_dist = mouth_y - eye_y
        expected_dist = face_width * 0.75  # Expected distance based on face size
        
        # If mouth is too low relative to eyes, reduce confidence (less likely to be a smile)
        if mouth_eye_dist > expected_dist:
            dist_ratio = expected_dist / mouth_eye_dist if mouth_eye_dist > 0 else 1
            adjusted_confidence *= dist_ratio
        
        # Update smile history for temporal smoothing
        self.smile_history.append(adjusted_confidence)
        avg_smile_confidence = sum(self.smile_history) / len(self.smile_history)
        
        # Determine if smiling based on threshold
        is_smiling = avg_smile_confidence > self.smile_threshold
        
        return is_smiling, avg_smile_confidence
    
    def get_smile_confidence(self, mouth_face_ratio):
        """Convert mouth-face ratio to smile confidence."""
        # Make thresholds even more strict to detect fewer smiles
        if mouth_face_ratio < 0.45:  # Increased threshold
            return 0.0  # Definitely not smiling
        elif mouth_face_ratio > 0.8:  # Increased threshold
            return 1.0  # Definitely smiling
        else:
            # Linear mapping from 0.45-0.8 to 0.0-1.0
            return (mouth_face_ratio - 0.45) / 0.35
    
    def post_process(self, preds, scale_h, scale_w, padh, padw):
        """Post-process network predictions to get bounding boxes, scores and landmarks."""
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15))  # x1,y1,score1, ..., x5,y5,score5

            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  # Convert to x,y,w,h
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        landmarks = landmarks[mask]
        
        # Apply Non-Maximum Suppression
        if len(bboxes_wh) > 0:
            indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), 
                                      self.conf_threshold, self.iou_threshold)
            
            # Handle different return types in different OpenCV versions
            if isinstance(indices, tuple):
                # OpenCV 4.7+ returns tuple
                indices = indices[0]
            elif isinstance(indices, np.ndarray):
                # Older versions return numpy array
                indices = indices.flatten()
            
            if len(indices) > 0:
                mlvl_bboxes = bboxes_wh[indices]
                confidences = confidences[indices]
                landmarks = landmarks[indices]
                return mlvl_bboxes, confidences, np.zeros(len(indices)), landmarks
        
        print('No faces detected')
        return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        """Convert distance predictions to bounding box coordinates."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def draw_faces(self, frame, faces, metrics):
        """Draw face boxes, landmarks and smile indicators on the frame."""
        self.perf_tracker.start('drawing')
        
        # Draw each detected face
        for face in faces:
            x1, y1, x2, y2 = face['box']
            confidence = face['confidence']
            landmarks = face['landmarks']
            is_smiling = face['is_smiling']
            smile_confidence = face['smile_confidence']
            
            # Draw bounding box with color based on smiling (green if smiling, blue if not)
            box_color = (0, 255, 0) if is_smiling else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Draw smile status
            smile_status = "SMILING" if is_smiling else "Not smiling"
            cv2.putText(frame, f"{smile_status}: {smile_confidence:.2f}", (x1, y2 + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw landmarks (eyes, nose, mouth corners)
            landmark_names = ['Right Eye', 'Left Eye', 'Nose', 'Right Mouth', 'Left Mouth']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            
            for i, landmark in enumerate(landmarks):
                cv2.circle(frame, landmark, 4, colors[i], -1)
                
                # Optional: display landmark names
                # cv2.putText(frame, landmark_names[i], 
                #            (landmark[0]+5, landmark[1]-5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
            
            # Draw mouth line connecting mouth landmarks
            cv2.line(frame, landmarks[3], landmarks[4], (0, 255, 255), 2)
            
            # Display face characteristics
            y_offset = y2 + 50
            
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
        cv2.putText(frame, f"Device: {self.device}", (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        drawing_time = self.perf_tracker.end('drawing')
        return frame


class PerformanceTracker:
    def __init__(self, max_samples=30):
        """Track performance metrics with rolling average."""
        self.timers = {}
        self.max_samples = max_samples
    
    def start(self, name):
        """Start timing an operation."""
        if name not in self.timers:
            self.timers[name] = {'start': 0, 'times': deque(maxlen=self.max_samples)}
        self.timers[name]['start'] = time.time()
    
    def end(self, name):
        """End timing and return elapsed time in milliseconds."""
        if name in self.timers and self.timers[name]['start'] > 0:
            elapsed = (time.time() - self.timers[name]['start']) * 1000  # ms
            self.timers[name]['times'].append(elapsed)
            return elapsed
        return 0
    
    def get_avg(self, name):
        """Get average time for an operation."""
        if name in self.timers and len(self.timers[name]['times']) > 0:
            return sum(self.timers[name]['times']) / len(self.timers[name]['times'])
        return 0
    
    def get_all_stats(self):
        """Get stats for all operations."""
        stats = {}
        for name, data in self.timers.items():
            if len(data['times']) > 0:
                stats[name] = self.get_avg(name)
        return stats


def main():
    """Main function to run the face detector with smile detection."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv8 Face Detector with Smile Detection')
    parser.add_argument('--model', type=str, 
                        default='yolov8-face-landmarks-opencv-dnn/weights/yolov8n-face.onnx',
                        help='Path to the ONNX model file')
    parser.add_argument('--conf', type=float, default=0.45,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--smile_threshold', type=float, default=0.7,
                        help='Threshold for smile detection (0.0-1.0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height')
    args = parser.parse_args()
    
    # Initialize the webcam with preferred settings
    print("Starting webcam...")
    
    # Try to use DirectShow backend on Windows for better performance
    if os.name == 'nt':  # Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    # Initialize the face detector with smile detection
    detector = YOLOSmileDetector(model_path=args.model, 
                                conf_threshold=args.conf,
                                iou_threshold=args.iou)
    detector.smile_threshold = args.smile_threshold
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect faces and smiles
        faces, metrics = detector.detect_faces(frame)
        
        # Draw faces on the frame
        frame = detector.draw_faces(frame, faces, metrics)
        
        # Display the resulting frame
        cv2.imshow('YOLO Face Detection with Smile Recognition', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
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