import cv2
import numpy as np
import time
import math
from collections import deque
import os

class YOLOFaceDetector:
    def __init__(self, model_path=None, conf_threshold=0.45, iou_threshold=0.5, 
                 input_size=320):  # Reduced from 640 to 320 for better performance
        """Initialize the YOLO face detector with face landmarks"""
        print("Initializing YOLOv8 Face Detector with landmarks...")
        
        # Set model path
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'yolov8-face-landmarks-opencv-dnn', 'weights', 'yolov8n-face.onnx')
        
        print(f"Using model: {model_path}")
        
        # Initialize OpenCV DNN
        self.net = cv2.dnn.readNet(model_path)
        
        # Check if CUDA is available and set preferred backend and target
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("CUDA is available! Using GPU acceleration")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.device = "GPU"
                # Verify CUDA is actually working
                print(f"CUDA Device: {cv2.cuda.getDevice()}")
                print(f"CUDA Capability: {cv2.cuda.deviceGetAttribute(cv2.cuda.DeviceAttr_ComputeCapabilityMajor, 0)}.{cv2.cuda.deviceGetAttribute(cv2.cuda.DeviceAttr_ComputeCapabilityMinor, 0)}")
            else:
                print("CUDA is not available. Using CPU")
                self.device = "CPU"
        except Exception as e:
            print(f"Error setting up CUDA: {e}")
            print("Falling back to CPU")
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
        """Detect faces and landmarks in a frame."""
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
        
        # Convert boxes and landmarks to more manageable format
        faces = []
        for i, (box, confidence) in enumerate(zip(boxes, confidences)):
            x, y, w, h = box.astype(int)
            
            # Extract facial landmarks
            kp = landmarks[i]
            face_landmarks = []
            for j in range(5):
                lm_x, lm_y = int(kp[j * 3]), int(kp[j * 3 + 1])
                face_landmarks.append((lm_x, lm_y))
            
            face_info = {
                'box': (x, y, x+w, y+h),  # x1, y1, x2, y2 format
                'confidence': float(confidence),
                'width': w,
                'height': h,
                'aspect_ratio': w/h if h > 0 else 0,
                'landmarks': face_landmarks  # List of (x,y) tuples
            }
            faces.append(face_info)
        
        total_time = self.perf_tracker.end('full_process')
        
        # Return faces and metrics
        metrics = {
            'fps': self.fps,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'total_time': total_time
        }
        
        return faces, metrics

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
        """Draw face boxes and landmarks on the frame."""
        self.perf_tracker.start('drawing')
        
        # Draw each detected face
        for face in faces:
            x1, y1, x2, y2 = face['box']
            confidence = face['confidence']
            landmarks = face['landmarks']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw landmarks (eyes, nose, mouth corners)
            landmark_names = ['Right Eye', 'Left Eye', 'Nose', 'Right Mouth', 'Left Mouth']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            
            for i, landmark in enumerate(landmarks):
                cv2.circle(frame, landmark, 4, colors[i], -1)
                
                # Optional: display landmark names
                # cv2.putText(frame, landmark_names[i], 
                #            (landmark[0]+5, landmark[1]-5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
            
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
    """Main function to run the face detector."""
    # Initialize the webcam with preferred settings
    print("Starting webcam...")
    
    # Try to use DirectShow backend on Windows for better performance
    if os.name == 'nt':  # Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
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
        cv2.imshow('YOLO Face Detection with Landmarks', frame)
        
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