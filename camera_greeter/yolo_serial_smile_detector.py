import cv2
import numpy as np
import time
import math
from collections import deque
import argparse
import copy
import serial
import serial.tools.list_ports

import datetime
import os, datetime, threading
from crop import prepare_image_for_thermal_printer
class SmileTimer:
    """Track smile duration and trigger events when threshold is reached."""
    def __init__(self, duration_threshold=5.0, cooldown_period=2.0):
        self.duration_threshold = duration_threshold  # 5 seconds threshold
        self.cooldown_period = cooldown_period  # Cooldown period to prevent rapid retriggering
        self.smile_start_time = None  # When smile was first detected
        self.last_trigger_time = None  # Last time the signal was triggered
        self.is_triggered = False  # Whether the signal is currently triggered
        self.continuous_smile = False  # Whether a smile is continuously detected
    
    def update(self, is_smiling):
        """Update the smile timer with the current smile state."""
        current_time = time.time()
        
        # Check if we're in cooldown period after triggering
        if self.last_trigger_time and (current_time - self.last_trigger_time) < self.cooldown_period:
            return False
        
        # Handle smile detection
        if is_smiling:
            # If this is the start of a smile, record the time
            if self.smile_start_time is None:
                self.smile_start_time = current_time
                self.continuous_smile = True
            
            # Check if smile duration threshold has been reached
            smile_duration = current_time - self.smile_start_time
            if smile_duration >= self.duration_threshold and not self.is_triggered:
                self.is_triggered = True
                self.last_trigger_time = current_time
                return True  # Signal to trigger action
        else:
            # Reset if smile detection is lost
            self.smile_start_time = None
            self.continuous_smile = False
            self.is_triggered = False
        
        return False  # No trigger

    def get_smile_duration(self):
        """Get the current smile duration in seconds."""
        if self.smile_start_time is None:
            return 0.0
        return time.time() - self.smile_start_time

class SerialDevice:
    """Handle communication with a serial device."""
    def __init__(self, port=None, baud_rate=9600, timeout=1.0, auto_connect=True):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.connection = None
        self.is_connected = False
        
        # Try to auto-connect if no specific port provided
        if auto_connect and not port:
            self.detect_and_connect()
        elif port:
            self.connect(port)
    
    def detect_and_connect(self):
        """Detect available serial ports and connect to the first one."""
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No serial ports detected.")
            return False
        
        # Display available ports
        print("Available serial ports:")
        for i, port in enumerate(ports):
            print(f"{i+1}. {port.device} - {port.description}")
        
        # Try to connect to the first available port
        try:
            self.connect(ports[0].device)
            return True
        except Exception as e:
            print(f"Error connecting to port {ports[0].device}: {e}")
            return False
    
    def connect(self, port=None):
        """Connect to the specified serial port."""
        if port:
            self.port = port
        
        if not self.port:
            print("No serial port specified.")
            return False
        
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the serial device."""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.is_connected = False
            print(f"Disconnected from {self.port}")
    
    def send_signal(self, signal=b'SMILE\n'):
        """Send a signal to the serial device."""
        if not self.is_connected or not self.connection or not self.connection.is_open:
            print("Not connected to any serial device.")
            return False
        
        try:
            self.connection.write(signal)
            self.connection.flush()
            print(f"Signal sent to {self.port}: {signal}")
            response = self.connection.readline().decode(errors='ignore').strip()
            print(f"Received from {self.port}: {response}")
            return True
        except Exception as e:
            print(f"Error sending signal to {self.port}: {e}")
            return False

class YOLOSmileDetector:
    def __init__(self, 
                 model_path=None, 
                 conf_threshold=0.45, 
                 iou_threshold=0.5, 
                 smile_duration=2.0, 
                 serial_port=None, 
                 smile_threshold=0.4,
                 photo_callback=None):
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
        self.smile_threshold = smile_threshold  # Higher threshold makes detection less sensitive
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
        # self.smile_threshold = 0.4  # Higher threshold makes detection less sensitive
        
        # Smile duration tracking
        self.smile_timer = SmileTimer(duration_threshold=smile_duration)
        self.serial_device = SerialDevice(port=serial_port) if serial_port else None
        
        # Performance tracking
        self.perf_tracker = PerformanceTracker()
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Photo/printing callback or handler
        self.photo_callback = photo_callback
        self._photo_counter = 0

    def handle_smile_event(self, frame):
        """
        Called when a smile event is triggered. Saves the original frame and prints it in a thread.
        """

        try:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._photo_counter += 1
            os.makedirs("photos", exist_ok=True)
            photo_filename = os.path.join("photos", f"smile_photo_{now}_{self._photo_counter}.jpg")
            # Save the original frame (no CV markings)
            import cv2
            cv2.imwrite(photo_filename, frame)
            print(f"[PHOTO] Saved smiling face photo: {photo_filename}")

            processed_path = prepare_image_for_thermal_printer(photo_filename, max_width=384)
            if processed_path:
                def print_photo_thread(image_path):
                    try:
                        from escpos.printer import Serial
                        print(f"[PRINTER] Printing {image_path} on COM9...")
                        p = Serial(devfile='COM9', baudrate=230400, bytesize=8, parity='N', stopbits=1, timeout=1.00, dsrdtr=True)
                        p._raw(bytes([29, 40, 75, 2, 0, 49, 69, 255]))
                        p.text("\nSmile!\nYou've been slapped with\nemotional damage\n")
                        p.image(image_path)
                        p.cut()
                        print("[PRINTER] Print job finished.")
                    except Exception as e:
                        print(f"[PRINTER] Error: {e}")
                threading.Thread(target=print_photo_thread, args=(processed_path,), daemon=True).start()
        except Exception as e:
            print(f"[PHOTO] Error in handle_smile_event: {e}")

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
        """Detect faces, landmarks and smiles in a frame, focusing on the largest face."""
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
        any_smiling = False
        
        # First: collect all faces and find the biggest one
        biggest_face_area = 0
        biggest_face_index = -1
        
        for i, (box, confidence) in enumerate(zip(boxes, confidences)):
            x, y, w, h = box.astype(int)
            face_area = w * h
            
            # Track the biggest face
            if face_area > biggest_face_area:
                biggest_face_area = face_area
                biggest_face_index = i
            
            # Extract facial landmarks
            kp = landmarks[i]
            face_landmarks = []
            for j in range(5):
                lm_x, lm_y = int(kp[j * 3]), int(kp[j * 3 + 1])
                face_landmarks.append((lm_x, lm_y))
            
            # Create basic face info (without smile detection for now)
            face_info = {
                'box': (x, y, x+w, y+h),  # x1, y1, x2, y2 format
                'confidence': float(confidence),
                'width': w,
                'height': h,
                'area': face_area,
                'aspect_ratio': w/h if h > 0 else 0,
                'landmarks': face_landmarks,  # List of (x,y) tuples
                'is_biggest': False,
                'is_smiling': False,
                'smile_confidence': 0.0,
                'smile_metrics': {}  # Will store smile metric details
            }
            faces.append(face_info)
        
        # Second: Process only the biggest face for smile detection
        if biggest_face_index >= 0:
            # Mark the biggest face
            faces[biggest_face_index]['is_biggest'] = True
            
            # Process smile detection only for the biggest face
            biggest_face = faces[biggest_face_index]
            is_smiling, smile_confidence, smile_metrics = self.detect_smile(biggest_face['landmarks'], frame)
            
            # Update the face with smile information
            biggest_face['is_smiling'] = is_smiling
            biggest_face['smile_confidence'] = smile_confidence
            biggest_face['smile_metrics'] = smile_metrics
            
            # Track if any face is smiling (in this case, just the biggest one)
            any_smiling = is_smiling
            
            # Print metrics for the biggest face only
            print(f"Biggest face - smile confidence: {smile_confidence:.2f}, smiling: {is_smiling}")
        
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
        
        # Update smile duration timer and check if we should send a signal
        timer_triggered = self.smile_timer.update(any_smiling)
        if timer_triggered:
            # Only trigger both handle_smile_event and send_signal together if serial_device is present and connected
            if self.serial_device and self.serial_device.is_connected:
                # Print and send signal at the same time
                if self.photo_callback is not None:
                    self.photo_callback(self._last_original_frame)
                elif hasattr(self, '_last_original_frame'):
                    self.handle_smile_event(self._last_original_frame)
                self.serial_device.send_signal()
            self.smile_timer.smile_start_time = None
            self.smile_timer.is_triggered = False
        
        smile_detection_time = self.perf_tracker.end('smile_detection')
        total_time = self.perf_tracker.end('full_process')

        # Return faces and metrics
        metrics = {
            'fps': self.fps,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'smile_detection_time': smile_detection_time,
            'total_time': total_time,
            'smile_duration': self.smile_timer.get_smile_duration()
        }
        return faces, metrics
    def set_last_original_frame(self, frame):
        """Store the last original frame (before any CV markings) for use in smile event."""
        self._last_original_frame = frame.copy()
    
    def detect_smile(self, landmarks, frame=None):
        """Detect if a face is smiling based on landmarks using multiple features."""
        # Need at least 5 landmarks (eyes, nose, mouth corners)
        if len(landmarks) < 5:
            return False, 0.0, {}
            
        # Extract landmarks (index 3 and 4 are mouth corners)
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        nose = landmarks[2]
        left_eye = landmarks[1]
        right_eye = landmarks[0]
        
        # Calculate mouth width
        mouth_width = np.sqrt((right_mouth[0] - left_mouth[0])**2 + (right_mouth[1] - left_mouth[1])**2)
        
        # Calculate face width based on eyes
        face_width = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        
        # Calculate vertical distances
        mouth_y = (left_mouth[1] + right_mouth[1]) / 2
        eye_y = (left_eye[1] + right_eye[1]) / 2
        nose_to_mouth = abs(nose[1] - mouth_y)
        eye_to_nose = abs(eye_y - nose[1])
        
        # ---------- Multiple smile indicators ----------
        
        # 1. Mouth width to face width ratio (primary indicator)
        # When smiling, mouth gets wider relative to face
        mouth_face_ratio = mouth_width / face_width if face_width > 0 else 0
        mouth_width_score = self.get_smile_confidence(mouth_face_ratio)
        
        # 2. Mouth curvature (check if mouth corners are higher than center)
        # In natural smiles, mouth corners turn upward
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        left_corner_higher = left_mouth[1] < mouth_center_y
        right_corner_higher = right_mouth[1] < mouth_center_y
        mouth_curved_up = left_corner_higher and right_corner_higher
        mouth_curve_score = 1.0 if mouth_curved_up else 0.0
        
        # 3. Cheek to eye ratio (face becomes more compact when smiling)
        # Distance from nose to mouth compared to distance from eyes to nose
        # This ratio decreases during smiling as cheeks are pushed up
        if eye_to_nose > 0:
            cheek_ratio = nose_to_mouth / eye_to_nose
            cheek_compactness = 1.0 - min(1.0, cheek_ratio / 1.5)  # Normalize: lower ratio = higher smile
        else:
            cheek_compactness = 0.0
            
        # 4. Mouth openness
        mouth_height_ratio = abs(left_mouth[1] - right_mouth[1]) / mouth_width
        mouth_openness = max(0, min(1.0, 1.0 - mouth_height_ratio * 5))  # Normalize
        
        # Store all metrics in a dictionary
        smile_metrics = {
            'mouth_width_ratio': mouth_face_ratio,
            'mouth_width_score': mouth_width_score,
            'mouth_curved_up': mouth_curved_up,
            'mouth_curve_score': mouth_curve_score,
            'cheek_compactness': cheek_compactness,
            'mouth_openness': mouth_openness
        }
        
        # Calculate compound smile score from multiple features
        # Weighted average of different smile indicators
        smile_confidence = (
            0.45 * mouth_width_score +  # Mouth width (primary)
            0.30 * mouth_curve_score +  # Mouth curvature
            0.15 * cheek_compactness +  # Cheek compactness
            0.10 * mouth_openness       # Mouth openness
        )
        
        # Update smile history for temporal smoothing
        self.smile_history.append(smile_confidence)
        avg_smile_confidence = sum(self.smile_history) / len(self.smile_history)
        
        # Determine if smiling based on threshold
        is_smiling = avg_smile_confidence > self.smile_threshold
        
        return is_smiling, avg_smile_confidence, smile_metrics
    
    def get_smile_confidence(self, mouth_face_ratio):
        """Convert mouth-face ratio to smile confidence.
        Research shows neutral expressions typically have mouth/face width ratio of ~0.4
        while smiling expressions have ratios of ~0.5-0.65
        """
        # Adjusted thresholds based on facial analysis research
        if mouth_face_ratio < 0.40:
            return 0.0  # Definitely not smiling
        elif mouth_face_ratio > 0.65:
            return 1.0  # Definitely smiling
        else:
            # Linear mapping from 0.40-0.65 to 0.0-1.0
            return (mouth_face_ratio - 0.40) / 0.25
    
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
            
            # Draw smile status with enhanced display
            smile_status = "SMILING" if is_smiling else "Not smiling"
            cv2.putText(frame, f"{smile_status}: {smile_confidence:.2f}", (x1, y2 + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Display threshold value for reference
            threshold_info = f"Threshold: {self.smile_threshold:.2f}"
            cv2.putText(frame, threshold_info, (x1, y2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Display smile progress bar
            bar_length = 100
            filled_length = int(smile_confidence * bar_length)
            bar_x = x1
            bar_y = y2 + 70
            bar_height = 10
            
            # Draw background bar (gray)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (100, 100, 100), -1)
            
            # Draw filled portion (green if smiling, yellow otherwise)
            fill_color = (0, 255, 0) if is_smiling else (0, 255, 255)
            if filled_length > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_length, bar_y + bar_height), fill_color, -1)
            
            # Draw threshold marker
            threshold_x = bar_x + int(self.smile_threshold * bar_length)
            cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), (255, 255, 255), 2)
            
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
            
            # Display smile metrics (only for the biggest face)
            y_offset = y2 + 90
            
            # Only show detailed smile metrics for the biggest face
            if face.get('is_biggest') and 'smile_metrics' in face and face['smile_metrics']:
                metrics = face['smile_metrics']
                
                # Display mouth width ratio and score
                cv2.putText(frame, f"Mouth width: {metrics['mouth_width_ratio']:.2f} ({metrics['mouth_width_score']:.2f})", 
                            (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20
                
                # Display mouth curvature
                curve_text = "Curved up" if metrics['mouth_curved_up'] else "Not curved"
                cv2.putText(frame, f"Mouth curve: {curve_text} ({metrics['mouth_curve_score']:.2f})", 
                            (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20
                
                # Display cheek compactness
                cv2.putText(frame, f"Cheek compact: {metrics['cheek_compactness']:.2f}", 
                            (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20
                
                # Display mouth openness
                cv2.putText(frame, f"Mouth open: {metrics['mouth_openness']:.2f}", 
                            (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Mark this as the target face
                cv2.putText(frame, "TARGET FACE", (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # For non-biggest faces, just indicate they're not being tracked
                if not face.get('is_biggest'):
                    cv2.putText(frame, "Not tracked", (x1, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw performance metrics on frame
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # cv2.putText(frame, f"FPS: {metrics.get('fps', 0)}", (10, 60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw smile duration progress bar if smiling
        if self.smile_timer.continuous_smile and 'smile_duration' in metrics:
            duration = metrics['smile_duration']
            threshold = self.smile_timer.duration_threshold
            progress = min(duration / threshold, 1.0)
            
            bar_width = 300
            bar_height = 30
            margin = 10
            
            # Position at bottom center of screen
            x = (frame.shape[1] - bar_width) // 2
            y = frame.shape[0] - bar_height - margin
            
            # Draw background bar
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
            
            # Draw progress bar
            progress_width = int(progress * bar_width)
            color = (0, 255, 0) if progress >= 1.0 else (0, 165, 255)  # Green if complete, orange if in progress
            cv2.rectangle(frame, (x, y), (x + progress_width, y + bar_height), color, -1)
            
            # Draw text showing seconds
            text = f"Smile: {duration:.1f}s / {threshold:.1f}s"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (bar_width - text_size[0]) // 2
            text_y = y + (bar_height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If signal triggered, show notification
            if self.smile_timer.is_triggered:
                notification = "SIGNAL SENT!"
                cv2.putText(frame, notification, (x + bar_width + 20, y + bar_height//2 + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
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


def list_serial_ports():
    """List all available serial ports."""
    ports = list(serial.tools.list_ports.comports())
    print("\nAvailable Serial Ports:")
    if not ports:
        print("  No ports available")
    else:
        for i, port in enumerate(ports):
            print(f"  {i+1}. {port.device}: {port.description}")
    print()
    return ports


def main():
    """Main function to run the face detector with smile detection."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv8 Face Detector with Smile Detection and Serial Signal')
    parser.add_argument('--model', type=str, 
                        default='yolov8-face-landmarks-opencv-dnn/weights/yolov8n-face.onnx',
                        help='Path to the ONNX model file')
    parser.add_argument('--conf', type=float, default=0.45,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--smile_threshold', type=float, default=0.5,
                        help='Threshold for smile detection (0.0-1.0)')
    parser.add_argument('--smile_duration', type=float, default=5.0,
                        help='Duration in seconds for smile detection to trigger signal')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height')
    parser.add_argument('--serial_port', type=str, default=None,
                        help='Serial port to connect to (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)')
    parser.add_argument('--baud_rate', type=int, default=9600,
                        help='Baud rate for serial communication')
    parser.add_argument('--list_ports', action='store_true',
                        help='List available serial ports and exit')
    args = parser.parse_args()
    
    # List available serial ports if requested
    if args.list_ports:
        list_serial_ports()
        return
    
    # Automatically detect serial port if not specified
    serial_port = args.serial_port
    if not serial_port:
        ports = list_serial_ports()
        if ports:
            serial_port = ports[0].device
            print(f"Auto-selecting serial port: {serial_port}")
    
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
    detector = YOLOSmileDetector(
        model_path=args.model, 
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        smile_duration=args.smile_duration,
        serial_port=args.serial_port,
        smile_threshold=args.smile_threshold
    )
    
    # Record start time
    start_time = time.time()
    
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
        
        # Display elapsed time
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Session: {int(elapsed//60):02d}:{int(elapsed%60):02d}", 
                    (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('YOLO Face Detection with Smile Recognition', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    # Clean up serial connection if active
    if hasattr(detector, 'serial_device') and detector.serial_device:
        detector.serial_device.disconnect()
    
    # Print final performance stats
    stats = detector.perf_tracker.get_all_stats()
    print("\nPerformance Statistics:")
    for name, avg_time in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {avg_time:.2f}ms")
    print(f"Average FPS: {metrics['fps']}")
    print("Webcam closed.")

if __name__ == "__main__":
    main()