"""
Face Detection Module
Handles face detection using multiple methods
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import FACE_DETECTION_SETTINGS

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using OpenCV and dlib"""

    def __init__(self):
        self.haar_cascade = None
        self.dnn_net = None
        self._load_detectors()

    def _load_detectors(self):
        """Load face detection models"""
        try:
            # Load Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

            # Load DNN face detector (more accurate)
            prototxt_path = cv2.data.haarcascades.replace('haarcascades', 'dnn') + 'deploy.prototxt'
            model_path = cv2.data.haarcascades.replace('haarcascades', 'dnn') + 'res10_300x300_ssd_iter_140000.caffemodel'

            if Path(prototxt_path).exists() and Path(model_path).exists():
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                logger.info("DNN face detector loaded")
            else:
                logger.info("DNN model not found, using Haar Cascade only")

        except Exception as e:
            logger.error(f"Error loading face detectors: {str(e)}")

    def detect_faces_haar(self, frame: np.ndarray, return_gray: bool = False) -> list:
        """Detect faces using Haar Cascade"""
        if self.haar_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SETTINGS['scale_factor'],
            minNeighbors=FACE_DETECTION_SETTINGS['min_neighbors'],
            minSize=FACE_DETECTION_SETTINGS['min_size']
        )

        if return_gray:
            return list(faces), gray
        return list(faces)

    def detect_faces_dnn(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list:
        """Detect faces using DNN (more accurate)"""
        if self.dnn_net is None:
            return self.detect_faces_haar(frame)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces

    def detect_faces(self, frame: np.ndarray, method: str = 'haar') -> list:
        """Detect faces using specified method"""
        if method == 'dnn':
            return self.detect_faces_dnn(frame)
        return self.detect_faces_haar(frame)

    def extract_face(self, frame: np.ndarray, face_rect: tuple,
                     padding: int = 20) -> np.ndarray:
        """Extract face region from frame with padding"""
        x, y, w, h = face_rect
        height, width = frame.shape[:2]

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        return frame[y1:y2, x1:x2]

    def draw_face_box(self, frame: np.ndarray, face_rect: tuple,
                      label: str = None, color: tuple = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        """Draw bounding box around detected face"""
        x, y, w, h = face_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if label:
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def preprocess_face(self, face_img: np.ndarray, target_size: tuple = (200, 200)) -> np.ndarray:
        """Preprocess face image for recognition"""
        if face_img is None or face_img.size == 0:
            return None

        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # Resize to target size
        resized = cv2.resize(gray, target_size)

        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(resized)

        return equalized


class LivenessDetector:
    """Liveness detection to prevent photo spoofing"""

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.blink_count = 0
        self.last_eye_state = True  # True = eyes open
        self.frame_history = []
        self.movement_threshold = 20

    def detect_eyes(self, face_gray: np.ndarray) -> list:
        """Detect eyes in face region"""
        eyes = self.eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        return list(eyes)

    def calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """Calculate eye aspect ratio for blink detection"""
        if eye_region is None or eye_region.size == 0:
            return 0.0

        # Simple intensity-based approach
        mean_intensity = np.mean(eye_region)
        return mean_intensity / 255.0

    def detect_blink(self, face_gray: np.ndarray, threshold: float = 0.25) -> bool:
        """Detect if a blink occurred"""
        eyes = self.detect_eyes(face_gray)

        if len(eyes) < 2:
            current_eye_state = False
        else:
            # Calculate average eye openness
            ear_values = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_region = face_gray[ey:ey + eh, ex:ex + ew]
                ear = self.calculate_eye_aspect_ratio(eye_region)
                ear_values.append(ear)

            avg_ear = np.mean(ear_values)
            current_eye_state = avg_ear > threshold

        # Detect blink transition (open -> closed -> open)
        if self.last_eye_state and not current_eye_state:
            self.blink_count += 1
            blink_detected = True
        else:
            blink_detected = False

        self.last_eye_state = current_eye_state
        return blink_detected

    def detect_movement(self, current_frame: np.ndarray) -> bool:
        """Detect face movement to verify liveness"""
        if len(self.frame_history) < 5:
            self.frame_history.append(current_frame.copy())
            return False

        # Compare with oldest frame
        old_frame = self.frame_history[0]
        diff = cv2.absdiff(current_frame, old_frame)
        movement = np.mean(diff)

        # Update history
        self.frame_history.append(current_frame.copy())
        self.frame_history.pop(0)

        return movement > self.movement_threshold

    def check_liveness(self, face_gray: np.ndarray, required_blinks: int = 2) -> tuple:
        """
        Check liveness based on blinks and movement
        Returns: (is_live, blink_count, has_movement)
        """
        blink_detected = self.detect_blink(face_gray)
        has_movement = self.detect_movement(face_gray)

        is_live = self.blink_count >= required_blinks or has_movement

        return is_live, self.blink_count, has_movement

    def reset(self):
        """Reset liveness detection state"""
        self.blink_count = 0
        self.last_eye_state = True
        self.frame_history = []
