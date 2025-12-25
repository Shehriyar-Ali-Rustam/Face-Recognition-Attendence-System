"""
Face Recognition Module
Handles face encoding and recognition using face_recognition library and LBPH
"""

import cv2
import numpy as np
import face_recognition
import pickle
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    FACE_RECOGNITION_SETTINGS, TRAINED_MODELS_DIR, DATASET_DIR,
    ATTENDANCE_SETTINGS
)

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognition using face_recognition library (dlib-based)"""

    def __init__(self):
        self.known_encodings = []
        self.known_ids = []
        self.known_names = []
        self.model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"
        self.load_model()

    def get_face_encoding(self, image: np.ndarray,
                          known_locations: list = None) -> Optional[np.ndarray]:
        """Get face encoding from an image"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Get face locations if not provided
            if known_locations is None:
                face_locations = face_recognition.face_locations(
                    rgb_image, model=FACE_RECOGNITION_SETTINGS['model']
                )
            else:
                face_locations = known_locations

            if not face_locations:
                return None

            # Get face encodings
            encodings = face_recognition.face_encodings(
                rgb_image, face_locations,
                num_jitters=FACE_RECOGNITION_SETTINGS['num_jitters'],
                model=FACE_RECOGNITION_SETTINGS['encoding_model']
            )

            return encodings[0] if encodings else None

        except Exception as e:
            logger.error(f"Error getting face encoding: {str(e)}")
            return None

    def get_all_face_encodings(self, image: np.ndarray) -> Tuple[list, list]:
        """Get all face encodings and locations from an image"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(
                rgb_image, model=FACE_RECOGNITION_SETTINGS['model']
            )

            if not face_locations:
                return [], []

            encodings = face_recognition.face_encodings(
                rgb_image, face_locations,
                num_jitters=FACE_RECOGNITION_SETTINGS['num_jitters']
            )

            # Convert locations to (x, y, w, h) format
            face_rects = []
            for (top, right, bottom, left) in face_locations:
                face_rects.append((left, top, right - left, bottom - top))

            return encodings, face_rects

        except Exception as e:
            logger.error(f"Error getting all face encodings: {str(e)}")
            return [], []

    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, str, float]:
        """
        Recognize a face from its encoding
        Returns: (student_id, name, confidence)
        """
        if not self.known_encodings:
            return "Unknown", "Unknown", 0.0

        try:
            # Calculate distances
            distances = face_recognition.face_distance(
                self.known_encodings, face_encoding
            )

            if len(distances) == 0:
                return "Unknown", "Unknown", 0.0

            # Get best match
            min_distance = np.min(distances)
            best_match_idx = np.argmin(distances)

            # Convert distance to confidence (0-1 scale)
            confidence = 1 - min_distance

            # Check if match is good enough
            if min_distance <= FACE_RECOGNITION_SETTINGS['tolerance']:
                return (
                    self.known_ids[best_match_idx],
                    self.known_names[best_match_idx],
                    confidence
                )

            return "Unknown", "Unknown", confidence

        except Exception as e:
            logger.error(f"Error recognizing face: {str(e)}")
            return "Unknown", "Unknown", 0.0

    def train_model(self, student_data: List[dict]) -> Tuple[bool, str]:
        """
        Train the recognition model with student data
        student_data: List of {'student_id': str, 'name': str, 'images_path': Path}
        """
        try:
            self.known_encodings = []
            self.known_ids = []
            self.known_names = []
            total_images = 0

            for student in student_data:
                student_id = student['student_id']
                name = student['name']
                images_path = Path(student['images_path'])

                if not images_path.exists():
                    logger.warning(f"No images found for {student_id}")
                    continue

                # Process all images for this student
                student_encodings = []
                for img_path in images_path.glob('*.jpg'):
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    encoding = self.get_face_encoding(image)
                    if encoding is not None:
                        student_encodings.append(encoding)
                        total_images += 1

                # Average the encodings for this student
                if student_encodings:
                    avg_encoding = np.mean(student_encodings, axis=0)
                    self.known_encodings.append(avg_encoding)
                    self.known_ids.append(student_id)
                    self.known_names.append(name)
                    logger.info(f"Processed {len(student_encodings)} images for {name}")

            # Save the model
            self.save_model()

            msg = f"Training complete: {len(self.known_ids)} students, {total_images} images"
            logger.info(msg)
            return True, msg

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False, f"Training failed: {str(e)}"

    def save_model(self):
        """Save trained model to file"""
        try:
            TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                'encodings': [enc.tolist() for enc in self.known_encodings],
                'ids': self.known_ids,
                'names': self.known_names
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self):
        """Load trained model from file"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                self.known_encodings = [np.array(enc) for enc in data['encodings']]
                self.known_ids = data['ids']
                self.known_names = data['names']
                logger.info(f"Model loaded: {len(self.known_ids)} students")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")


class LBPHRecognizer:
    """LBPH Face Recognizer for lightweight offline recognition"""

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100
        )
        self.label_map = {}  # Maps numeric labels to student IDs
        self.name_map = {}   # Maps numeric labels to names
        self.model_path = TRAINED_MODELS_DIR / "lbph_model.yml"
        self.label_path = TRAINED_MODELS_DIR / "label_map.json"
        self.load_model()

    def train_model(self, student_data: List[dict]) -> Tuple[bool, str]:
        """Train LBPH model with student images"""
        try:
            faces = []
            labels = []
            label_counter = 0
            total_images = 0

            self.label_map = {}
            self.name_map = {}

            for student in student_data:
                student_id = student['student_id']
                name = student['name']
                images_path = Path(student['images_path'])

                if not images_path.exists():
                    continue

                self.label_map[label_counter] = student_id
                self.name_map[label_counter] = name

                for img_path in images_path.glob('*.jpg'):
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    # Resize for consistency
                    image = cv2.resize(image, (200, 200))
                    faces.append(image)
                    labels.append(label_counter)
                    total_images += 1

                label_counter += 1

            if not faces:
                return False, "No valid images found for training"

            # Train the model
            self.recognizer.train(faces, np.array(labels))
            self.save_model()

            msg = f"LBPH Training complete: {label_counter} students, {total_images} images"
            logger.info(msg)
            return True, msg

        except Exception as e:
            logger.error(f"Error training LBPH: {str(e)}")
            return False, f"Training failed: {str(e)}"

    def recognize_face(self, face_gray: np.ndarray) -> Tuple[str, str, float]:
        """Recognize a face using LBPH"""
        try:
            if not self.label_map:
                return "Unknown", "Unknown", 0.0

            # Resize to expected size
            face_resized = cv2.resize(face_gray, (200, 200))

            # Predict
            label, confidence = self.recognizer.predict(face_resized)

            # Convert LBPH distance to confidence (lower is better)
            # LBPH returns distance, not confidence
            normalized_confidence = max(0, 1 - (confidence / 200))

            if confidence < 100:  # Good match
                student_id = self.label_map.get(label, "Unknown")
                name = self.name_map.get(label, "Unknown")
                return student_id, name, normalized_confidence

            return "Unknown", "Unknown", normalized_confidence

        except Exception as e:
            logger.error(f"Error in LBPH recognition: {str(e)}")
            return "Unknown", "Unknown", 0.0

    def save_model(self):
        """Save LBPH model and label maps"""
        try:
            TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.recognizer.save(str(self.model_path))

            label_data = {
                'label_map': {str(k): v for k, v in self.label_map.items()},
                'name_map': {str(k): v for k, v in self.name_map.items()}
            }
            with open(self.label_path, 'w') as f:
                json.dump(label_data, f)

            logger.info("LBPH model saved")
        except Exception as e:
            logger.error(f"Error saving LBPH model: {str(e)}")

    def load_model(self):
        """Load LBPH model and label maps"""
        try:
            if self.model_path.exists() and self.label_path.exists():
                self.recognizer.read(str(self.model_path))

                with open(self.label_path, 'r') as f:
                    label_data = json.load(f)

                self.label_map = {int(k): v for k, v in label_data['label_map'].items()}
                self.name_map = {int(k): v for k, v in label_data['name_map'].items()}
                logger.info(f"LBPH model loaded: {len(self.label_map)} students")
        except Exception as e:
            logger.error(f"Error loading LBPH model: {str(e)}")


class HybridRecognizer:
    """Combines multiple recognition methods for better accuracy"""

    def __init__(self, use_lbph: bool = True, use_dlib: bool = True):
        self.lbph = LBPHRecognizer() if use_lbph else None
        self.dlib = FaceRecognizer() if use_dlib else None

    def recognize_face(self, image: np.ndarray,
                       face_gray: np.ndarray = None) -> Tuple[str, str, float]:
        """Recognize using multiple methods and combine results"""
        results = []

        # Dlib-based recognition
        if self.dlib:
            encoding = self.dlib.get_face_encoding(image)
            if encoding is not None:
                student_id, name, conf = self.dlib.recognize_face(encoding)
                if student_id != "Unknown":
                    results.append((student_id, name, conf, 'dlib'))

        # LBPH recognition
        if self.lbph and face_gray is not None:
            student_id, name, conf = self.lbph.recognize_face(face_gray)
            if student_id != "Unknown":
                results.append((student_id, name, conf, 'lbph'))

        if not results:
            return "Unknown", "Unknown", 0.0

        # Use dlib result if available (more accurate)
        for r in results:
            if r[3] == 'dlib':
                return r[0], r[1], r[2]

        # Otherwise use best confidence
        best = max(results, key=lambda x: x[2])
        return best[0], best[1], best[2]

    def train_all(self, student_data: List[dict]) -> Tuple[bool, str]:
        """Train all recognizers"""
        messages = []

        if self.dlib:
            success, msg = self.dlib.train_model(student_data)
            messages.append(f"Dlib: {msg}")

        if self.lbph:
            success, msg = self.lbph.train_model(student_data)
            messages.append(f"LBPH: {msg}")

        return True, " | ".join(messages)
