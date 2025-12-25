"""
Camera Module
Handles camera operations and frame capture
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Generator
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import CAMERA_SETTINGS

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera operations"""

    def __init__(self, camera_id: int = None):
        self.camera_id = camera_id if camera_id is not None else CAMERA_SETTINGS['default_camera']
        self.cap = None
        self.is_running = False

    def start(self) -> bool:
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['frame_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['frame_height'])
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])

            self.is_running = True
            logger.info(f"Camera {self.camera_id} started")
            return True

        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return False

    def stop(self):
        """Stop the camera"""
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            logger.info(f"Camera {self.camera_id} stopped")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera"""
        if self.cap is None or not self.is_running:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            return False, None

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        return True, frame

    def get_frame_generator(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously"""
        while self.is_running:
            ret, frame = self.read_frame()
            if ret:
                yield frame

    def change_camera(self, camera_id: int) -> bool:
        """Switch to a different camera"""
        self.stop()
        self.camera_id = camera_id
        return self.start()

    @staticmethod
    def list_available_cameras(max_cameras: int = 10) -> list:
        """List all available camera indices"""
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def get_camera_info(self) -> dict:
        """Get current camera properties"""
        if self.cap is None:
            return {}

        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'backend': self.cap.getBackendName()
        }


def frame_to_bytes(frame: np.ndarray, format: str = '.jpg') -> bytes:
    """Convert frame to bytes for streaming"""
    _, buffer = cv2.imencode(format, frame)
    return buffer.tobytes()


def add_overlay_text(frame: np.ndarray, text: str,
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 0.7,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 2) -> np.ndarray:
    """Add text overlay to frame"""
    # Add background for better visibility
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 5),
                  (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
    return frame


def add_status_bar(frame: np.ndarray, status_text: str,
                   bg_color: Tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    """Add status bar at the bottom of frame"""
    height, width = frame.shape[:2]
    bar_height = 40

    # Create status bar
    cv2.rectangle(frame, (0, height - bar_height), (width, height), bg_color, -1)
    cv2.putText(frame, status_text, (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def resize_frame(frame: np.ndarray, max_width: int = 640,
                 max_height: int = 480) -> np.ndarray:
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    aspect = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect)

    if height > max_height:
        height = max_height
        width = int(height * aspect)

    return cv2.resize(frame, (width, height))
