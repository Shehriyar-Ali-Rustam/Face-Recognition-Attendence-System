"""
Helper Utilities
Common functions used across the application
"""

import logging
import os
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import hashlib

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATASET_DIR, LOGS_DIR, LOGGING_CONFIG

# Lazy imports for optional dependencies
cv2 = None
np = None

def _load_cv2():
    global cv2, np
    if cv2 is None:
        import cv2 as _cv2
        import numpy as _np
        cv2 = _cv2
        np = _np

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_logging(module_name: str) -> logging.Logger:
    """Setup logging for a module"""
    return logging.getLogger(module_name)


def create_student_folder(student_id: str) -> Path:
    """Create a folder for storing student face images"""
    folder_path = DATASET_DIR / student_id
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def get_student_image_count(student_id: str) -> int:
    """Get the number of images for a student"""
    folder_path = DATASET_DIR / student_id
    if not folder_path.exists():
        return 0
    return len(list(folder_path.glob('*.jpg')))


def save_face_image(student_id: str, image,
                    image_index: int = None) -> str:
    """Save a face image for a student"""
    try:
        _load_cv2()
        folder_path = create_student_folder(student_id)

        if image_index is None:
            image_index = get_student_image_count(student_id) + 1

        filename = f"{student_id}_{image_index:04d}.jpg"
        filepath = folder_path / filename

        cv2.imwrite(str(filepath), image)
        logger.debug(f"Saved face image: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving face image: {str(e)}")
        return None


def delete_student_images(student_id: str) -> bool:
    """Delete all images for a student"""
    try:
        folder_path = DATASET_DIR / student_id
        if folder_path.exists():
            import shutil
            shutil.rmtree(folder_path)
            logger.info(f"Deleted images for student: {student_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting images: {str(e)}")
        return False


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime object to string"""
    if dt is None:
        return ""
    return dt.strftime(format_str)


def format_date(d: date, format_str: str = "%Y-%m-%d") -> str:
    """Format date object to string"""
    if d is None:
        return ""
    return d.strftime(format_str)


def format_time(t, format_str: str = "%H:%M:%S") -> str:
    """Format time object to string"""
    if t is None:
        return ""
    if hasattr(t, 'strftime'):
        return t.strftime(format_str)
    return str(t)


def get_date_range(period: str) -> tuple:
    """Get date range for a period"""
    today = date.today()

    if period == 'today':
        return today, today
    elif period == 'yesterday':
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    elif period == 'week':
        start = today - timedelta(days=today.weekday())
        return start, today
    elif period == 'month':
        start = today.replace(day=1)
        return start, today
    elif period == 'year':
        start = today.replace(month=1, day=1)
        return start, today
    else:
        return today, today


def generate_unique_id(prefix: str = "STU") -> str:
    """Generate a unique ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    hash_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:4].upper()
    return f"{prefix}{timestamp[-6:]}{hash_suffix}"


def validate_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Simple phone validation"""
    import re
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    # Check if remaining is digits and reasonable length
    return cleaned.isdigit() and 10 <= len(cleaned) <= 15


def calculate_attendance_percentage(present: int, total_days: int) -> float:
    """Calculate attendance percentage"""
    if total_days == 0:
        return 0.0
    return (present / total_days) * 100


def get_status_color(status: str) -> tuple:
    """Get color for status display (BGR format)"""
    colors = {
        'Present': (0, 200, 0),      # Green
        'Absent': (0, 0, 200),       # Red
        'Late': (0, 165, 255),       # Orange
        'Unknown': (128, 128, 128)   # Gray
    }
    return colors.get(status, (255, 255, 255))


def resize_image_for_display(image, max_size: tuple = (200, 200)):
    """Resize image while maintaining aspect ratio"""
    _load_cv2()
    height, width = image.shape[:2]
    max_width, max_height = max_size

    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))

    return image


def convert_cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL Image"""
    _load_cv2()
    from PIL import Image
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def convert_pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV image"""
    _load_cv2()
    import numpy as _np
    return cv2.cvtColor(_np.array(pil_image), cv2.COLOR_RGB2BGR)


class FrameRateCalculator:
    """Calculate and display frame rate"""

    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames
        self.frame_times = []
        self.last_time = None

    def update(self) -> float:
        """Update and return current FPS"""
        current_time = datetime.now()

        if self.last_time is not None:
            delta = (current_time - self.last_time).total_seconds()
            self.frame_times.append(delta)

            if len(self.frame_times) > self.avg_frames:
                self.frame_times.pop(0)

        self.last_time = current_time

        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0.0

        return 0.0
