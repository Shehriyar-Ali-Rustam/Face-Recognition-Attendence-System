"""Utils package - imports are lazy loaded to avoid startup errors"""

# Only import non-opencv dependent utilities at startup
from .helpers import (
    setup_logging, create_student_folder, get_student_image_count,
    save_face_image, delete_student_images, format_datetime, format_date,
    format_time, get_date_range, generate_unique_id, validate_email,
    validate_phone, calculate_attendance_percentage, get_status_color,
    FrameRateCalculator
)
from .export import ExportManager, generate_attendance_summary_report

# Lazy load opencv-dependent modules
def get_face_detector():
    from .face_detector import FaceDetector
    return FaceDetector

def get_liveness_detector():
    from .face_detector import LivenessDetector
    return LivenessDetector

def get_face_recognizer():
    from .face_recognizer import FaceRecognizer
    return FaceRecognizer

def get_lbph_recognizer():
    from .face_recognizer import LBPHRecognizer
    return LBPHRecognizer

def get_camera_manager():
    from .camera import CameraManager
    return CameraManager
