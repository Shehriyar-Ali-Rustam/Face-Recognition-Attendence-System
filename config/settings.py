"""
Application Configuration Settings
Face Recognition Attendance System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database settings
DATABASE_PATH = BASE_DIR / "database" / "attendance.db"

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
EXPORTS_DIR = BASE_DIR / "exports"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATASET_DIR, TRAINED_MODELS_DIR, EXPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Face recognition settings
FACE_RECOGNITION_SETTINGS = {
    "model": "hog",  # 'hog' for CPU, 'cnn' for GPU
    "tolerance": 0.5,  # Lower = more strict matching
    "num_jitters": 1,  # Number of times to re-sample face
    "encoding_model": "large",  # 'small' or 'large'
}

# Face detection settings
FACE_DETECTION_SETTINGS = {
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": (100, 100),
    "max_size": (800, 800),
}

# Camera settings
CAMERA_SETTINGS = {
    "default_camera": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
}

# Dataset capture settings
CAPTURE_SETTINGS = {
    "num_images": 50,  # Number of images to capture per person
    "capture_interval": 0.1,  # Seconds between captures
    "image_size": (200, 200),
}

# Attendance settings
ATTENDANCE_SETTINGS = {
    "duplicate_check_hours": 24,  # Hours before allowing duplicate entry
    "confidence_threshold": 0.6,  # Minimum confidence for marking attendance
    "unknown_threshold": 0.4,  # Below this, mark as unknown
}

# Liveness detection settings
LIVENESS_SETTINGS = {
    "enabled": True,
    "blink_threshold": 0.25,
    "movement_threshold": 20,
    "required_blinks": 2,
}

# UI Theme colors (professional, minimal)
THEME = {
    "primary": "#1a1a2e",
    "secondary": "#16213e",
    "accent": "#0f3460",
    "text": "#eaeaea",
    "success": "#4ecca3",
    "warning": "#ffc107",
    "error": "#dc3545",
    "background": "#0f0f23",
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log",
}
