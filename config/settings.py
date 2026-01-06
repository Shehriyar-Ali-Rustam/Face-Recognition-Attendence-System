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
    "tolerance": 0.4,  # Lower = more strict matching (0.4 is strict, 0.6 is lenient)
    "num_jitters": 10,  # Number of times to re-sample face (higher = more accurate but slower)
    "encoding_model": "large",  # 'small' or 'large' - large is more accurate
    "min_face_size": 80,  # Minimum face size in pixels for quality
    "min_encodings_per_student": 5,  # Minimum encodings needed for reliable recognition
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
    "confidence_threshold": 0.65,  # Minimum confidence for marking attendance (stricter)
    "unknown_threshold": 0.5,  # Below this, mark as unknown
}

# Liveness detection settings
LIVENESS_SETTINGS = {
    "enabled": True,
    "blink_threshold": 0.25,
    "movement_threshold": 20,
    "required_blinks": 2,
}

# UI Theme colors (Professional Indigo)
THEME = {
    "primary": "#4f46e5",
    "primary_dark": "#4338ca",
    "primary_light": "#6366f1",
    "secondary": "#1e1b4b",
    "accent": "#7c3aed",
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "background": "#f8fafc",
    "card": "#ffffff",
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log",
}
