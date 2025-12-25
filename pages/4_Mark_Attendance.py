"""
Mark Attendance Page
Real-time face recognition for attendance marking
"""

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime, date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.operations import StudentOperations, AttendanceOperations
from utils.face_detector import FaceDetector, LivenessDetector
from utils.face_recognizer import FaceRecognizer, LBPHRecognizer
from utils.helpers import format_time
from config.settings import ATTENDANCE_SETTINGS, TRAINED_MODELS_DIR, LIVENESS_SETTINGS

# Page configuration
st.set_page_config(
    page_title="Mark Attendance",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0f3460;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #16213e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .attendance-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .recognized-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .confidence-high {
        color: #28a745;
    }
    .confidence-medium {
        color: #ffc107;
    }
    .confidence-low {
        color: #dc3545;
    }
    .today-attendance {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .stButton>button {
        background-color: #1a1a2e;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0f3460;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'recognition_active' not in st.session_state:
        st.session_state.recognition_active = False
    if 'last_recognized' not in st.session_state:
        st.session_state.last_recognized = None
    if 'today_marked' not in st.session_state:
        st.session_state.today_marked = set()
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = 0


def get_available_cameras():
    """Get list of available cameras"""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras if cameras else [0]


def check_models_exist():
    """Check if recognition models are trained"""
    dlib_model = TRAINED_MODELS_DIR / "face_encodings.pkl"
    lbph_model = TRAINED_MODELS_DIR / "lbph_model.yml"
    return dlib_model.exists() or lbph_model.exists()


def run_attendance_recognition():
    """Run real-time face recognition for attendance"""
    if not check_models_exist():
        st.error("No trained models found. Please train the model first.")
        return

    # Initialize components
    face_detector = FaceDetector()
    liveness_detector = LivenessDetector() if LIVENESS_SETTINGS['enabled'] else None

    # Try to load recognizers
    dlib_recognizer = None
    lbph_recognizer = None

    dlib_model = TRAINED_MODELS_DIR / "face_encodings.pkl"
    lbph_model = TRAINED_MODELS_DIR / "lbph_model.yml"

    if dlib_model.exists():
        dlib_recognizer = FaceRecognizer()
    if lbph_model.exists():
        lbph_recognizer = LBPHRecognizer()

    if not dlib_recognizer and not lbph_recognizer:
        st.error("Failed to load recognition models.")
        return

    # UI elements
    camera_placeholder = st.empty()
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    attendance_placeholder = st.empty()

    cap = cv2.VideoCapture(st.session_state.camera_index)

    if not cap.isOpened():
        st.error("Failed to open camera. Please check camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Recognition state
    recognition_buffer = {}
    REQUIRED_FRAMES = 5
    last_attendance_time = {}
    COOLDOWN_SECONDS = 5

    stop_button = st.button("Stop Recognition", key="stop_recognition")

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        current_time = time.time()

        # Detect faces
        faces = face_detector.detect_faces_haar(frame)

        for (x, y, w, h) in faces:
            face_color = (128, 128, 128)  # Gray for unknown
            label = "Detecting..."

            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Check liveness if enabled
            is_live = True
            if liveness_detector:
                is_live, blink_count, has_movement = liveness_detector.check_liveness(face_gray)

            # Recognize face
            student_id = "Unknown"
            name = "Unknown"
            confidence = 0.0

            if dlib_recognizer:
                # Get larger region for dlib
                padding = 50
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                face_region = frame[y1:y2, x1:x2]

                encoding = dlib_recognizer.get_face_encoding(face_region)
                if encoding is not None:
                    student_id, name, confidence = dlib_recognizer.recognize_face(encoding)

            elif lbph_recognizer:
                face_resized = cv2.resize(face_gray, (200, 200))
                student_id, name, confidence = lbph_recognizer.recognize_face(face_resized)

            # Update recognition buffer for stability
            if student_id != "Unknown":
                if student_id not in recognition_buffer:
                    recognition_buffer[student_id] = {'count': 0, 'confidence': []}

                recognition_buffer[student_id]['count'] += 1
                recognition_buffer[student_id]['confidence'].append(confidence)

                # Check if recognized consistently
                if recognition_buffer[student_id]['count'] >= REQUIRED_FRAMES:
                    avg_confidence = np.mean(recognition_buffer[student_id]['confidence'])

                    # Check cooldown
                    last_time = last_attendance_time.get(student_id, 0)
                    if current_time - last_time > COOLDOWN_SECONDS:
                        # Check if already marked today
                        if not AttendanceOperations.check_attendance_exists(student_id):
                            # Mark attendance
                            success, msg = AttendanceOperations.mark_attendance(
                                student_id=student_id,
                                confidence_score=avg_confidence,
                                status='Present'
                            )
                            if success:
                                st.session_state.today_marked.add(student_id)
                                result_placeholder.success(
                                    f"Attendance marked for {name} ({student_id}) - Confidence: {avg_confidence:.1%}"
                                )
                            last_attendance_time[student_id] = current_time
                        else:
                            result_placeholder.info(
                                f"{name} ({student_id}) - Already marked today"
                            )
                            last_attendance_time[student_id] = current_time

                    # Set display values
                    face_color = (0, 200, 0)  # Green for recognized
                    label = f"{name} ({confidence:.0%})"

                    # Reset buffer for this person
                    recognition_buffer[student_id] = {'count': 0, 'confidence': []}

            else:
                # Unknown face
                face_color = (0, 0, 200)  # Red for unknown
                label = "Unknown"
                recognition_buffer = {}

            # Draw face box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), face_color, 2)

            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(display_frame, (x, y - 25), (x + label_size[0] + 10, y), face_color, -1)
            cv2.putText(display_frame, label, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show liveness status if enabled
            if liveness_detector:
                liveness_text = "Live" if is_live else "Check liveness"
                liveness_color = (0, 255, 0) if is_live else (0, 165, 255)
                cv2.putText(display_frame, liveness_text, (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, liveness_color, 1)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add status
        status_text = f"Faces: {len(faces)} | Marked today: {len(st.session_state.today_marked)}"
        cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Convert to RGB for Streamlit
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)

        # Update today's attendance list
        today_records = AttendanceOperations.get_daily_attendance()
        if today_records:
            with attendance_placeholder.container():
                st.markdown("**Today's Attendance:**")
                for record in today_records[-5:]:  # Show last 5
                    student = StudentOperations.get_student(record.student_id)
                    name = student.name if student else record.student_id
                    st.markdown(f"- {name}: {format_time(record.time_in)}")

        time.sleep(0.03)

    cap.release()
    camera_placeholder.empty()


def show_today_attendance():
    """Show today's attendance records"""
    records = AttendanceOperations.get_daily_attendance()

    if not records:
        st.info("No attendance records for today yet.")
        return

    st.markdown(f"**Total Present: {len(records)}**")

    for record in records:
        student = StudentOperations.get_student(record.student_id)
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            st.markdown(f"**{student.name if student else record.student_id}**")
        with col2:
            st.markdown(f"ID: {record.student_id}")
        with col3:
            time_in = format_time(record.time_in)
            time_out = format_time(record.time_out) if record.time_out else "-"
            st.markdown(f"In: {time_in} | Out: {time_out}")
        with col4:
            conf = record.confidence_score
            if conf:
                if conf >= 0.8:
                    st.markdown(f'<span class="confidence-high">{conf:.0%}</span>', unsafe_allow_html=True)
                elif conf >= 0.6:
                    st.markdown(f'<span class="confidence-medium">{conf:.0%}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="confidence-low">{conf:.0%}</span>', unsafe_allow_html=True)

        st.divider()


def main():
    initialize_session_state()

    st.markdown('<p class="main-header">Mark Attendance</p>', unsafe_allow_html=True)

    # Check if models exist
    if not check_models_exist():
        st.warning("No trained recognition models found.")
        st.info("Please go to 'Model Training' to train the recognition model first.")
        return

    # Sidebar settings
    with st.sidebar:
        st.markdown("### Settings")

        # Camera selection
        available_cameras = get_available_cameras()
        st.session_state.camera_index = st.selectbox(
            "Select Camera",
            options=available_cameras,
            format_func=lambda x: f"Camera {x}"
        )

        # Liveness detection
        liveness_enabled = st.checkbox(
            "Enable Liveness Detection",
            value=LIVENESS_SETTINGS['enabled'],
            help="Detect blinks and movement to prevent photo spoofing"
        )

        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Face the camera clearly
        2. Keep good lighting
        3. Stay still for recognition
        4. Wait for confirmation
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<p class="sub-header">Camera Feed</p>', unsafe_allow_html=True)

        # Start recognition button
        if st.button("Start Recognition", type="primary", use_container_width=True):
            run_attendance_recognition()

    with col2:
        st.markdown('<p class="sub-header">Today\'s Attendance</p>', unsafe_allow_html=True)

        # Refresh button
        if st.button("Refresh", key="refresh_attendance"):
            st.rerun()

        st.markdown("---")
        show_today_attendance()

    # Manual attendance option
    st.markdown("---")
    with st.expander("Manual Attendance Entry"):
        st.markdown("Mark attendance manually (in case face recognition fails)")

        students = StudentOperations.get_all_students()
        if students:
            student_options = {s.student_id: f"{s.name} ({s.student_id})" for s in students}
            selected_student = st.selectbox(
                "Select Student",
                options=list(student_options.keys()),
                format_func=lambda x: student_options[x]
            )

            if st.button("Mark Present"):
                success, msg = AttendanceOperations.mark_attendance(
                    student_id=selected_student,
                    confidence_score=None,
                    status='Present',
                    notes='Manual entry'
                )
                if success:
                    st.success(msg)
                else:
                    st.error(msg)


if __name__ == "__main__":
    main()
