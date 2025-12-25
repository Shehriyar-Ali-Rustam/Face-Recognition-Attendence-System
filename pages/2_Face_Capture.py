"""
Face Capture Page
Capture and save face images for training
"""

import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.operations import StudentOperations
from utils.face_detector import FaceDetector, LivenessDetector
from utils.helpers import save_face_image, get_student_image_count, create_student_folder
from config.settings import CAPTURE_SETTINGS, DATASET_DIR

# Page configuration
st.set_page_config(
    page_title="Face Capture",
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
    .status-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .progress-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
        text-align: center;
    }
    .instruction-box {
        background-color: #e7f3ff;
        border-left: 4px solid #0f3460;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
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
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    if 'captured_count' not in st.session_state:
        st.session_state.captured_count = 0
    if 'selected_student' not in st.session_state:
        st.session_state.selected_student = None
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


def capture_face_images(student_id: str, student_name: str, num_images: int = 50):
    """Capture face images from camera"""
    face_detector = FaceDetector()
    liveness_detector = LivenessDetector()

    # Camera placeholder
    camera_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    info_placeholder = st.empty()

    cap = cv2.VideoCapture(st.session_state.camera_index)

    if not cap.isOpened():
        st.error("Failed to open camera. Please check camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured = 0
    last_capture_time = 0
    capture_interval = CAPTURE_SETTINGS['capture_interval']

    stop_button = st.button("Stop Capture", key="stop_capture")

    while captured < num_images and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Detect faces
        faces = face_detector.detect_faces_haar(frame)

        current_time = time.time()

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            # Check liveness
            face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

            # Capture at intervals
            if current_time - last_capture_time >= capture_interval:
                # Extract and save face
                face_img = frame[max(0, y-20):min(frame.shape[0], y+h+20),
                                 max(0, x-20):min(frame.shape[1], x+w+20)]

                if face_img.size > 0:
                    # Resize face image
                    face_resized = cv2.resize(face_img, CAPTURE_SETTINGS['image_size'])
                    save_face_image(student_id, face_resized, captured + 1)
                    captured += 1
                    last_capture_time = current_time

                    # Update progress
                    progress = captured / num_images
                    progress_bar.progress(progress)
                    status_placeholder.markdown(
                        f'<p class="progress-text">Captured: {captured} / {num_images}</p>',
                        unsafe_allow_html=True
                    )

        # Add overlay text
        cv2.putText(display_frame, f"Student: {student_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Captured: {captured}/{num_images}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not faces:
            cv2.putText(display_frame, "No face detected - Please face the camera", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Convert to RGB for Streamlit
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)

        # Instructions
        info_placeholder.markdown("""
        <div class="instruction-box">
            <strong>Instructions:</strong><br>
            - Keep your face centered in the frame<br>
            - Slowly move your head left, right, up, and down<br>
            - Vary your expression slightly<br>
            - Ensure good lighting on your face
        </div>
        """, unsafe_allow_html=True)

        # Check stop button
        if stop_button:
            break

        time.sleep(0.03)

    cap.release()
    camera_placeholder.empty()

    if captured > 0:
        # Update student record
        StudentOperations.update_student(student_id, image_count=get_student_image_count(student_id))
        st.success(f"Successfully captured {captured} images for {student_name}")
        st.info("You can now proceed to 'Model Training' to train the recognition model.")
    else:
        st.warning("No images were captured. Please try again.")


def show_existing_images(student_id: str):
    """Show existing captured images for a student"""
    images_path = DATASET_DIR / student_id
    if not images_path.exists():
        st.info("No images captured yet for this student.")
        return

    images = list(images_path.glob('*.jpg'))
    if not images:
        st.info("No images captured yet for this student.")
        return

    st.markdown(f"**Existing Images: {len(images)}**")

    # Display images in a grid
    cols = st.columns(5)
    for idx, img_path in enumerate(images[:10]):  # Show first 10
        with cols[idx % 5]:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, width=100)

    if len(images) > 10:
        st.caption(f"... and {len(images) - 10} more images")

    # Delete option
    if st.button("Delete All Images", type="secondary"):
        import shutil
        shutil.rmtree(images_path)
        StudentOperations.update_student(student_id, image_count=0)
        st.success("All images deleted.")
        st.rerun()


def main():
    initialize_session_state()

    st.markdown('<p class="main-header">Face Capture</p>', unsafe_allow_html=True)

    # Get students without enough images
    students = StudentOperations.get_all_students()

    if not students:
        st.warning("No students registered. Please register a student first.")
        return

    # Sidebar settings
    with st.sidebar:
        st.markdown("### Capture Settings")

        # Camera selection
        available_cameras = get_available_cameras()
        st.session_state.camera_index = st.selectbox(
            "Select Camera",
            options=available_cameras,
            format_func=lambda x: f"Camera {x}"
        )

        # Number of images
        num_images = st.slider(
            "Images to Capture",
            min_value=10,
            max_value=100,
            value=CAPTURE_SETTINGS['num_images'],
            step=10
        )

        st.markdown("---")
        st.markdown("### Tips")
        st.markdown("""
        - Capture in good lighting
        - Move head slowly during capture
        - Remove glasses if possible
        - Keep neutral background
        """)

    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="sub-header">Select Student</p>', unsafe_allow_html=True)

        # Create student selection
        student_options = {s.student_id: f"{s.name} ({s.student_id})" for s in students}

        selected_id = st.selectbox(
            "Choose a student",
            options=list(student_options.keys()),
            format_func=lambda x: student_options[x]
        )

        if selected_id:
            student = StudentOperations.get_student(selected_id)
            image_count = get_student_image_count(selected_id)

            st.markdown("---")
            st.markdown(f"**Name:** {student.name}")
            st.markdown(f"**Department:** {student.department or 'N/A'}")
            st.markdown(f"**Current Images:** {image_count}")

            if image_count >= 10:
                st.success("Sufficient images for training")
            else:
                st.warning(f"Need at least 10 images (have {image_count})")

            st.markdown("---")
            show_existing_images(selected_id)

    with col2:
        st.markdown('<p class="sub-header">Camera Preview & Capture</p>', unsafe_allow_html=True)

        if selected_id:
            student = StudentOperations.get_student(selected_id)

            # Start capture button
            if st.button("Start Capture", type="primary", use_container_width=True):
                capture_face_images(selected_id, student.name, num_images)

            # Preview camera
            st.markdown("---")
            if st.button("Preview Camera"):
                cap = cv2.VideoCapture(st.session_state.camera_index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, channels="RGB", use_container_width=True)
                    cap.release()
                else:
                    st.error("Could not open camera")


if __name__ == "__main__":
    main()
