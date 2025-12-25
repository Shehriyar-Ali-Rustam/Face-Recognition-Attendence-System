"""
Model Training Page
Train face recognition model with captured images
"""

import streamlit as st
import time
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.operations import StudentOperations, TrainingLogOperations
from utils.face_recognizer import FaceRecognizer, LBPHRecognizer
from utils.helpers import get_student_image_count
from config.settings import DATASET_DIR, TRAINED_MODELS_DIR

# Page configuration
st.set_page_config(
    page_title="Model Training",
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
    .stat-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .training-log {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        max-height: 300px;
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


def get_training_statistics():
    """Get statistics for training data"""
    students = StudentOperations.get_students_with_encodings()
    all_students = StudentOperations.get_all_students()

    total_students = len(all_students)
    students_with_images = 0
    total_images = 0
    ready_for_training = 0

    for student in all_students:
        img_count = get_student_image_count(student.student_id)
        if img_count > 0:
            students_with_images += 1
            total_images += img_count
        if img_count >= 10:
            ready_for_training += 1

    return {
        'total_students': total_students,
        'students_with_images': students_with_images,
        'total_images': total_images,
        'ready_for_training': ready_for_training
    }


def get_student_training_data():
    """Prepare student data for training"""
    students = StudentOperations.get_all_students()
    training_data = []

    for student in students:
        images_path = DATASET_DIR / student.student_id
        img_count = get_student_image_count(student.student_id)

        if img_count >= 10:
            training_data.append({
                'student_id': student.student_id,
                'name': student.name,
                'images_path': images_path,
                'image_count': img_count
            })

    return training_data


def train_model(use_dlib: bool = True, use_lbph: bool = True):
    """Train the face recognition model"""
    training_data = get_student_training_data()

    if not training_data:
        st.error("No students with sufficient images found. Need at least 10 images per student.")
        return False

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    logs = []
    start_time = time.time()

    def add_log(message):
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        with log_container:
            st.markdown(
                f'<div class="training-log">{"<br>".join(logs)}</div>',
                unsafe_allow_html=True
            )

    add_log(f"Starting training with {len(training_data)} students...")

    total_images = sum(d['image_count'] for d in training_data)
    add_log(f"Total images to process: {total_images}")

    success_messages = []

    # Train Dlib-based recognizer
    if use_dlib:
        try:
            add_log("Training Dlib face recognition model...")
            status_text.text("Training Dlib model (this may take a while)...")
            progress_bar.progress(0.2)

            recognizer = FaceRecognizer()
            success, msg = recognizer.train_model(training_data)

            if success:
                add_log(f"Dlib training complete: {msg}")
                success_messages.append(msg)
            else:
                add_log(f"Dlib training failed: {msg}")

            progress_bar.progress(0.5)

        except Exception as e:
            add_log(f"Error in Dlib training: {str(e)}")
            st.warning(f"Dlib training failed: {str(e)}")

    # Train LBPH recognizer
    if use_lbph:
        try:
            add_log("Training LBPH face recognition model...")
            status_text.text("Training LBPH model...")
            progress_bar.progress(0.7)

            lbph_recognizer = LBPHRecognizer()
            success, msg = lbph_recognizer.train_model(training_data)

            if success:
                add_log(f"LBPH training complete: {msg}")
                success_messages.append(msg)
            else:
                add_log(f"LBPH training failed: {msg}")

            progress_bar.progress(0.9)

        except Exception as e:
            add_log(f"Error in LBPH training: {str(e)}")
            st.warning(f"LBPH training failed: {str(e)}")

    # Complete
    elapsed_time = time.time() - start_time
    add_log(f"Training completed in {elapsed_time:.2f} seconds")
    progress_bar.progress(1.0)
    status_text.text("Training complete!")

    # Log to database
    if success_messages:
        TrainingLogOperations.create_training_log(
            num_students=len(training_data),
            num_images=total_images,
            model_path=str(TRAINED_MODELS_DIR),
            notes=" | ".join(success_messages)
        )
        return True

    return False


def show_training_history():
    """Display training history"""
    logs = TrainingLogOperations.get_all_training_logs()

    if not logs:
        st.info("No training history available.")
        return

    st.markdown('<p class="sub-header">Training History</p>', unsafe_allow_html=True)

    for log in logs[:10]:  # Show last 10
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

        with col1:
            st.markdown(f"**{log.training_date.strftime('%Y-%m-%d %H:%M')}**")
        with col2:
            st.markdown(f"Students: {log.num_students}")
        with col3:
            st.markdown(f"Images: {log.num_images}")
        with col4:
            st.markdown(f"_{log.status}_")

        st.divider()


def show_student_readiness():
    """Show which students are ready for training"""
    students = StudentOperations.get_all_students()

    ready = []
    not_ready = []

    for student in students:
        img_count = get_student_image_count(student.student_id)
        student_info = {
            'id': student.student_id,
            'name': student.name,
            'images': img_count
        }
        if img_count >= 10:
            ready.append(student_info)
        else:
            not_ready.append(student_info)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ready for Training**")
        if ready:
            for s in ready:
                st.markdown(f"- {s['name']} ({s['id']}): {s['images']} images")
        else:
            st.info("No students ready for training")

    with col2:
        st.markdown("**Need More Images**")
        if not_ready:
            for s in not_ready:
                st.markdown(f"- {s['name']} ({s['id']}): {s['images']}/10 images")
        else:
            st.success("All students have sufficient images")


def main():
    st.markdown('<p class="main-header">Model Training</p>', unsafe_allow_html=True)

    # Get statistics
    stats = get_training_statistics()

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_students']}</div>
            <div class="stat-label">Total Students</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['students_with_images']}</div>
            <div class="stat-label">With Images</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_images']}</div>
            <div class="stat-label">Total Images</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['ready_for_training']}</div>
            <div class="stat-label">Ready to Train</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Training section
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<p class="sub-header">Train Model</p>', unsafe_allow_html=True)

        # Model options
        st.markdown("**Select Recognition Models:**")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_dlib = st.checkbox("Dlib (High Accuracy)", value=True,
                                   help="Uses deep learning for better accuracy")
        with col_opt2:
            use_lbph = st.checkbox("LBPH (Lightweight)", value=True,
                                   help="Faster, works offline")

        st.markdown("---")

        if stats['ready_for_training'] == 0:
            st.warning("No students are ready for training. Each student needs at least 10 face images.")
        else:
            if st.button("Start Training", type="primary", use_container_width=True):
                with st.spinner("Training in progress..."):
                    success = train_model(use_dlib=use_dlib, use_lbph=use_lbph)

                if success:
                    st.success("Model training completed successfully!")
                    st.info("You can now use 'Mark Attendance' to recognize faces.")
                    st.balloons()
                else:
                    st.error("Training failed. Please check the logs above.")

    with col_right:
        st.markdown('<p class="sub-header">Model Status</p>', unsafe_allow_html=True)

        # Check if models exist
        dlib_model = TRAINED_MODELS_DIR / "face_encodings.pkl"
        lbph_model = TRAINED_MODELS_DIR / "lbph_model.yml"

        if dlib_model.exists():
            mod_time = datetime.fromtimestamp(dlib_model.stat().st_mtime)
            st.markdown(f"**Dlib Model:** Trained")
            st.caption(f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.markdown("**Dlib Model:** Not trained")

        st.markdown("---")

        if lbph_model.exists():
            mod_time = datetime.fromtimestamp(lbph_model.stat().st_mtime)
            st.markdown(f"**LBPH Model:** Trained")
            st.caption(f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.markdown("**LBPH Model:** Not trained")

    st.markdown("---")

    # Student readiness
    st.markdown('<p class="sub-header">Student Readiness</p>', unsafe_allow_html=True)
    show_student_readiness()

    st.markdown("---")

    # Training history
    show_training_history()


if __name__ == "__main__":
    main()
