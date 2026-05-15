"""
Student Registration Page
Face Recognition Attendance System
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.operations import StudentOperations
from utils.helpers import validate_email, validate_phone, get_student_image_count

# Page configuration
st.set_page_config(
    page_title="Student Registration",
    page_icon="",
    layout="wide"
)

# Custom CSS for professional look
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
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    .student-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    div[data-testid="stForm"] {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


def show_registration_form():
    """Display student registration form"""
    st.markdown('<p class="sub-header">New Student Registration</p>', unsafe_allow_html=True)

    with st.form("registration_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            student_id = st.text_input(
                "Student ID *",
                placeholder="e.g., STU001",
                help="Unique identifier for the student"
            )
            name = st.text_input(
                "Full Name *",
                placeholder="Enter full name"
            )
            email = st.text_input(
                "Email",
                placeholder="student@example.com"
            )

        with col2:
            phone = st.text_input(
                "Phone Number",
                placeholder="+1234567890"
            )
            department = st.selectbox(
                "Department",
                options=["", "Computer Science", "Electrical Engineering",
                         "Mechanical Engineering", "Civil Engineering",
                         "Business Administration", "Other"]
            )
            batch = st.text_input(
                "Batch/Year",
                placeholder="e.g., 2024"
            )

        submitted = st.form_submit_button("Register Student", use_container_width=True)

        if submitted:
            # Validation
            errors = []
            if not student_id or not student_id.strip():
                errors.append("Student ID is required")
            if not name or not name.strip():
                errors.append("Full Name is required")
            if email and not validate_email(email):
                errors.append("Invalid email format")
            if phone and not validate_phone(phone):
                errors.append("Invalid phone number format")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Create student
                success, message = StudentOperations.create_student(
                    student_id=student_id.strip(),
                    name=name.strip(),
                    email=email.strip() if email else None,
                    phone=phone.strip() if phone else None,
                    department=department if department else None,
                    batch=batch.strip() if batch else None
                )

                if success:
                    st.success(message)
                    st.info("Next step: Go to 'Face Capture' to capture face images for this student.")
                else:
                    st.error(message)


def show_student_list():
    """Display list of registered students"""
    st.markdown('<p class="sub-header">Registered Students</p>', unsafe_allow_html=True)

    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search by name or ID", placeholder="Type to search...")
    with col2:
        show_inactive = st.checkbox("Show inactive", value=False)

    # Get students
    if search_query:
        students = StudentOperations.search_students(search_query)
    else:
        students = StudentOperations.get_all_students(active_only=not show_inactive)

    if not students:
        st.info("No students found. Register a new student above.")
        return

    # Display count
    st.markdown(f"**Total: {len(students)} student(s)**")

    # Display students in a table-like format
    for student in students:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 3, 2, 2])

            with col1:
                st.markdown(f"**{student.student_id}**")
            with col2:
                st.markdown(f"{student.name}")
                if student.department:
                    st.caption(student.department)
            with col3:
                image_count = get_student_image_count(student.student_id)
                status = "Ready" if image_count >= 10 else f"Need images ({image_count}/10)"
                if image_count >= 10:
                    st.markdown(f"Images: {image_count}")
                else:
                    st.warning(status)
            with col4:
                col_edit, col_delete = st.columns(2)
                with col_edit:
                    if st.button("Edit", key=f"edit_{student.student_id}"):
                        st.session_state['edit_student'] = student.student_id
                        st.rerun()
                with col_delete:
                    if st.button("Delete", key=f"del_{student.student_id}"):
                        st.session_state['delete_student'] = student.student_id

            st.divider()

    # Handle delete confirmation
    if 'delete_student' in st.session_state:
        student_id = st.session_state['delete_student']
        st.warning(f"Are you sure you want to delete student {student_id}?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete", type="primary"):
                success, msg = StudentOperations.delete_student(student_id, soft_delete=True)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                del st.session_state['delete_student']
                st.rerun()
        with col2:
            if st.button("Cancel"):
                del st.session_state['delete_student']
                st.rerun()


def show_edit_form(student_id: str):
    """Display edit form for a student"""
    student = StudentOperations.get_student(student_id)
    if not student:
        st.error("Student not found")
        return

    st.markdown(f'<p class="sub-header">Edit Student: {student.name}</p>', unsafe_allow_html=True)

    with st.form("edit_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.text_input("Student ID", value=student.student_id, disabled=True)
            name = st.text_input("Full Name *", value=student.name)
            email = st.text_input("Email", value=student.email or "")

        with col2:
            phone = st.text_input("Phone Number", value=student.phone or "")
            department = st.selectbox(
                "Department",
                options=["", "Computer Science", "Electrical Engineering",
                         "Mechanical Engineering", "Civil Engineering",
                         "Business Administration", "Other"],
                index=0 if not student.department else
                ["", "Computer Science", "Electrical Engineering",
                 "Mechanical Engineering", "Civil Engineering",
                 "Business Administration", "Other"].index(student.department)
                if student.department in ["Computer Science", "Electrical Engineering",
                                          "Mechanical Engineering", "Civil Engineering",
                                          "Business Administration", "Other"] else 0
            )
            batch = st.text_input("Batch/Year", value=student.batch or "")

        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.form_submit_button("Save Changes", use_container_width=True):
                success, msg = StudentOperations.update_student(
                    student_id,
                    name=name.strip(),
                    email=email.strip() if email else None,
                    phone=phone.strip() if phone else None,
                    department=department if department else None,
                    batch=batch.strip() if batch else None
                )
                if success:
                    st.success(msg)
                    del st.session_state['edit_student']
                    st.rerun()
                else:
                    st.error(msg)

    if st.button("Cancel"):
        del st.session_state['edit_student']
        st.rerun()


def main():
    st.markdown('<p class="main-header">Student Registration</p>', unsafe_allow_html=True)

    # Check if editing
    if 'edit_student' in st.session_state:
        show_edit_form(st.session_state['edit_student'])
    else:
        # Tabs for registration and list
        tab1, tab2 = st.tabs(["Register New Student", "View All Students"])

        with tab1:
            show_registration_form()

        with tab2:
            show_student_list()


if __name__ == "__main__":
    main()
