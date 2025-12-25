"""
Face Recognition Attendance System
Main Application with Separate Student/Admin Portals
"""

import streamlit as st
from datetime import datetime, date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from database.models import init_database
init_database()

from database.operations import (
    UserOperations, StudentOperations, AttendanceOperations,
    TrainingLogOperations
)

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

# Page config
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS with fixed text visibility
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* Fix all text colors for visibility */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #ffffff;
    }

    /* Main content area text */
    .main .block-container {
        color: #ffffff;
    }

    /* Form labels */
    .stTextInput label, .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #ffffff !important;
    }

    /* Bold text in markdown */
    .stMarkdown strong {
        color: #ffffff !important;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 12px 15px;
        font-size: 14px;
        background: white;
        color: #1a1a2e !important;
    }

    /* Select box */
    .stSelectbox > div > div {
        background: white !important;
        border-radius: 10px;
        color: #1a1a2e !important;
    }

    .stSelectbox > div > div > div {
        color: #1a1a2e !important;
    }

    /* Selectbox - selected value text */
    .stSelectbox [data-baseweb="select"] {
        background: white !important;
    }

    .stSelectbox [data-baseweb="select"] * {
        color: #1a1a2e !important;
    }

    .stSelectbox [data-baseweb="select"] span {
        color: #1a1a2e !important;
    }

    /* Dropdown menu options */
    [data-baseweb="popover"] {
        background: white !important;
    }

    [data-baseweb="popover"] li {
        color: #1a1a2e !important;
    }

    [data-baseweb="popover"] li:hover {
        background: #f0f0f0 !important;
    }

    /* Select menu list */
    [data-baseweb="menu"] {
        background: white !important;
    }

    [data-baseweb="menu"] * {
        color: #1a1a2e !important;
    }

    /* Selectbox input area */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: white !important;
        color: #1a1a2e !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
        color: #1a1a2e !important;
    }

    /* Slider - force all text white */
    .stSlider > div > div > div {
        color: #ffffff !important;
    }

    .stSlider label {
        color: #ffffff !important;
    }

    .stSlider p {
        color: #ffffff !important;
    }

    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #ffffff !important;
    }

    /* Slider value text - the number display */
    .stSlider [data-testid="stThumbValue"] {
        color: #1a1a2e !important;
        background: white !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
    }

    /* Slider current value display box */
    .stSlider div[data-baseweb="slider"] + div {
        color: #ffffff !important;
    }

    /* All text inside slider container */
    [data-testid="stSlider"] * {
        color: #ffffff !important;
    }

    /* Slider value bubble/tooltip */
    [data-baseweb="tooltip"] {
        background: white !important;
        color: #1a1a2e !important;
    }

    [data-baseweb="tooltip"] * {
        color: #1a1a2e !important;
    }

    /* Slider thumb value display */
    div[role="slider"] + div {
        color: #1a1a2e !important;
        background: white !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Slider track */
    .stSlider [data-baseweb="slider"] {
        margin-top: 10px;
    }

    /* Checkbox text */
    .stCheckbox > label > span {
        color: #ffffff !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }

    /* Role cards */
    .role-card {
        background: white;
        border-radius: 20px;
        padding: 40px 30px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        cursor: pointer;
        transition: all 0.3s ease;
        border: 3px solid transparent;
    }

    .role-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.25);
        border-color: #0f3460;
    }

    .role-icon {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 48px;
        color: white;
    }

    .role-title {
        font-size: 24px;
        font-weight: 700;
        color: #1a1a2e !important;
        margin-bottom: 10px;
    }

    .role-desc {
        font-size: 14px;
        color: #6c757d !important;
        line-height: 1.5;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 15px;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Stat cards */
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 15px;
    }

    .stat-value {
        font-size: 36px;
        font-weight: 700;
        color: #1a1a2e !important;
    }

    .stat-label {
        font-size: 13px;
        color: #6c757d !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }

    /* Header bar */
    .header-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 15px;
        margin-bottom: 30px;
    }

    .header-bar h2, .header-bar p {
        color: white !important;
    }

    /* Section title with white background */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff !important;
        margin: 25px 0 15px 0;
        padding: 15px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }

    /* Attendance row */
    .attendance-row {
        background: white;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .attendance-row strong {
        color: #1a1a2e !important;
    }

    .attendance-row span {
        color: #6c757d !important;
    }

    /* Status badges */
    .status-present {
        background: #d4edda;
        color: #155724 !important;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    .status-absent {
        background: #f8d7da;
        color: #721c24 !important;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Profile card */
    .profile-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }

    .profile-card h3 {
        color: #1a1a2e !important;
    }

    .profile-card p {
        color: #6c757d !important;
    }

    .profile-avatar {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 48px;
        color: white !important;
    }

    /* Form container with background */
    .stForm {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 15px;
    }

    /* Info, warning, error, success messages */
    .stAlert {
        border-radius: 10px;
    }

    /* Horizontal rule */
    hr {
        border-color: rgba(255,255,255,0.2);
    }

    /* Write text (st.write) */
    .element-container .stMarkdown p {
        color: #ffffff !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}

    /* Fix write statements inside columns */
    [data-testid="column"] p {
        color: #ffffff !important;
    }

    /* Fix info box text */
    .stInfo, .stWarning, .stError, .stSuccess {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    defaults = {
        'logged_in': False,
        'user_role': None,
        'student_id': None,
        'username': None,
        'page': 'role_select',
        'selected_role': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_role_selection():
    """Show role selection page - Student or Admin"""
    st.markdown("""
    <div style="text-align:center;padding:40px 0;">
        <h1 style="color:white;font-size:36px;margin-bottom:10px;">Face Recognition</h1>
        <p style="color:rgba(255,255,255,0.7);font-size:16px;">Attendance Management System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color:white;text-align:center;margin-bottom:30px;'>Select Your Portal</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            <div class="role-card">
                <div class="role-icon">S</div>
                <div class="role-title">Student</div>
                <div class="role-desc">Mark attendance, view your records, and manage profile</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Enter as Student", key="student_btn", use_container_width=True):
                st.session_state.selected_role = 'student'
                st.session_state.page = 'student_login'
                st.rerun()

        with col_b:
            st.markdown("""
            <div class="role-card">
                <div class="role-icon">A</div>
                <div class="role-title">Admin</div>
                <div class="role-desc">Manage students, train model, and view reports</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Enter as Admin", key="admin_btn", use_container_width=True):
                st.session_state.selected_role = 'admin'
                st.session_state.page = 'admin_login'
                st.rerun()


def show_student_login():
    """Show student login page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:30px 0;">
            <h1 style="color:white;font-size:32px;margin-bottom:10px;">Student Portal</h1>
            <p style="color:rgba(255,255,255,0.7);">Login to access your dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("### Login")
            username = st.text_input("Username", placeholder="Enter your username", key="student_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="student_password")

            if st.button("Login", use_container_width=True, key="student_login_btn"):
                if username and password:
                    success, role, student_id = UserOperations.authenticate(username, password)
                    if success and role == 'student':
                        st.session_state.logged_in = True
                        st.session_state.user_role = role
                        st.session_state.student_id = student_id
                        st.session_state.username = username
                        st.session_state.page = 'student_dashboard'
                        st.rerun()
                    elif success and role == 'admin':
                        st.error("This is an admin account. Please use Admin Portal.")
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter username and password")

            st.markdown("---")

            if st.button("Create New Account", use_container_width=True, key="student_register_btn"):
                st.session_state.page = 'student_register'
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Back to Portal Selection", use_container_width=True, key="student_back_btn"):
                st.session_state.page = 'role_select'
                st.rerun()


def show_student_register():
    """Show student registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:30px 0;">
            <h1 style="color:white;font-size:32px;margin-bottom:10px;">Student Registration</h1>
            <p style="color:rgba(255,255,255,0.7);">Create your account</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("student_register_form"):
            st.markdown("**Account Details**")
            username = st.text_input("Username *")
            password = st.text_input("Password *", type="password")
            confirm_password = st.text_input("Confirm Password *", type="password")

            st.markdown("**Personal Information**")
            col_a, col_b = st.columns(2)
            with col_a:
                student_id = st.text_input("Student ID *")
                name = st.text_input("Full Name *")
                email = st.text_input("Email")
            with col_b:
                phone = st.text_input("Phone")
                department = st.selectbox("Department", [
                    "", "Computer Science", "Electrical Engineering",
                    "Mechanical Engineering", "Civil Engineering", "Other"
                ])
                batch = st.text_input("Batch/Year")

            col_c, col_d = st.columns(2)
            with col_c:
                semester = st.selectbox("Semester", ["", "1", "2", "3", "4", "5", "6", "7", "8"])
            with col_d:
                section = st.text_input("Section")

            submitted = st.form_submit_button("Register", use_container_width=True)

            if submitted:
                if not all([username, password, student_id, name]):
                    st.error("Please fill all required fields (marked with *)")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, msg = StudentOperations.create_student(
                        student_id=student_id, name=name,
                        email=email or None, phone=phone or None,
                        department=department or None, batch=batch or None,
                        semester=semester or None, section=section or None
                    )
                    if success:
                        success2, msg2 = UserOperations.create_user(
                            username=username, password=password,
                            role='student', student_id=student_id
                        )
                        if success2:
                            st.success("Registration successful! Please login.")
                            st.session_state.page = 'student_login'
                            st.rerun()
                        else:
                            st.error(msg2)
                    else:
                        st.error(msg)

        if st.button("Back to Login", use_container_width=True):
            st.session_state.page = 'student_login'
            st.rerun()


def show_admin_login():
    """Show admin login page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:30px 0;">
            <h1 style="color:white;font-size:32px;margin-bottom:10px;">Admin Portal</h1>
            <p style="color:rgba(255,255,255,0.7);">Login to manage system</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("### Admin Login")
            username = st.text_input("Username", placeholder="Enter admin username", key="admin_username")
            password = st.text_input("Password", type="password", placeholder="Enter admin password", key="admin_password")

            if st.button("Login", use_container_width=True, key="admin_login_btn"):
                if username and password:
                    success, role, student_id = UserOperations.authenticate(username, password)
                    if success and role == 'admin':
                        st.session_state.logged_in = True
                        st.session_state.user_role = role
                        st.session_state.student_id = student_id
                        st.session_state.username = username
                        st.session_state.page = 'admin_dashboard'
                        st.rerun()
                    elif success and role == 'student':
                        st.error("This is a student account. Please use Student Portal.")
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter username and password")

            st.markdown("---")
            st.markdown("**Default Admin:** admin / admin123")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Register New Admin", use_container_width=True, key="admin_register_nav_btn"):
                st.session_state.page = 'admin_register_self'
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Back to Portal Selection", use_container_width=True, key="admin_back_btn"):
                st.session_state.page = 'role_select'
                st.rerun()


def show_admin_register_self():
    """Show admin self-registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:30px 0;">
            <h1 style="color:white;font-size:32px;margin-bottom:10px;">Admin Registration</h1>
            <p style="color:rgba(255,255,255,0.7);">Create new admin account</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("admin_register_form"):
            st.markdown("**Account Details**")
            username = st.text_input("Username *")
            password = st.text_input("Password *", type="password")
            confirm_password = st.text_input("Confirm Password *", type="password")
            admin_code = st.text_input("Admin Registration Code *", type="password",
                                       help="Contact system administrator for the code")

            submitted = st.form_submit_button("Register", use_container_width=True)

            if submitted:
                if not all([username, password, admin_code]):
                    st.error("Please fill all required fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif admin_code != "ADMIN2024":
                    st.error("Invalid admin registration code")
                else:
                    success, msg = UserOperations.create_user(
                        username=username, password=password, role='admin'
                    )
                    if success:
                        st.success("Admin registered! Please login.")
                        st.session_state.page = 'admin_login'
                        st.rerun()
                    else:
                        st.error(msg)

        if st.button("Back to Login", use_container_width=True):
            st.session_state.page = 'admin_login'
            st.rerun()


def show_student_dashboard():
    """Show student dashboard"""
    student = StudentOperations.get_student(st.session_state.student_id)
    stats = AttendanceOperations.get_student_attendance_stats(st.session_state.student_id)

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {student.name if student else 'Student'}")
        st.markdown(f"ID: {st.session_state.student_id}")
        st.markdown("---")

        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = 'student_dashboard'
            st.rerun()
        if st.button("Mark Attendance", use_container_width=True):
            st.session_state.page = 'mark_attendance'
            st.rerun()
        if st.button("My Profile", use_container_width=True):
            st.session_state.page = 'profile'
            st.rerun()
        if st.button("Attendance History", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in ['logged_in', 'user_role', 'student_id', 'username']:
                st.session_state[key] = None if key != 'logged_in' else False
            st.session_state.page = 'role_select'
            st.rerun()

    # Header
    st.markdown(f"""
    <div class="header-bar">
        <h2 style="margin:0;">Welcome, {student.name if student else 'Student'}</h2>
        <p style="margin:5px 0 0 0;opacity:0.8;">{datetime.now().strftime('%A, %B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["total"]}</div><div class="stat-label">Total Classes</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#28a745;">{stats["present"]}</div><div class="stat-label">Present</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#dc3545;">{stats["absent"]}</div><div class="stat-label">Absent</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#0f3460;">{stats["percentage"]}%</div><div class="stat-label">Attendance</div></div>', unsafe_allow_html=True)

    # Recent attendance
    st.markdown('<div class="section-title">Recent Attendance</div>', unsafe_allow_html=True)
    records = AttendanceOperations.get_student_attendance(st.session_state.student_id)[:5]

    if records:
        for record in records:
            status_class = "status-present" if record.status == 'Present' else "status-absent"
            time_str = record.time_in.strftime('%H:%M') if record.time_in else 'N/A'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div><strong>{record.date.strftime('%B %d, %Y')}</strong><br><span style="font-size:12px;color:#6c757d;">Time: {time_str}</span></div>
                <span class="{status_class}">{record.status}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No attendance records yet")


def show_mark_attendance():
    """Show mark attendance page"""
    st.markdown('<div class="header-bar"><h2 style="margin:0;">Mark Attendance</h2></div>', unsafe_allow_html=True)

    with st.sidebar:
        if st.button("Back to Dashboard", use_container_width=True):
            st.session_state.page = 'student_dashboard'
            st.rerun()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = 'role_select'
            st.rerun()

    already_marked = AttendanceOperations.check_attendance_exists(st.session_state.student_id)

    # Check if model exists
    model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"
    model_exists = model_path.exists()

    # Check if student face is registered
    student = StudentOperations.get_student(st.session_state.student_id)
    face_registered = student and student.face_encoding is not None

    col1, col2 = st.columns([2, 1])

    with col1:
        if already_marked:
            st.success("You have already marked attendance today!")
        elif not model_exists:
            st.error("Face recognition model not trained yet.")
            st.warning("Please contact admin to:")
            st.markdown("""
            1. Capture your face images (Admin Portal > Capture Faces)
            2. Train the recognition model (Admin Portal > Train Model)
            """)
        elif not face_registered:
            st.error("Your face is not registered in the system.")
            st.warning("Please contact admin to capture your face images first.")
        else:
            st.info("Click the button below to open camera and scan your face")
            if st.button("Open Camera & Scan", type="primary", use_container_width=True):
                run_student_recognition()

    with col2:
        status = "Present" if already_marked else "Pending"
        color = "#28a745" if already_marked else "#ffc107"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color:{color};font-size:24px;">{status}</div>
            <div class="stat-label">Today's Status</div>
        </div>
        """, unsafe_allow_html=True)

        # Show face registration status
        if not already_marked:
            face_status = "Registered" if face_registered else "Not Registered"
            face_color = "#28a745" if face_registered else "#dc3545"
            st.markdown(f"""
            <div class="stat-card" style="margin-top:15px;">
                <div class="stat-value" style="color:{face_color};font-size:18px;">{face_status}</div>
                <div class="stat-label">Face Status</div>
            </div>
            """, unsafe_allow_html=True)


def run_student_recognition():
    """Run face recognition for student"""
    model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"

    if not model_path.exists():
        st.error("Recognition model not trained. Contact admin.")
        return

    try:
        import cv2
        import numpy as np
        import pickle
        import face_recognition

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        known_encodings = [np.array(enc) for enc in data['encodings']]
        known_ids = data['ids']
        known_names = data['names']

        if st.session_state.student_id not in known_ids:
            st.error("Your face is not registered. Contact admin.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera")
            return

        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        result_placeholder = st.empty()

        status_placeholder.info("Scanning... Face the camera")

        frame_count = 0
        matched = False

        while frame_count < 100 and not matched:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame, model='hog')

            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    distances = face_recognition.face_distance(known_encodings, face_encoding)

                    if len(distances) > 0:
                        min_distance = np.min(distances)
                        best_match_idx = np.argmin(distances)

                        if min_distance < 0.5:
                            matched_id = known_ids[best_match_idx]
                            matched_name = known_names[best_match_idx]

                            if matched_id == st.session_state.student_id:
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                                cv2.putText(frame, "MATCHED!", (left, top-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                                success, msg = AttendanceOperations.mark_attendance(
                                    student_id=matched_id,
                                    confidence_score=1-min_distance,
                                    status='Present'
                                )
                                if success:
                                    matched = True
                                    result_placeholder.success(f"Attendance marked! Welcome, {matched_name}")
                            else:
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                                cv2.putText(frame, "Wrong person", (left, top-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 3)
                            cv2.putText(frame, "Unknown", (left, top-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            frame_count += 1

        cap.release()
        camera_placeholder.empty()
        status_placeholder.empty()

        if not matched:
            result_placeholder.error("Could not verify. Try again.")

    except ImportError:
        st.error("face_recognition not installed. Run: pip install face-recognition")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_student_profile():
    """Show student profile"""
    student = StudentOperations.get_student(st.session_state.student_id)

    with st.sidebar:
        if st.button("Back to Dashboard", use_container_width=True):
            st.session_state.page = 'student_dashboard'
            st.rerun()
        if st.button("Edit Profile", use_container_width=True):
            st.session_state.page = 'edit_profile'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">My Profile</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        initial = student.name[0].upper() if student else 'S'
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-avatar">{initial}</div>
            <h3>{student.name if student else 'Student'}</h3>
            <p style="color:#6c757d;">{student.student_id if student else ''}</p>
            <p style="color:#0f3460;">{student.department if student else ''}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
        if student:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Email:** {student.email or 'N/A'}")
                st.write(f"**Phone:** {student.phone or 'N/A'}")
                st.write(f"**Department:** {student.department or 'N/A'}")
            with col_b:
                st.write(f"**Batch:** {student.batch or 'N/A'}")
                st.write(f"**Semester:** {student.semester or 'N/A'}")
                st.write(f"**Section:** {student.section or 'N/A'}")
            st.write(f"**Face Registered:** {'Yes' if student.face_encoding else 'No'}")


def show_edit_profile():
    """Show edit profile page for student"""
    student = StudentOperations.get_student(st.session_state.student_id)

    with st.sidebar:
        if st.button("Back to Profile", use_container_width=True):
            st.session_state.page = 'profile'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Edit Profile</h2></div>', unsafe_allow_html=True)

    if not student:
        st.error("Student not found")
        return

    with st.form("edit_profile_form"):
        st.markdown("**Personal Information**")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name *", value=student.name or "")
            email = st.text_input("Email", value=student.email or "")
            phone = st.text_input("Phone", value=student.phone or "")

        with col2:
            department = st.selectbox("Department",
                ["", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other"],
                index=["", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other"].index(student.department) if student.department in ["", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other"] else 0
            )
            batch = st.text_input("Batch/Year", value=student.batch or "")
            semester = st.selectbox("Semester",
                ["", "1", "2", "3", "4", "5", "6", "7", "8"],
                index=["", "1", "2", "3", "4", "5", "6", "7", "8"].index(student.semester) if student.semester in ["", "1", "2", "3", "4", "5", "6", "7", "8"] else 0
            )

        section = st.text_input("Section", value=student.section or "")
        address = st.text_area("Address", value=student.address or "")

        submitted = st.form_submit_button("Save Changes", use_container_width=True)

        if submitted:
            if not name:
                st.error("Name is required")
            else:
                success, msg = StudentOperations.update_student(
                    student_id=st.session_state.student_id,
                    name=name,
                    email=email or None,
                    phone=phone or None,
                    department=department or None,
                    batch=batch or None,
                    semester=semester or None,
                    section=section or None,
                    address=address or None
                )
                if success:
                    st.success("Profile updated successfully!")
                    st.session_state.page = 'profile'
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")
    st.markdown("**Change Password**")

    with st.form("change_password_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        if st.form_submit_button("Update Password", use_container_width=True):
            if not new_password:
                st.error("Please enter a new password")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, msg = UserOperations.update_password(st.session_state.username, new_password)
                if success:
                    st.success("Password updated successfully!")
                else:
                    st.error(msg)


def show_attendance_history():
    """Show attendance history"""
    with st.sidebar:
        if st.button("Back to Dashboard", use_container_width=True):
            st.session_state.page = 'student_dashboard'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Attendance History</h2></div>', unsafe_allow_html=True)

    records = AttendanceOperations.get_student_attendance(st.session_state.student_id)

    if records:
        for record in records:
            status_class = "status-present" if record.status == 'Present' else "status-absent"
            time_in = record.time_in.strftime('%H:%M:%S') if record.time_in else 'N/A'
            time_out = record.time_out.strftime('%H:%M:%S') if record.time_out else '-'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div><strong>{record.date.strftime('%A, %B %d, %Y')}</strong><br><span style="font-size:12px;color:#6c757d;">In: {time_in} | Out: {time_out}</span></div>
                <span class="{status_class}">{record.status}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No attendance records")


def show_admin_dashboard():
    """Show admin dashboard"""
    with st.sidebar:
        st.markdown("### Admin Panel")
        st.markdown("---")
        if st.button("Dashboard", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
        if st.button("All Students", use_container_width=True):
            st.session_state.page = 'admin_students'
            st.rerun()
        if st.button("Register Student", use_container_width=True):
            st.session_state.page = 'admin_register'
            st.rerun()
        if st.button("Capture Faces", use_container_width=True):
            st.session_state.page = 'admin_capture'
            st.rerun()
        if st.button("Train Model", use_container_width=True):
            st.session_state.page = 'admin_train'
            st.rerun()
        if st.button("Mark Attendance", use_container_width=True):
            st.session_state.page = 'admin_mark'
            st.rerun()
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = 'role_select'
            st.rerun()

    st.markdown(f"""
    <div class="header-bar">
        <h2 style="margin:0;">Admin Dashboard</h2>
        <p style="margin:5px 0 0 0;opacity:0.8;">{datetime.now().strftime('%A, %B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

    total_students = StudentOperations.get_student_count()
    today_attendance = AttendanceOperations.get_today_attendance_count()
    absent = total_students - today_attendance
    rate = round((today_attendance / max(total_students, 1)) * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{total_students}</div><div class="stat-label">Total Students</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#28a745;">{today_attendance}</div><div class="stat-label">Present Today</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#dc3545;">{absent}</div><div class="stat-label">Absent Today</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#0f3460;">{rate}%</div><div class="stat-label">Attendance Rate</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Today\'s Attendance</div>', unsafe_allow_html=True)
    records = AttendanceOperations.get_daily_attendance()

    if records:
        for record in records:
            student = StudentOperations.get_student(record.student_id)
            time_str = record.time_in.strftime('%H:%M:%S') if record.time_in else 'N/A'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div><strong>{student.name if student else record.student_id}</strong><br><span style="font-size:12px;color:#6c757d;">{record.student_id}</span></div>
                <div><span style="font-size:12px;color:#6c757d;">{time_str}</span> <span class="status-present" style="margin-left:10px;">{record.status}</span></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No attendance today")


def show_admin_students():
    """Show all students"""
    with st.sidebar:
        if st.button("Back", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">All Students</h2></div>', unsafe_allow_html=True)

    students = StudentOperations.get_all_students()

    if not students:
        st.info("No students registered yet")
        return

    for student in students:
        stats = AttendanceOperations.get_student_attendance_stats(student.student_id)
        face_status = "status-present" if student.face_encoding else "status-absent"
        face_text = "Face OK" if student.face_encoding else "No Face"

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div><strong>{student.name}</strong><br><span style="font-size:12px;color:#6c757d;">{student.student_id} | {student.department or 'N/A'}</span></div>
                <div><span style="font-size:12px;color:#1a1a2e;">{stats['percentage']}%</span> <span class="{face_status}" style="margin-left:10px;">{face_text}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Edit", key=f"edit_{student.student_id}", use_container_width=True):
                st.session_state.edit_student_id = student.student_id
                st.session_state.page = 'admin_edit_student'
                st.rerun()


def show_admin_edit_student():
    """Admin edit student page"""
    with st.sidebar:
        if st.button("Back to Students", use_container_width=True):
            st.session_state.page = 'admin_students'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Edit Student</h2></div>', unsafe_allow_html=True)

    student_id = st.session_state.get('edit_student_id')
    if not student_id:
        st.error("No student selected")
        return

    student = StudentOperations.get_student(student_id)
    if not student:
        st.error("Student not found")
        return

    # Student info card
    st.markdown(f"""
    <div class="stat-card" style="text-align:left;margin-bottom:20px;">
        <strong style="color:#1a1a2e;">Student ID:</strong> <span style="color:#6c757d;">{student.student_id}</span><br>
        <strong style="color:#1a1a2e;">Face Status:</strong> <span style="color:{'#28a745' if student.face_encoding else '#dc3545'};">{'Registered' if student.face_encoding else 'Not Registered'}</span><br>
        <strong style="color:#1a1a2e;">Images:</strong> <span style="color:#6c757d;">{student.image_count or 0}</span>
    </div>
    """, unsafe_allow_html=True)

    with st.form("admin_edit_student_form"):
        st.markdown("**Personal Information**")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name *", value=student.name or "")
            email = st.text_input("Email", value=student.email or "")
            phone = st.text_input("Phone", value=student.phone or "")
            batch = st.text_input("Batch/Year", value=student.batch or "")

        with col2:
            department = st.selectbox("Department",
                ["", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other"],
                index=["", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other"].index(student.department) if student.department in ["", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other"] else 0
            )
            semester = st.selectbox("Semester",
                ["", "1", "2", "3", "4", "5", "6", "7", "8"],
                index=["", "1", "2", "3", "4", "5", "6", "7", "8"].index(student.semester) if student.semester in ["", "1", "2", "3", "4", "5", "6", "7", "8"] else 0
            )
            section = st.text_input("Section", value=student.section or "")

        address = st.text_area("Address", value=student.address or "")

        col_a, col_b = st.columns(2)
        with col_a:
            submitted = st.form_submit_button("Save Changes", use_container_width=True)
        with col_b:
            pass

        if submitted:
            if not name:
                st.error("Name is required")
            else:
                success, msg = StudentOperations.update_student(
                    student_id=student_id,
                    name=name,
                    email=email or None,
                    phone=phone or None,
                    department=department or None,
                    batch=batch or None,
                    semester=semester or None,
                    section=section or None,
                    address=address or None
                )
                if success:
                    st.success("Student updated successfully!")
                else:
                    st.error(msg)

    # Danger zone
    st.markdown("---")
    st.markdown("**Danger Zone**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Delete Face Data", use_container_width=True):
            # Clear face encoding
            success, msg = StudentOperations.update_student(student_id, face_encoding=None, image_count=0)
            if success:
                # Delete face images
                import shutil
                folder = DATASET_DIR / student_id
                if folder.exists():
                    shutil.rmtree(folder)
                st.success("Face data deleted. Student needs to re-register face.")
                st.rerun()
            else:
                st.error(msg)

    with col2:
        if st.button("Delete Student", type="primary", use_container_width=True):
            st.session_state.confirm_delete = student_id

    if st.session_state.get('confirm_delete') == student_id:
        st.warning("Are you sure you want to delete this student? This action cannot be undone.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Yes, Delete", use_container_width=True):
                success, msg = StudentOperations.delete_student(student_id, soft_delete=False)
                if success:
                    # Delete face images
                    import shutil
                    folder = DATASET_DIR / student_id
                    if folder.exists():
                        shutil.rmtree(folder)
                    st.session_state.confirm_delete = None
                    st.session_state.page = 'admin_students'
                    st.rerun()
                else:
                    st.error(msg)
        with col_b:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()


def show_admin_capture():
    """Capture faces"""
    with st.sidebar:
        if st.button("Back", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Capture Faces</h2></div>', unsafe_allow_html=True)

    students = StudentOperations.get_all_students()

    if not students:
        st.warning("No students registered. Please register students first.")
        return

    student_options = {s.student_id: f"{s.name} ({s.student_id})" for s in students}

    selected_id = st.selectbox("Select Student", list(student_options.keys()),
                               format_func=lambda x: student_options[x])

    if selected_id:
        folder = DATASET_DIR / selected_id
        existing = len(list(folder.glob('*.jpg'))) if folder.exists() else 0
        st.info(f"Current images: {existing}")

        num_images = st.slider("Images to capture", 10, 100, 50)

        if st.button("Start Capture", type="primary"):
            capture_faces(selected_id, num_images)


def capture_faces(student_id: str, num_images: int):
    """Capture faces"""
    try:
        import cv2

        folder = DATASET_DIR / student_id
        folder.mkdir(parents=True, exist_ok=True)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera")
            return

        camera_placeholder = st.empty()
        progress = st.progress(0)
        status = st.empty()

        captured = 0
        existing = len(list(folder.glob('*.jpg')))
        status.info("Capturing... Move head slowly")

        while captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_img = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (200, 200))
                img_path = folder / f"{student_id}_{existing + captured + 1:04d}.jpg"
                cv2.imwrite(str(img_path), face_resized)
                captured += 1
                progress.progress(captured / num_images)

            cv2.putText(frame, f"Captured: {captured}/{num_images}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        camera_placeholder.empty()

        total = len(list(folder.glob('*.jpg')))
        StudentOperations.update_student(student_id, image_count=total)
        st.success(f"Captured {captured} images!")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_admin_train():
    """Train model"""
    with st.sidebar:
        if st.button("Back", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Train Model</h2></div>', unsafe_allow_html=True)

    students = StudentOperations.get_all_students()
    ready = sum(1 for s in students if (DATASET_DIR / s.student_id).exists() and
                len(list((DATASET_DIR / s.student_id).glob('*.jpg'))) >= 10)

    st.info(f"Students ready: {ready}/{len(students)}")

    if ready == 0:
        st.warning("No students with enough images (need 10+)")
    else:
        if st.button("Start Training", type="primary"):
            train_model()


def train_model():
    """Train model"""
    try:
        import face_recognition
        import pickle
        import numpy as np

        TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        students = StudentOperations.get_all_students()
        encodings, ids, names = [], [], []

        progress = st.progress(0)
        status = st.empty()

        for i, student in enumerate(students):
            folder = DATASET_DIR / student.student_id
            if not folder.exists():
                continue
            images = list(folder.glob('*.jpg'))
            if len(images) < 10:
                continue

            status.info(f"Processing {student.name}...")
            student_encodings = []

            for img_path in images:
                image = face_recognition.load_image_file(str(img_path))
                face_encs = face_recognition.face_encodings(image)
                if face_encs:
                    student_encodings.append(face_encs[0])

            if student_encodings:
                avg_encoding = np.mean(student_encodings, axis=0)
                encodings.append(avg_encoding.tolist())
                ids.append(student.student_id)
                names.append(student.name)
                StudentOperations.update_face_encoding(student.student_id, avg_encoding.tolist(), len(images))

            progress.progress((i + 1) / len(students))

        model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'encodings': encodings, 'ids': ids, 'names': names}, f)

        TrainingLogOperations.create_training_log(len(ids), sum(len(list((DATASET_DIR / sid).glob('*.jpg'))) for sid in ids))
        status.empty()
        st.success(f"Training complete! {len(ids)} students trained.")

    except ImportError:
        st.error("face_recognition not installed")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_admin_mark():
    """Admin mark attendance"""
    with st.sidebar:
        if st.button("Back", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Mark Attendance</h2></div>', unsafe_allow_html=True)

    model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"
    if not model_path.exists():
        st.error("Train model first")
        return

    if st.button("Start Recognition", type="primary"):
        run_admin_recognition()


def run_admin_recognition():
    """Run admin recognition"""
    try:
        import cv2
        import face_recognition
        import pickle
        import numpy as np

        model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        known_encodings = [np.array(enc) for enc in data['encodings']]
        known_ids = data['ids']
        known_names = data['names']

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera")
            return

        camera_placeholder = st.empty()
        result_placeholder = st.empty()
        marked = set()

        st.info("Press 'q' in camera window or refresh page to stop")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame, model='hog')

            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    distances = face_recognition.face_distance(known_encodings, face_encoding)

                    if len(distances) > 0:
                        min_distance = np.min(distances)
                        best_idx = np.argmin(distances)

                        if min_distance < 0.5:
                            matched_id = known_ids[best_idx]
                            matched_name = known_names[best_idx]

                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                            cv2.putText(frame, matched_name, (left, top-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            if matched_id not in marked:
                                if not AttendanceOperations.check_attendance_exists(matched_id):
                                    success, msg = AttendanceOperations.mark_attendance(matched_id, 1-min_distance, 'Present')
                                    if success:
                                        marked.add(matched_id)
                                        result_placeholder.success(f"Marked: {matched_name}")
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                            cv2.putText(frame, "Unknown", (left, top-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()


def show_admin_register():
    """Admin register student"""
    with st.sidebar:
        if st.button("Back", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Register Student</h2></div>', unsafe_allow_html=True)

    with st.form("admin_register"):
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID *")
            name = st.text_input("Full Name *")
            email = st.text_input("Email")
            phone = st.text_input("Phone")
        with col2:
            department = st.selectbox("Department", ["", "Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other"])
            batch = st.text_input("Batch")
            semester = st.selectbox("Semester", ["", "1", "2", "3", "4", "5", "6", "7", "8"])
            section = st.text_input("Section")

        create_account = st.checkbox("Create login account")
        if create_account:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

        if st.form_submit_button("Register", use_container_width=True):
            if not student_id or not name:
                st.error("Student ID and Name are required")
            else:
                success, msg = StudentOperations.create_student(
                    student_id=student_id, name=name,
                    email=email or None, phone=phone or None,
                    department=department or None, batch=batch or None,
                    semester=semester or None, section=section or None
                )
                if success:
                    if create_account and username and password:
                        UserOperations.create_user(username, password, 'student', student_id)
                    st.success("Student registered!")
                else:
                    st.error(msg)


def main():
    """Main app"""
    init_session_state()

    if not st.session_state.logged_in:
        page = st.session_state.page

        # Role selection and login pages
        if page == 'role_select':
            show_role_selection()
        elif page == 'student_login':
            show_student_login()
        elif page == 'student_register':
            show_student_register()
        elif page == 'admin_login':
            show_admin_login()
        elif page == 'admin_register_self':
            show_admin_register_self()
        else:
            show_role_selection()
    else:
        # Logged in pages
        if st.session_state.user_role == 'admin':
            page = st.session_state.page
            if page == 'admin_students':
                show_admin_students()
            elif page == 'admin_edit_student':
                show_admin_edit_student()
            elif page == 'admin_register':
                show_admin_register()
            elif page == 'admin_capture':
                show_admin_capture()
            elif page == 'admin_train':
                show_admin_train()
            elif page == 'admin_mark':
                show_admin_mark()
            else:
                show_admin_dashboard()
        else:
            page = st.session_state.page
            if page == 'mark_attendance':
                show_mark_attendance()
            elif page == 'profile':
                show_student_profile()
            elif page == 'edit_profile':
                show_edit_profile()
            elif page == 'history':
                show_attendance_history()
            else:
                show_student_dashboard()


if __name__ == "__main__":
    main()
