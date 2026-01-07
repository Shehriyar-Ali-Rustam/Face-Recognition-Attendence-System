"""
Face Recognition Attendance System
Main Application with Separate Student/Admin Portals
Professional Orange, White & Black Theme UI
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
    page_title="AttendEase - Smart Attendance",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Orange, White & Black Theme CSS
THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        /* Primary Colors - Orange */
        --primary: #f97316;
        --primary-dark: #ea580c;
        --primary-light: #fb923c;
        --primary-50: #fff7ed;
        --primary-100: #ffedd5;

        /* Background Colors - White */
        --bg-primary: #ffffff;
        --bg-secondary: #fafafa;
        --bg-card: #ffffff;
        --bg-sidebar: #ffffff;

        /* Text Colors - Black */
        --text-primary: #171717;
        --text-secondary: #525252;
        --text-muted: #a3a3a3;
        --text-light: #fafafa;

        /* Accent & Status Colors */
        --accent: #f97316;
        --accent-hover: #ea580c;
        --success: #22c55e;
        --success-bg: rgba(34, 197, 94, 0.1);
        --warning: #f97316;
        --warning-bg: rgba(249, 115, 22, 0.1);
        --error: #ef4444;
        --error-bg: rgba(239, 68, 68, 0.1);

        /* Borders & Shadows */
        --border-color: #e5e5e5;
        --border-light: #f5f5f5;
        --hover-bg: #f5f5f5;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background: var(--bg-primary);
    }

    /* All text */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown strong,
    [data-testid="column"] p, .element-container .stMarkdown p {
        color: var(--text-primary) !important;
    }

    /* Form labels */
    .stTextInput label, .stSelectbox label, .stSlider label,
    .stCheckbox label, .stTextArea label, .stNumberInput label,
    .stFileUploader label {
        color: var(--text-secondary) !important;
        font-weight: 500;
        font-size: 13px;
        text-transform: none;
        letter-spacing: 0;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stNumberInput > div > div > input {
        background: var(--bg-secondary) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px var(--primary-100) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--text-muted) !important;
    }

    /* Select box */
    .stSelectbox > div > div,
    [data-baseweb="select"] {
        background: var(--bg-secondary) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    .stSelectbox [data-baseweb="select"] *,
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
    }

    /* Dropdown menus */
    [data-baseweb="popover"], [data-baseweb="menu"],
    [data-baseweb="listbox"], [role="listbox"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        box-shadow: var(--shadow-xl) !important;
        overflow: hidden;
    }

    [data-baseweb="popover"] li, [data-baseweb="menu"] li,
    [data-baseweb="listbox"] li, [role="option"] {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        padding: 10px 16px !important;
    }

    [data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover,
    [data-baseweb="listbox"] li:hover, [role="option"]:hover {
        background: var(--primary-50) !important;
    }

    /* Slider */
    .stSlider label, .stSlider p, [data-testid="stSlider"] * {
        color: var(--text-primary) !important;
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--primary) !important;
    }

    /* Checkbox */
    .stCheckbox > label > span {
        color: var(--text-primary) !important;
    }

    /* Sidebar - White Theme with dark text */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid #e5e5e5;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        background: transparent !important;
    }

    [data-testid="stSidebar"] * {
        color: #171717 !important;
    }

    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #171717 !important;
        font-weight: 600;
        font-size: 16px;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #525252 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #e5e5e5 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        color: #525252 !important;
        padding: 12px 16px !important;
        text-align: left !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        margin: 4px 8px !important;
        width: calc(100% - 16px) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f97316 !important;
        color: #ffffff !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary) !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: #ffffff !important;
    }

    /* Main Buttons */
    .stButton > button {
        background: var(--primary) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stButton > button:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-1px);
        box-shadow: var(--shadow-md) !important;
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Cards */
    .stat-card, .role-card, .profile-card, .attendance-row {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        box-shadow: var(--shadow);
        transition: all 0.2s ease;
    }

    .stat-card {
        padding: 24px;
        text-align: center;
    }

    .stat-card:hover {
        box-shadow: var(--shadow-md);
    }

    .stat-value {
        font-size: 36px;
        font-weight: 700;
        color: var(--text-primary) !important;
        line-height: 1.2;
    }

    .stat-label {
        font-size: 11px;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
        font-weight: 500;
    }

    .role-card {
        padding: 40px 28px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .role-card:hover {
        transform: translateY(-6px);
        box-shadow: var(--shadow-xl);
        border-color: var(--primary);
    }

    .role-icon {
        width: 72px;
        height: 72px;
        border-radius: 16px;
        background: var(--primary);
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        box-shadow: var(--shadow-md);
    }

    .role-title {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-bottom: 10px;
        letter-spacing: 0.3px;
    }

    .role-desc {
        font-size: 13px;
        color: var(--text-secondary) !important;
        line-height: 1.6;
    }

    /* Header bar */
    .header-bar {
        background: var(--primary);
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: var(--shadow-md);
    }

    .header-bar h2 {
        color: #ffffff !important;
        margin: 0;
        font-weight: 600;
    }

    .header-bar p {
        color: rgba(255,255,255,0.9) !important;
        margin: 6px 0 0 0;
    }

    /* Section title */
    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin: 24px 0 16px 0;
        padding: 14px 18px;
        background: var(--primary-50);
        border-radius: 8px;
        border-left: 4px solid var(--primary);
    }

    /* Attendance row */
    .attendance-row {
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.2s ease;
    }

    .attendance-row:hover {
        background: var(--hover-bg) !important;
    }

    .attendance-row strong { color: var(--text-primary) !important; }
    .attendance-row span { color: var(--text-secondary) !important; }

    /* Status badges */
    .status-present {
        background: var(--success-bg);
        color: var(--success) !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .status-absent {
        background: var(--error-bg);
        color: var(--error) !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Profile card */
    .profile-card {
        padding: 40px;
        text-align: center;
    }

    .profile-card h3 {
        color: var(--text-primary) !important;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .profile-card p {
        color: var(--text-secondary) !important;
    }

    .profile-avatar {
        width: 100px;
        height: 100px;
        border-radius: 20px;
        background: var(--primary);
        margin: 0 auto 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        font-weight: 700;
        color: #ffffff !important;
        box-shadow: var(--shadow-md);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: var(--primary) !important;
        border-radius: 4px;
    }

    /* File uploader */
    .stFileUploader > div,
    .stFileUploader section,
    .stFileUploader [data-testid="stFileUploader"],
    .stFileUploader [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 2px dashed #e5e5e5 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease;
    }

    .stFileUploader > div:hover,
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary) !important;
        background: #fff7ed !important;
    }

    /* File uploader text */
    .stFileUploader span,
    .stFileUploader p,
    .stFileUploader small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p {
        color: #525252 !important;
    }

    /* File uploader button */
    .stFileUploader button,
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--primary) !important;
        color: #ffffff !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }

    /* Info box styling */
    .stInfo {
        background: var(--primary-50) !important;
        color: var(--primary-dark) !important;
    }

    /* Success message */
    .stSuccess {
        background: var(--success-bg) !important;
    }

    /* Warning message */
    .stWarning {
        background: var(--warning-bg) !important;
    }

    /* Error message */
    .stError {
        background: var(--error-bg) !important;
    }

    /* Divider */
    hr {
        border-color: var(--border-color) !important;
        margin: 20px 0 !important;
    }

    /* Form styling */
    [data-testid="stForm"] {
        background: var(--bg-card);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }

    /* Login card styling */
    .login-card {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        max-width: 420px;
        margin: 0 auto;
    }

    .login-header {
        text-align: center;
        margin-bottom: 32px;
    }

    .login-header h1 {
        font-size: 24px;
        font-weight: 600;
        color: #171717;
        margin-bottom: 8px;
    }

    .login-header p {
        font-size: 14px;
        color: #525252;
    }

    .google-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        width: 100%;
        padding: 12px 24px;
        background: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        color: #171717;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 24px;
    }

    .google-btn:hover {
        background: #f5f5f5;
        border-color: #d4d4d4;
    }

    .divider {
        display: flex;
        align-items: center;
        margin: 24px 0;
    }

    .divider::before, .divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #e5e5e5;
    }

    .divider span {
        padding: 0 16px;
        font-size: 12px;
        color: #a3a3a3;
        text-transform: uppercase;
    }

    /* Quick Attendance Card */
    .quick-attendance-card {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        color: white;
        margin-bottom: 32px;
        box-shadow: 0 10px 15px -3px rgba(249, 115, 22, 0.3);
    }

    .quick-attendance-card h2 {
        color: white !important;
        font-size: 24px;
        margin-bottom: 8px;
    }

    .quick-attendance-card p {
        color: rgba(255,255,255,0.9) !important;
        margin-bottom: 20px;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, div[data-testid="stSidebarNav"] { display: none !important; }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
</style>
"""

# Apply theme CSS
st.markdown(THEME_CSS, unsafe_allow_html=True)


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
    """Show role selection page - Quick Attendance at top, Login portals below"""
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px;">
        <h1 style="font-size:42px;margin-bottom:8px;font-weight:700;letter-spacing:-1px;color:#171717;">AttendEase</h1>
        <p style="font-size:14px;color:#525252;letter-spacing:1px;">Smart Face Recognition Attendance System</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Attendance - Featured at top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="quick-attendance-card">
            <h2>Quick Attendance</h2>
            <p>Scan your face to mark attendance instantly - no login required</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Scan Face Now", key="quick_attendance_btn", use_container_width=True, type="primary"):
            st.session_state.page = 'quick_attendance'
            st.rerun()

    # Divider
    st.markdown("""
    <div style="display:flex;align-items:center;margin:40px 0;">
        <div style="flex:1;height:1px;background:#e5e5e5;"></div>
        <span style="padding:0 20px;font-size:12px;color:#a3a3a3;text-transform:uppercase;letter-spacing:2px;">or sign in to your account</span>
        <div style="flex:1;height:1px;background:#e5e5e5;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Login Portals - Side by side with equal spacing
    col1, col_a, col_b, col4 = st.columns([1, 1, 1, 1])

    with col_a:
        st.markdown("""
        <div class="role-card" style="min-height:220px;">
            <div class="role-icon">S</div>
            <div class="role-title">Student Portal</div>
            <div class="role-desc">Mark attendance and view your records</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Student Login", key="student_btn", use_container_width=True):
            st.session_state.selected_role = 'student'
            st.session_state.page = 'student_login'
            st.rerun()

    with col_b:
        st.markdown("""
        <div class="role-card" style="min-height:220px;">
            <div class="role-icon">A</div>
            <div class="role-title">Admin Portal</div>
            <div class="role-desc">Manage students and view reports</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Admin Login", key="admin_btn", use_container_width=True):
            st.session_state.selected_role = 'admin'
            st.session_state.page = 'admin_login'
            st.rerun()


def show_student_login():
    """Show student login page - Conventional style with Google option"""
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        # Login Card Header
        st.markdown("""
        <div style="text-align:center;padding:30px 0 20px;">
            <div class="role-icon" style="margin:0 auto 16px;width:56px;height:56px;font-size:22px;">S</div>
            <h1 style="font-size:24px;margin-bottom:6px;font-weight:600;color:#171717;">Student Login</h1>
            <p style="font-size:14px;color:#525252;">Sign in to access your dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        # Google Sign In Button (Visual only - placeholder)
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e5e5e5;border-radius:8px;padding:12px 24px;display:flex;align-items:center;justify-content:center;gap:12px;cursor:pointer;margin-bottom:20px;transition:all 0.2s;">
            <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
            <span style="font-size:14px;font-weight:500;color:#171717;">Continue with Google</span>
        </div>
        """, unsafe_allow_html=True)

        # Divider
        st.markdown("""
        <div style="display:flex;align-items:center;margin:20px 0;">
            <div style="flex:1;height:1px;background:#e5e5e5;"></div>
            <span style="padding:0 16px;font-size:12px;color:#a3a3a3;">or continue with email</span>
            <div style="flex:1;height:1px;background:#e5e5e5;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Login Form
        username = st.text_input("Email or Username", placeholder="Enter your email or username", key="student_username")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="student_password")

        if st.button("Sign In", use_container_width=True, key="student_login_btn", type="primary"):
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

        # Register Link
        st.markdown("""
        <p style="text-align:center;margin-top:24px;font-size:14px;color:#525252;">
            Don't have an account?
        </p>
        """, unsafe_allow_html=True)

        if st.button("Create Account", use_container_width=True, key="student_register_btn"):
            st.session_state.page = 'student_register'
            st.rerun()

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        if st.button("Back to Home", use_container_width=True, key="student_back_btn"):
            st.session_state.page = 'role_select'
            st.rerun()


def show_student_register():
    """Show student registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:30px 0;">
            <div class="role-icon" style="margin:0 auto 20px;">R</div>
            <h1 style="font-size:28px;margin-bottom:10px;font-weight:600;">Student Registration</h1>
            <p style="font-size:14px;opacity:0.6;">Create your account</p>
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
                dept_options = ["", "Computer Science", "Software Engineering", "Electrical Engineering",
                    "Mechanical Engineering", "Civil Engineering", "Other (Custom)"]
                department = st.selectbox("Department", dept_options)
                if department == "Other (Custom)":
                    department = st.text_input("Enter Department Name")
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
    """Show admin login page - Conventional style with Google option"""
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        # Login Card Header
        st.markdown("""
        <div style="text-align:center;padding:30px 0 20px;">
            <div class="role-icon" style="margin:0 auto 16px;width:56px;height:56px;font-size:22px;">A</div>
            <h1 style="font-size:24px;margin-bottom:6px;font-weight:600;color:#171717;">Admin Login</h1>
            <p style="font-size:14px;color:#525252;">Sign in to manage the system</p>
        </div>
        """, unsafe_allow_html=True)

        # Google Sign In Button (Visual only - placeholder)
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e5e5e5;border-radius:8px;padding:12px 24px;display:flex;align-items:center;justify-content:center;gap:12px;cursor:pointer;margin-bottom:20px;transition:all 0.2s;">
            <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
            <span style="font-size:14px;font-weight:500;color:#171717;">Continue with Google</span>
        </div>
        """, unsafe_allow_html=True)

        # Divider
        st.markdown("""
        <div style="display:flex;align-items:center;margin:20px 0;">
            <div style="flex:1;height:1px;background:#e5e5e5;"></div>
            <span style="padding:0 16px;font-size:12px;color:#a3a3a3;">or continue with email</span>
            <div style="flex:1;height:1px;background:#e5e5e5;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Login Form
        username = st.text_input("Email or Username", placeholder="Enter your email or username", key="admin_username")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="admin_password")

        if st.button("Sign In", use_container_width=True, key="admin_login_btn", type="primary"):
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

        # Default credentials info
        st.markdown("""
        <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:12px 16px;margin-top:16px;">
            <p style="font-size:13px;color:#9a3412;margin:0;"><strong>Default credentials:</strong> admin / admin123</p>
        </div>
        """, unsafe_allow_html=True)

        # Register Link
        st.markdown("""
        <p style="text-align:center;margin-top:24px;font-size:14px;color:#525252;">
            Need an admin account?
        </p>
        """, unsafe_allow_html=True)

        if st.button("Register New Admin", use_container_width=True, key="admin_register_nav_btn"):
            st.session_state.page = 'admin_register_self'
            st.rerun()

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        if st.button("Back to Home", use_container_width=True, key="admin_back_btn"):
            st.session_state.page = 'role_select'
            st.rerun()


def show_admin_register_self():
    """Show admin self-registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:30px 0;">
            <div class="role-icon" style="margin:0 auto 20px;">R</div>
            <h1 style="font-size:28px;margin-bottom:10px;font-weight:600;">Admin Registration</h1>
            <p style="font-size:14px;opacity:0.6;">Create new admin account</p>
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
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#22c55e;">{stats["present"]}</div><div class="stat-label">Present</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#ef4444;">{stats["absent"]}</div><div class="stat-label">Absent</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#f97316;">{stats["percentage"]}%</div><div class="stat-label">Attendance</div></div>', unsafe_allow_html=True)

    # Recent attendance
    st.markdown('<div class="section-title">Recent Attendance</div>', unsafe_allow_html=True)
    records = AttendanceOperations.get_student_attendance(st.session_state.student_id)[:5]

    if records:
        for record in records:
            status_class = "status-present" if record.status == 'Present' else "status-absent"
            time_str = record.time_in.strftime('%H:%M') if record.time_in else 'N/A'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div><strong>{record.date.strftime('%B %d, %Y')}</strong><br><span style="font-size:12px;color:#64748b;">Time: {time_str}</span></div>
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
        color = "#22c55e" if already_marked else "#f97316"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color:{color};font-size:24px;">{status}</div>
            <div class="stat-label">Today's Status</div>
        </div>
        """, unsafe_allow_html=True)

        # Show face registration status
        if not already_marked:
            face_status = "Registered" if face_registered else "Not Registered"
            face_color = "#22c55e" if face_registered else "#ef4444"
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
            <p style="color:#64748b;">{student.student_id if student else ''}</p>
            <p style="color:#f97316;">{student.department if student else ''}</p>
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
            dept_options = ["", "Computer Science", "Software Engineering", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other (Custom)"]
            current_dept = student.department or ""
            if current_dept in dept_options:
                dept_idx = dept_options.index(current_dept)
            else:
                dept_idx = dept_options.index("Other (Custom)")
            department = st.selectbox("Department", dept_options, index=dept_idx, key="edit_dept")
            if department == "Other (Custom)":
                department = st.text_input("Enter Department Name", value=current_dept if current_dept not in dept_options else "")
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
                <div><strong>{record.date.strftime('%A, %B %d, %Y')}</strong><br><span style="font-size:12px;color:#64748b;">In: {time_in} | Out: {time_out}</span></div>
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
        if st.button("Add Face Images", use_container_width=True):
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
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#22c55e;">{today_attendance}</div><div class="stat-label">Present Today</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#ef4444;">{absent}</div><div class="stat-label">Absent Today</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#f97316;">{rate}%</div><div class="stat-label">Attendance Rate</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Today\'s Attendance</div>', unsafe_allow_html=True)
    records = AttendanceOperations.get_daily_attendance()

    if records:
        for record in records:
            student = StudentOperations.get_student(record.student_id)
            time_str = record.time_in.strftime('%H:%M:%S') if record.time_in else 'N/A'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div><strong>{student.name if student else record.student_id}</strong><br><span style="font-size:12px;color:#64748b;">{record.student_id}</span></div>
                <div><span style="font-size:12px;color:#64748b;">{time_str}</span> <span class="status-present" style="margin-left:10px;">{record.status}</span></div>
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
                <div><strong>{student.name}</strong><br><span style="font-size:12px;color:#64748b;">{student.student_id} | {student.department or 'N/A'}</span></div>
                <div><span style="font-size:12px;color:#f97316;font-weight:600;">{stats['percentage']}%</span> <span class="{face_status}" style="margin-left:10px;">{face_text}</span></div>
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
        <strong style="color:#1e293b;">Student ID:</strong> <span style="color:#64748b;">{student.student_id}</span><br>
        <strong style="color:#171717;">Face Status:</strong> <span style="color:{'#22c55e' if student.face_encoding else '#ef4444'};">{'Registered' if student.face_encoding else 'Not Registered'}</span><br>
        <strong style="color:#1e293b;">Images:</strong> <span style="color:#64748b;">{student.image_count or 0}</span>
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
            dept_options = ["", "Computer Science", "Software Engineering", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other (Custom)"]
            current_dept = student.department or ""
            if current_dept in dept_options:
                dept_idx = dept_options.index(current_dept)
            else:
                dept_idx = dept_options.index("Other (Custom)")
            department = st.selectbox("Department", dept_options, index=dept_idx, key="admin_edit_dept")
            if department == "Other (Custom)":
                department = st.text_input("Enter Department Name", value=current_dept if current_dept not in dept_options else "", key="admin_custom_dept")
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
    """Capture faces and upload photos - combined page"""
    with st.sidebar:
        if st.button("Back", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
            st.rerun()

    st.markdown('<div class="header-bar"><h2 style="margin:0;">Add Face Images</h2></div>', unsafe_allow_html=True)

    students = StudentOperations.get_all_students()

    if not students:
        st.warning("No students registered. Please register students first.")
        return

    student_options = {s.student_id: f"{s.name} ({s.student_id})" for s in students}

    selected_id = st.selectbox("Select Student", list(student_options.keys()),
                               format_func=lambda x: student_options[x])

    if selected_id:
        student = StudentOperations.get_student(selected_id)
        folder = DATASET_DIR / selected_id
        existing = len(list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg'))) if folder.exists() else 0

        # Student info card
        st.markdown(f"""
        <div class="stat-card" style="text-align:left;margin:15px 0;">
            <strong style="color:#1e293b;">{student.name}</strong><br>
            <span style="color:#64748b;font-size:14px;">Current images: {existing} | Min required: 5</span>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for different methods
        tab1, tab2 = st.tabs(["Camera Capture", "Upload Photos"])

        with tab1:
            st.markdown("**Capture faces using camera**")
            st.markdown("<span style='color:#666;font-size:13px;'>Move your head slowly while capturing for better accuracy</span>", unsafe_allow_html=True)

            num_images = st.slider("Images to capture", 10, 100, 50)

            if st.button("Start Camera Capture", type="primary", use_container_width=True):
                capture_faces(selected_id, num_images)

        with tab2:
            st.markdown("**Upload face photos**")
            st.markdown("""
            <span style="color:#666;font-size:13px;">
            Tips for best results:<br>
            • Upload 10-20 clear face photos from different angles<br>
            • Good lighting, no heavy shadows<br>
            • Face should be clearly visible and centered<br>
            • Avoid blurry or dark photos
            </span>
            """, unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Choose photos (JPG, PNG)",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="photo_uploader_combined"
            )

            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} photos selected**")

                # Preview uploaded images
                cols = st.columns(5)
                for i, file in enumerate(uploaded_files[:10]):
                    with cols[i % 5]:
                        st.image(file, use_container_width=True)

                if len(uploaded_files) > 10:
                    st.caption(f"...and {len(uploaded_files) - 10} more")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Photos Only", use_container_width=True):
                        process_uploaded_photos(selected_id, student.name, uploaded_files, train_after=False)
                with col2:
                    if st.button("Save & Train Model", type="primary", use_container_width=True):
                        process_uploaded_photos(selected_id, student.name, uploaded_files, train_after=True)


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


def process_uploaded_photos(student_id: str, student_name: str, uploaded_files, train_after: bool = False):
    """Process and save uploaded photos, optionally train model"""
    import cv2
    import numpy as np
    from PIL import Image
    import io

    try:
        folder = DATASET_DIR / student_id
        folder.mkdir(parents=True, exist_ok=True)

        existing_count = len(list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg')))

        progress = st.progress(0)
        status = st.empty()

        saved_count = 0
        rejected_count = 0
        saved_images = []

        for i, uploaded_file in enumerate(uploaded_files):
            status.info(f"Processing {uploaded_file.name}...")

            # Read image
            image_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert to numpy array (BGR for OpenCV)
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Import face recognition for validation
            import face_recognition
            from utils.face_recognizer import FaceQualityValidator

            validator = FaceQualityValidator()

            # Check image quality
            is_quality_ok, quality_report = validator.validate_face_image(image)

            if not is_quality_ok:
                rejected_count += 1
                continue

            # Check for face
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image, model='hog')

            if not face_locations:
                rejected_count += 1
                continue

            # Check face size
            size_ok, _ = validator.check_face_size(face_locations[0])
            if not size_ok:
                rejected_count += 1
                continue

            # Crop and resize face
            top, right, bottom, left = face_locations[0]
            # Add some padding
            padding = 30
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(image.shape[0], bottom + padding)
            right = min(image.shape[1], right + padding)

            face_img = image[top:bottom, left:right]
            face_resized = cv2.resize(face_img, (200, 200))

            # Save image
            img_path = folder / f"{student_id}_{existing_count + saved_count + 1:04d}.jpg"
            cv2.imwrite(str(img_path), face_resized)
            saved_count += 1
            saved_images.append(image)

            progress.progress((i + 1) / len(uploaded_files))

        status.empty()
        progress.empty()

        # Update student image count
        total_images = len(list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg')))
        StudentOperations.update_student(student_id, image_count=total_images)

        if saved_count > 0:
            st.success(f"Saved {saved_count} quality photos! ({rejected_count} rejected for quality issues)")

            if train_after:
                st.info("Training model with new photos...")

                # Use the enhanced training
                from utils.face_recognizer import FaceRecognizer
                recognizer = FaceRecognizer()

                # Prepare training data for all students with enough images
                students = StudentOperations.get_all_students()
                training_data = []

                for student in students:
                    images_path = DATASET_DIR / student.student_id
                    if images_path.exists():
                        img_count = len(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + list(images_path.glob('*.jpeg')))
                        if img_count >= 5:  # Minimum 5 images
                            training_data.append({
                                'student_id': student.student_id,
                                'name': student.name,
                                'images_path': images_path
                            })

                if training_data:
                    success, msg = recognizer.train_model(training_data)
                    if success:
                        st.success(f"Model trained successfully! {msg}")
                        # Update face encoding status
                        StudentOperations.update_face_encoding(student_id, [1], total_images)  # Placeholder encoding
                    else:
                        st.error(f"Training failed: {msg}")
                else:
                    st.warning("Not enough images for training. Need at least 5 quality photos per student.")
        else:
            st.error(f"No valid face photos found. All {rejected_count} photos were rejected for quality issues (blur, no face detected, or face too small).")

    except Exception as e:
        st.error(f"Error processing photos: {str(e)}")


def show_quick_attendance():
    """Quick attendance marking without login"""
    st.markdown("""
    <div style="text-align:center;padding:40px 0 30px;">
        <div class="role-icon" style="margin:0 auto 20px;">Q</div>
        <h1 style="font-size:28px;margin-bottom:10px;font-weight:600;">Quick Attendance</h1>
        <p style="font-size:14px;opacity:0.6;">Face scan to mark your attendance</p>
    </div>
    """, unsafe_allow_html=True)

    model_path = TRAINED_MODELS_DIR / "face_encodings.pkl"
    if not model_path.exists():
        st.error("Face recognition model not trained yet. Please contact admin.")
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = 'role_select'
            st.rerun()
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Face Scan", type="primary", use_container_width=True):
            run_quick_attendance()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = 'role_select'
            st.rerun()


def run_quick_attendance():
    """Run quick attendance recognition"""
    try:
        import cv2
        import face_recognition
        import pickle
        import numpy as np
        import time

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
        status_placeholder = st.empty()

        status_placeholder.info("Looking for your face... Please look at the camera.")

        attendance_marked = False
        start_time = time.time()
        timeout = 30  # 30 seconds timeout

        while not attendance_marked and (time.time() - start_time) < timeout:
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

                            # Check if student exists
                            student = StudentOperations.get_student(matched_id)
                            if student:
                                # Check if already marked today
                                if AttendanceOperations.check_attendance_exists(matched_id):
                                    status_placeholder.warning(f"Attendance already marked for {matched_name} today!")
                                else:
                                    success, msg = AttendanceOperations.mark_attendance(matched_id, 1-min_distance, 'Present')
                                    if success:
                                        status_placeholder.success(f"Attendance marked successfully for {matched_name}!")
                                    else:
                                        status_placeholder.error(f"Failed to mark attendance: {msg}")
                                attendance_marked = True
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                            cv2.putText(frame, "Unknown", (left, top-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

        if not attendance_marked:
            status_placeholder.error("Face not recognized. Please ensure you are registered in the system.")

        # Show back button after completion
        time.sleep(2)
        st.info("Redirecting to home page...")
        time.sleep(1)
        st.session_state.page = 'role_select'
        st.rerun()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        if st.button("Back to Home"):
            st.session_state.page = 'role_select'
            st.rerun()


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
            dept_options = ["", "Computer Science", "Software Engineering", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Other (Custom)"]
            department = st.selectbox("Department", dept_options, key="admin_reg_dept")
            if department == "Other (Custom)":
                department = st.text_input("Enter Department Name", key="admin_reg_custom_dept")
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
        elif page == 'quick_attendance':
            show_quick_attendance()
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
