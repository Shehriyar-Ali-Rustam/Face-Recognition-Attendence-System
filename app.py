"""
Face Recognition Attendance System
Main Application with Separate Student/Admin Portals
Premium Dark Glassmorphism UI with Unsplash Backgrounds
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
    initial_sidebar_state="auto"
)

# Professional Dark Theme CSS — Linear/Vercel inspired
THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');
    .material-symbols-outlined { font-family: 'Material Symbols Outlined' !important; font-weight: normal; font-style: normal; font-size: 24px; line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; word-wrap: normal; direction: ltr; -webkit-font-smoothing: antialiased; }

    :root {
        --primary: #70E6ED;
        --primary-dark: #4dd4db;
        --primary-light: #a8f0f4;
        --accent-lime: #CAF291;

        --bg-app: #0d0d0d;
        --bg-surface: #111111;
        --bg-card: #181818;
        --bg-card-hover: #1e1e1e;
        --bg-input: #1a1a1a;
        --bg-sidebar: #111111;

        --text-primary: #ededed;
        --text-secondary: #888888;
        --text-muted: #555555;
        --text-bright: #ffffff;

        --success: #4ade80;
        --success-bg: rgba(74, 222, 128, 0.10);
        --warning: #fbbf24;
        --warning-bg: rgba(251, 191, 36, 0.10);
        --error: #f87171;
        --error-bg: rgba(248, 113, 113, 0.10);

        --border: #222222;
        --border-light: #2a2a2a;
        --border-focus: #70E6ED;

        --shadow: 0 1px 3px rgba(0,0,0,0.4);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.5);
        --shadow-lg: 0 12px 40px rgba(0,0,0,0.6);

        --radius: 12px;
        --radius-sm: 8px;
        --radius-lg: 16px;
    }

    /* ===== GLOBAL ===== */
    body, p, span, div, label, input, textarea, select, button, h1, h2, h3, h4, h5, li, a {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    /* Preserve icon fonts — don't override Streamlit's icon buttons */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    .stApp { background: var(--bg-app) !important; }

    /* ===== TEXT ===== */
    .stApp p, .stApp label, .stApp li,
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown strong,
    [data-testid="column"] p, .element-container .stMarkdown p { color: var(--text-primary) !important; }
    .stMarkdown strong { color: var(--text-bright) !important; }
    h1, h2, h3, h4 { color: var(--text-bright) !important; font-weight: 700; }

    /* ===== BUTTON TEXT — override broad color rules ===== */
    .stButton > button, .stButton > button p, .stButton > button span,
    .stButton > button div, .stButton > button * {
        color: #d0d0d0 !important;
    }
    .stButton > button[kind="primary"], .stButton > button[kind="primary"] *,
    .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span {
        color: #0a0a0a !important;
    }

    /* ===== INPUTS ===== */
    .stTextInput label, .stSelectbox label, .stSlider label,
    .stCheckbox label, .stTextArea label, .stNumberInput label, .stFileUploader label {
        color: var(--text-secondary) !important; font-size: 13px !important; font-weight: 500 !important;
    }
    .stTextInput > div > div > input, .stTextArea textarea, .stNumberInput > div > div > input {
        background: var(--bg-input) !important; border: 1.5px solid var(--border) !important;
        border-radius: var(--radius-sm) !important; color: var(--text-bright) !important;
        padding: 11px 14px !important; font-size: 14px !important; transition: border-color 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea textarea:focus {
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 3px rgba(112,230,237,0.12) !important;
    }
    .stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

    /* ===== SELECT ===== */
    .stSelectbox > div > div, [data-baseweb="select"] {
        background: var(--bg-input) !important; border: 1.5px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }
    .stSelectbox [data-baseweb="select"] *, .stSelectbox > div > div > div { color: var(--text-primary) !important; }
    [data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="listbox"], [role="listbox"] {
        background: #1a1a1a !important; border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-sm) !important; box-shadow: var(--shadow-lg) !important;
    }
    [data-baseweb="popover"] li, [data-baseweb="menu"] li,
    [data-baseweb="listbox"] li, [role="option"] {
        background: transparent !important; color: var(--text-primary) !important;
        padding: 10px 16px !important; transition: background 0.15s ease !important;
    }
    [data-baseweb="popover"] li:hover, [role="option"]:hover {
        background: rgba(112,230,237,0.08) !important; color: var(--primary) !important;
    }

    /* ===== SLIDER / CHECKBOX ===== */
    .stSlider label, .stSlider p, [data-testid="stSlider"] * { color: var(--text-primary) !important; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--primary) !important; }
    .stCheckbox > label > span { color: var(--text-primary) !important; }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; background: transparent !important; }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    [data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary) !important; }
    [data-testid="stSidebar"] hr { border-color: var(--border) !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important; border: none !important;
        border-radius: var(--radius-sm) !important; color: #888888 !important;
        padding: 10px 14px !important; text-align: left !important;
        font-weight: 500 !important; font-size: 14px !important;
        transition: background 0.2s ease, color 0.2s ease !important; width: 100% !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1a1a1a !important; color: #ededed !important;
        transform: none !important; border: none !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card); border-radius: var(--radius-sm);
        padding: 4px; gap: 2px; border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: var(--text-secondary) !important;
        border-radius: 6px; padding: 8px 18px; font-weight: 500; font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important; color: #0d0d0d !important; font-weight: 600;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: #1e1e1e !important; color: #d0d0d0 !important;
        border: 1px solid #3a3a3a !important; border-radius: var(--radius-sm) !important;
        padding: 9px 20px !important; font-weight: 500 !important; font-size: 14px !important;
        transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease !important;
        letter-spacing: 0.1px !important; box-shadow: none !important;
    }
    .stButton > button:hover {
        background: #2a2a2a !important; border-color: #555 !important;
        color: #ffffff !important; transform: none !important; box-shadow: none !important;
    }
    .stButton > button:active { opacity: 0.75 !important; }
    /* Primary button — white, solid, high contrast */
    .stButton > button[kind="primary"] {
        background: #ffffff !important; color: #0a0a0a !important;
        border: 1px solid #ffffff !important; font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #e8e8e8 !important; border-color: #e8e8e8 !important;
        color: #0a0a0a !important;
    }

    /* ===== STAT CARDS ===== */
    .stat-card {
        background: var(--bg-card) !important; border: 1px solid var(--border);
        border-radius: var(--radius); padding: 24px; text-align: center;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover { border-color: var(--border-light); box-shadow: var(--shadow-md); }
    .stat-value {
        font-size: 36px; font-weight: 800; color: var(--text-bright) !important;
        line-height: 1.1; letter-spacing: -1px; font-family: 'JetBrains Mono', monospace !important;
    }
    .stat-label {
        font-size: 11px; color: var(--text-muted) !important; text-transform: uppercase;
        letter-spacing: 1.2px; margin-top: 8px; font-weight: 500;
    }

    /* ===== ROLE CARDS ===== */
    .role-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 40px 28px; text-align: center;
        transition: all 0.3s ease; cursor: pointer;
    }
    .role-card:hover {
        border-color: var(--primary); box-shadow: 0 8px 32px rgba(112,230,237,0.12);
        transform: translateY(-4px);
    }
    .role-icon {
        width: 64px; height: 64px; border-radius: 16px;
        background: var(--primary); margin: 0 auto 20px;
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; font-weight: 800; color: #0d0d0d !important;
    }
    .role-title { font-size: 18px; font-weight: 700; color: var(--text-bright) !important; margin-bottom: 8px; }
    .role-desc { font-size: 13px; color: var(--text-secondary) !important; line-height: 1.5; }

    /* ===== HEADER BAR ===== */
    .header-bar {
        background: var(--bg-card); border: 1px solid var(--border);
        border-left: 3px solid var(--primary);
        padding: 20px 24px; border-radius: var(--radius); margin-bottom: 24px;
    }
    .header-bar h2 { color: var(--text-bright) !important; margin: 0; font-size: 20px; font-weight: 700; }
    .header-bar p { color: var(--text-secondary) !important; margin: 4px 0 0; font-size: 13px; }

    /* ===== SECTION TITLE ===== */
    .section-title {
        font-size: 12px; font-weight: 600; color: var(--text-secondary) !important;
        margin: 24px 0 12px; padding: 10px 16px;
        background: var(--bg-card); border-radius: var(--radius-sm);
        border-left: 2px solid var(--primary);
        text-transform: uppercase; letter-spacing: 1px;
    }

    /* ===== ATTENDANCE ROWS ===== */
    .attendance-row {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: var(--radius-sm); padding: 14px 18px; margin: 6px 0;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .attendance-row:hover { background: var(--bg-card-hover) !important; border-color: var(--border-light); }
    .attendance-row strong { color: var(--text-bright) !important; }
    .attendance-row span { color: var(--text-secondary) !important; }

    /* ===== STATUS BADGES ===== */
    .status-present {
        background: var(--success-bg); color: var(--success) !important;
        padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        border: 1px solid rgba(74,222,128,0.2); display: inline-block;
    }
    .status-absent {
        background: var(--error-bg); color: var(--error) !important;
        padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        border: 1px solid rgba(248,113,113,0.2); display: inline-block;
    }

    /* ===== PROFILE CARD ===== */
    .profile-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 40px; text-align: center;
    }
    .profile-card h3 { color: var(--text-bright) !important; font-weight: 700; }
    .profile-card p { color: var(--text-secondary) !important; }
    .profile-avatar {
        width: 88px; height: 88px; border-radius: 20px;
        background: var(--primary); margin: 0 auto 20px;
        display: flex; align-items: center; justify-content: center;
        font-size: 32px; font-weight: 800; color: #0d0d0d !important;
    }

    /* ===== MISC ===== */
    .stProgress > div > div { background: var(--primary) !important; border-radius: 4px; }

    .stFileUploader > div, [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-card) !important; border: 1.5px dashed var(--border-light) !important;
        border-radius: var(--radius) !important;
    }
    .stFileUploader > div:hover, [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary) !important;
    }
    .stFileUploader span, .stFileUploader p, .stFileUploader small,
    [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] p {
        color: var(--text-secondary) !important;
    }
    .stFileUploader button, [data-testid="stFileUploaderDropzone"] button {
        background: var(--primary) !important; color: #0d0d0d !important;
    }

    .stAlert { border-radius: var(--radius-sm) !important; }
    [data-testid="stForm"] {
        background: var(--bg-card); padding: 24px; border-radius: var(--radius);
        border: 1px solid var(--border);
    }

    hr { border-color: var(--border) !important; margin: 20px 0 !important; }

    /* ===== QUICK ATTENDANCE CARD ===== */
    .quick-attendance-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-top: 2px solid var(--primary);
        border-radius: var(--radius-lg); padding: 36px; text-align: center; margin-bottom: 28px;
    }
    .quick-attendance-card h2 { color: var(--text-bright) !important; font-size: 22px; margin-bottom: 8px; }
    .quick-attendance-card p { color: var(--text-secondary) !important; margin-bottom: 20px; font-size: 14px; }

    /* ===== FORGOT PW ===== */
    .forgot-pw-link button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        color: #666666 !important; font-size: 13px !important; padding: 2px 0 !important;
        text-decoration: underline !important; text-underline-offset: 3px !important;
        min-height: unset !important;
    }
    .forgot-pw-link button:hover { color: #aaaaaa !important; border: none !important; background: transparent !important; transform: none !important; box-shadow: none !important; }
    .forgot-pw-link { text-align: right; margin: -6px 0 14px; }

    /* ===== INFO BOX ===== */
    .info-box {
        background: rgba(112,230,237,0.06); border: 1px solid rgba(112,230,237,0.15);
        border-radius: var(--radius-sm); padding: 14px 18px;
    }
    .info-box p { color: var(--primary) !important; font-size: 13px; margin: 0; }

    /* ===== HEADER / STREAMLIT CHROME ===== */
    #MainMenu, footer, div[data-testid="stSidebarNav"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .stDeployButton, [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-app); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #333; }

    /* ===== SPLIT LOGIN PANEL ===== */
    .split-image-panel {
        border-radius: var(--radius-lg); overflow: hidden; min-height: 560px;
        background-size: cover; background-position: center;
        display: flex; flex-direction: column; justify-content: flex-end; padding: 36px;
        position: relative;
    }
    .split-image-panel::after {
        content: ''; position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(13,13,13,0.1) 0%, rgba(13,13,13,0.85) 100%);
        border-radius: var(--radius-lg);
    }
    .split-image-panel .panel-content { position: relative; z-index: 1; }

    /* ===== HOME HERO ===== */
    .hero-image-card {
        border-radius: var(--radius-lg); overflow: hidden; min-height: 380px;
        background-size: cover; background-position: center;
        display: flex; flex-direction: column; justify-content: flex-end;
        padding: 40px; position: relative; margin-bottom: 32px;
    }
    .hero-image-card::after {
        content: ''; position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(13,13,13,0.0) 30%, rgba(13,13,13,0.92) 100%);
    }
    .hero-image-card .hero-content { position: relative; z-index: 1; }

    /* ===== BRAND BADGE ===== */
    .brand-badge {
        display: inline-flex; align-items: center; gap: 8px;
        border: 1px solid var(--border-light); background: var(--bg-card);
        padding: 6px 14px; border-radius: 100px; margin-bottom: 16px;
    }
    .brand-badge span { font-size: 12px; color: var(--primary) !important; font-weight: 600; letter-spacing: 0.5px; }
    .brand-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); flex-shrink: 0; }
</style>
"""

# Apply theme CSS
st.markdown(THEME_CSS, unsafe_allow_html=True)

# Load hero image as base64
import base64 as _b64
_hero_path = BASE_DIR / "HERO IMAGE.webp"
_HERO_B64 = ""
if _hero_path.exists():
    with open(_hero_path, "rb") as _f:
        _HERO_B64 = _b64.b64encode(_f.read()).decode()


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
    """Show role selection page — full hero background with glass UI"""

    # Full-viewport hero with the face recognition image
    st.markdown(f"""
    <style>
        /* ===== HERO PAGE BACKGROUND ===== */
        .hero-bg-wrapper {{
            position: fixed;
            inset: 0;
            z-index: 0;
            background-image: url('data:image/webp;base64,{_HERO_B64}');
            background-size: cover;
            background-position: center 20%;
            background-repeat: no-repeat;
        }}
        .hero-bg-wrapper::after {{
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(
                135deg,
                rgba(5, 10, 25, 0.82) 0%,
                rgba(5, 10, 25, 0.60) 50%,
                rgba(5, 10, 25, 0.80) 100%
            );
        }}
        /* ===== GLASS CARDS ===== */
        .glass-card {{
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 20px;
            padding: 32px 28px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        .glass-card:hover {{
            background: rgba(255,255,255,0.10);
            border-color: rgba(112,230,237,0.35);
            box-shadow: 0 8px 40px rgba(112,230,237,0.12);
            transform: translateY(-4px);
        }}
        .glass-quick {{
            background: rgba(112,230,237,0.08);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(112,230,237,0.25);
            border-radius: 20px;
            padding: 36px 32px;
            text-align: center;
        }}
        .glass-divider {{
            display: flex; align-items: center;
            margin: 36px auto; max-width: 560px;
        }}
        .glass-divider-line {{ flex:1; height:1px; background:rgba(255,255,255,0.10); }}
        .glass-divider-text {{ padding:0 20px; font-size:11px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:2.5px; }}
        /* Ensure content sits above hero bg */
        .block-container {{ position: relative; z-index: 1; }}
        [data-testid="stVerticalBlock"] {{ position: relative; z-index: 1; }}
    </style>
    <div class="hero-bg-wrapper"></div>
    """, unsafe_allow_html=True)

    # Hero headline
    st.markdown("""
    <div style="text-align:center;padding:72px 0 36px;position:relative;z-index:1;">
        <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(112,230,237,0.10);border:1px solid rgba(112,230,237,0.25);padding:6px 16px;border-radius:100px;margin-bottom:22px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#70E6ED;"></div>
            <span style="font-size:12px;color:#70E6ED;font-weight:600;letter-spacing:1px;text-transform:uppercase;">AI-Powered Attendance</span>
        </div>
        <h1 style="font-size:48px;font-weight:900;color:#ffffff;margin:0 0 14px;letter-spacing:-1.5px;line-height:1.1;">AttendEase</h1>
        <p style="font-size:16px;color:rgba(255,255,255,0.5);margin:0;font-weight:400;">Smart Face Recognition · Instant Verification · Zero Friction</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Attendance glass card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="glass-quick">
            <div style="font-size:36px;margin-bottom:12px;">⚡</div>
            <h2 style="font-size:20px;font-weight:700;color:#fff;margin:0 0 8px;">Quick Attendance</h2>
            <p style="font-size:14px;color:rgba(255,255,255,0.5);margin:0 0 20px;line-height:1.6;">Scan your face to mark attendance instantly — no login required</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡  Scan Face Now", key="quick_attendance_btn", use_container_width=True, type="primary"):
            st.session_state.page = 'quick_attendance'
            st.rerun()

    # Divider
    st.markdown("""
    <div class="glass-divider">
        <div class="glass-divider-line"></div>
        <span class="glass-divider-text">or sign in as</span>
        <div class="glass-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # Login Portals — glass cards
    col1, col_a, col_b, col4 = st.columns([0.5, 1, 1, 0.5])

    with col_a:
        st.markdown("""
        <div class="glass-card" style="min-height:200px;">
            <div style="width:56px;height:56px;border-radius:14px;background:rgba(112,230,237,0.15);border:1px solid rgba(112,230,237,0.3);margin:0 auto 18px;display:flex;align-items:center;justify-content:center;font-size:24px;">🎓</div>
            <div style="font-size:17px;font-weight:700;color:#fff;margin-bottom:8px;">Student Portal</div>
            <div style="font-size:13px;color:rgba(255,255,255,0.45);line-height:1.5;">Mark attendance, view records, and manage your profile</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Student Login", key="student_btn", use_container_width=True):
            st.session_state.selected_role = 'student'
            st.session_state.page = 'student_login'
            st.rerun()

    with col_b:
        st.markdown("""
        <div class="glass-card" style="min-height:200px;border-color:rgba(202,242,145,0.2);">
            <div style="width:56px;height:56px;border-radius:14px;background:rgba(202,242,145,0.12);border:1px solid rgba(202,242,145,0.3);margin:0 auto 18px;display:flex;align-items:center;justify-content:center;font-size:24px;">🛡️</div>
            <div style="font-size:17px;font-weight:700;color:#fff;margin-bottom:8px;">Admin Portal</div>
            <div style="font-size:13px;color:rgba(255,255,255,0.45);line-height:1.5;">Manage students, train models, and generate reports</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Admin Login", key="admin_btn", use_container_width=True):
            st.session_state.selected_role = 'admin'
            st.session_state.page = 'admin_login'
            st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:40px 0 24px;">
        <p style="font-size:12px;color:rgba(255,255,255,0.2);letter-spacing:1.5px;text-transform:uppercase;">Powered by Face Recognition AI</p>
    </div>
    """, unsafe_allow_html=True)


def show_student_login():
    """Show student login page — split screen layout"""
    col_img, col_form, col_pad = st.columns([1.4, 1, 0.2], gap="large")

    with col_img:
        st.markdown("""
        <div class="split-image-panel" style="background-image:url('https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=1200&q=80');">
            <div class="panel-content">
                <div class="brand-badge" style="margin-bottom:18px;">
                    <div class="brand-dot"></div>
                    <span>Student Portal</span>
                </div>
                <h2 style="font-size:26px;font-weight:800;color:#fff;margin:0 0 10px;line-height:1.3;">Track your attendance<br>effortlessly</h2>
                <p style="font-size:13px;color:rgba(255,255,255,0.55);margin:0;line-height:1.6;">View records, mark attendance, and manage your academic profile.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown("<div style='padding:52px 0 24px;'>", unsafe_allow_html=True)
        st.markdown("""
        <h1 style="font-size:22px;font-weight:700;color:#fff;margin:0 0 4px;letter-spacing:-0.3px;">Sign in</h1>
        <p style="font-size:13px;color:#555;margin:0 0 24px;">Student account</p>
        """, unsafe_allow_html=True)

        if st.button("↗  Continue with Google", key="student_google_btn"):
            st.session_state.google_login_role = 'student'
            st.session_state.page = 'google_login'
            st.rerun()

        st.markdown("""
        <div style="display:flex;align-items:center;margin:18px 0 16px;">
            <div style="flex:1;height:1px;background:#1e1e1e;"></div>
            <span style="padding:0 12px;font-size:11px;color:#3a3a3a;letter-spacing:1px;">OR</span>
            <div style="flex:1;height:1px;background:#1e1e1e;"></div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Email or Username", placeholder="you@example.com", key="student_username")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="student_password")

        st.markdown('<div class="forgot-pw-link">', unsafe_allow_html=True)
        if st.button("Forgot password?", key="student_forgot_pw"):
            st.session_state.forgot_password_role = 'student'
            st.session_state.page = 'forgot_password'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Sign In", key="student_login_btn", type="primary", use_container_width=True):
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
                    st.error("Invalid credentials")
            else:
                st.warning("Please fill in all fields")

        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-top:20px;">
            <span style="font-size:13px;color:#444;">No account?</span>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Create account", key="student_register_btn", use_container_width=True):
                st.session_state.page = 'student_register'
                st.rerun()
        with col_b:
            if st.button("← Back", key="student_back_btn", use_container_width=True):
                st.session_state.page = 'role_select'
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def show_student_register():
    """Show student registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 24px;">
            <div class="role-icon" style="margin:0 auto 24px;font-size:26px;">👤</div>
            <h1 style="font-size:28px;margin-bottom:10px;font-weight:700;color:#f0f0f0;letter-spacing:-0.5px;">Student Registration</h1>
            <p style="font-size:14px;color:#666;">Create your account to get started</p>
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
                        success2, msg2 = UserOperations.create_user_with_email(
                            username=username, email=email or None, password=password,
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
    """Show admin login page — split screen layout"""
    col_img, col_form, col_pad = st.columns([1.4, 1, 0.2], gap="large")

    with col_img:
        st.markdown("""
        <div class="split-image-panel" style="background-image:url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1200&q=80');">
            <div class="panel-content">
                <div class="brand-badge" style="margin-bottom:18px;border-color:rgba(202,242,145,0.25);">
                    <div class="brand-dot" style="background:#CAF291;"></div>
                    <span style="color:#CAF291 !important;">Admin Portal</span>
                </div>
                <h2 style="font-size:26px;font-weight:800;color:#fff;margin:0 0 10px;line-height:1.3;">Manage your institution<br>with precision</h2>
                <p style="font-size:13px;color:rgba(255,255,255,0.55);margin:0;line-height:1.6;">Register students, train models, and oversee attendance reports.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown("<div style='padding:52px 0 24px;'>", unsafe_allow_html=True)
        st.markdown("""
        <h1 style="font-size:22px;font-weight:700;color:#fff;margin:0 0 4px;letter-spacing:-0.3px;">Sign in</h1>
        <p style="font-size:13px;color:#555;margin:0 0 24px;">Admin account</p>
        """, unsafe_allow_html=True)

        if st.button("↗  Continue with Google", key="admin_google_btn"):
            st.session_state.google_login_role = 'admin'
            st.session_state.page = 'google_login'
            st.rerun()

        st.markdown("""
        <div style="display:flex;align-items:center;margin:18px 0 16px;">
            <div style="flex:1;height:1px;background:#1e1e1e;"></div>
            <span style="padding:0 12px;font-size:11px;color:#3a3a3a;letter-spacing:1px;">OR</span>
            <div style="flex:1;height:1px;background:#1e1e1e;"></div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Email or Username", placeholder="admin", key="admin_username")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="admin_password")

        st.markdown('<div class="forgot-pw-link">', unsafe_allow_html=True)
        if st.button("Forgot password?", key="admin_forgot_pw"):
            st.session_state.forgot_password_role = 'admin'
            st.session_state.page = 'forgot_password'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Sign In", key="admin_login_btn", type="primary", use_container_width=True):
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
                    st.error("Invalid credentials")
            else:
                st.warning("Please fill in all fields")

        st.markdown("""
        <div style="margin-top:14px;padding:10px 14px;background:#111;border:1px solid #222;border-radius:8px;">
            <p style="font-size:12px;color:#444;margin:0;">Default: <span style="color:#666;">admin / admin123</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:13px;color:#444;margin-top:20px;">Need an account?</p>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Register Admin", key="admin_register_nav_btn", use_container_width=True):
                st.session_state.page = 'admin_register_self'
                st.rerun()
        with col_b:
            if st.button("← Back", key="admin_back_btn", use_container_width=True):
                st.session_state.page = 'role_select'
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def show_admin_register_self():
    """Show admin self-registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 24px;">
            <div class="role-icon" style="margin:0 auto 24px;background:#CAF291;font-size:26px;">🛡️</div>
            <h1 style="font-size:28px;margin-bottom:10px;font-weight:700;color:#f0f0f0;letter-spacing:-0.5px;">Admin Registration</h1>
            <p style="font-size:14px;color:#666;">Create new admin account</p>
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


def show_forgot_password():
    """Show forgot password page with email verification"""
    from utils.email_service import generate_reset_code, send_reset_email

    # Initialize session state for forgot password flow
    if 'reset_step' not in st.session_state:
        st.session_state.reset_step = 1
    if 'reset_email' not in st.session_state:
        st.session_state.reset_email = ''
    if 'reset_code' not in st.session_state:
        st.session_state.reset_code = ''

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        role = st.session_state.get('forgot_password_role', 'student')
        role_icon = 'S' if role == 'student' else 'A'
        role_title = 'Student' if role == 'student' else 'Admin'

        # Header
        st.markdown(f"""
        <div style="text-align:center;padding:40px 0 24px;">
            <div class="role-icon" style="margin:0 auto 20px;font-size:26px;">🔒</div>
            <h1 style="font-size:28px;margin-bottom:8px;font-weight:700;color:#f0f0f0;letter-spacing:-0.5px;">Reset Password</h1>
            <p style="font-size:14px;color:#666;">{role_title} Account Recovery</p>
        </div>
        """, unsafe_allow_html=True)

        # Step 1: Enter email
        if st.session_state.reset_step == 1:
            st.markdown("""
            <div class="info-box" style="margin-bottom:20px;">
                <p style="font-size:14px;color:#70E6ED !important;margin:0;">Enter your registered email address. We'll send you a verification code.</p>
            </div>
            """, unsafe_allow_html=True)

            email = st.text_input("Email Address", placeholder="Enter your registered email", key="reset_email_input")

            if st.button("Send Reset Code", use_container_width=True, type="primary"):
                if email:
                    # Check if email exists
                    user = UserOperations.get_user_by_email(email)
                    if user:
                        # Generate and send code
                        code = generate_reset_code()
                        success, username = UserOperations.set_reset_token(email, code)

                        if success:
                            result = send_reset_email(email, code, username)
                            st.session_state.reset_email = email
                            st.session_state.reset_code = code

                            if result.get('demo_mode'):
                                st.warning(f"Email not configured. Your reset code is: **{code}**")
                                st.info("Set up SMTP_EMAIL and SMTP_PASSWORD environment variables to enable email sending.")
                            else:
                                st.success(f"Reset code sent to {email}")

                            st.session_state.reset_step = 2
                            st.rerun()
                        else:
                            st.error("Failed to generate reset code. Please try again.")
                    else:
                        st.error("No account found with this email address.")
                else:
                    st.warning("Please enter your email address")

        # Step 2: Enter verification code
        elif st.session_state.reset_step == 2:
            st.markdown(f"""
            <div class="info-box" style="margin-bottom:20px;">
                <p style="font-size:14px;color:#70E6ED !important;margin:0;">Enter the 6-digit code sent to <strong style="color:#70E6ED !important;">{st.session_state.reset_email}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            code = st.text_input("Verification Code", placeholder="Enter 6-digit code", key="verify_code_input", max_chars=6)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Verify Code", use_container_width=True, type="primary"):
                    if code:
                        success, msg = UserOperations.verify_reset_token(st.session_state.reset_email, code)
                        if success:
                            st.session_state.reset_step = 3
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please enter the verification code")

            with col_b:
                if st.button("Resend Code", use_container_width=True):
                    code = generate_reset_code()
                    success, username = UserOperations.set_reset_token(st.session_state.reset_email, code)
                    if success:
                        result = send_reset_email(st.session_state.reset_email, code, username)
                        st.session_state.reset_code = code
                        if result.get('demo_mode'):
                            st.warning(f"New reset code: **{code}**")
                        else:
                            st.success("New code sent!")
                        st.rerun()

        # Step 3: Set new password
        elif st.session_state.reset_step == 3:
            st.markdown("""
            <div style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);border-radius:10px;padding:16px;margin-bottom:20px;">
                <p style="font-size:14px;color:#4ade80 !important;margin:0;">Code verified! Enter your new password below.</p>
            </div>
            """, unsafe_allow_html=True)

            new_password = st.text_input("New Password", type="password", placeholder="Enter new password", key="new_pass_input")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm new password", key="confirm_pass_input")

            if st.button("Reset Password", use_container_width=True, type="primary"):
                if new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            success, msg = UserOperations.reset_password_with_token(
                                st.session_state.reset_email,
                                st.session_state.reset_code,
                                new_password
                            )
                            if success:
                                st.success("Password reset successfully! Please login with your new password.")
                                # Clear reset session state
                                st.session_state.reset_step = 1
                                st.session_state.reset_email = ''
                                st.session_state.reset_code = ''
                                # Redirect to login
                                import time
                                time.sleep(2)
                                st.session_state.page = f"{role}_login"
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.error("Password must be at least 6 characters")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill in both password fields")

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        if st.button("Back to Login", use_container_width=True):
            st.session_state.reset_step = 1
            st.session_state.reset_email = ''
            st.session_state.reset_code = ''
            st.session_state.page = f"{st.session_state.get('forgot_password_role', 'student')}_login"
            st.rerun()


def show_google_login():
    """Show Google OAuth login page"""
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        role = st.session_state.get('google_login_role', 'student')
        role_icon = 'G' if role == 'student' else 'G'
        role_title = 'Student' if role == 'student' else 'Admin'

        # Header
        st.markdown(f"""
        <div style="text-align:center;padding:40px 0 24px;">
            <div class="role-icon" style="margin:0 auto 20px;background:linear-gradient(135deg,#4285F4,#34A853);font-size:26px;">G</div>
            <h1 style="font-size:28px;margin-bottom:8px;font-weight:700;color:#f0f0f0;letter-spacing:-0.5px;">Google Sign In</h1>
            <p style="font-size:14px;color:#666;">{role_title} Account</p>
        </div>
        """, unsafe_allow_html=True)

        # Info about Google OAuth setup
        st.markdown("""
        <div class="info-box" style="margin-bottom:24px;padding:24px;">
            <h4 style="color:#70E6ED !important;margin:0 0 12px 0;font-size:16px;">Google OAuth Configuration Required</h4>
            <p style="font-size:14px;color:#a0a0a0 !important;margin:0 0 16px 0;">
                To enable Google Sign-In, set up OAuth credentials in Google Cloud Console.
            </p>
            <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:16px;border:1px solid rgba(255,255,255,0.06);">
                <p style="font-size:13px;color:#a0a0a0 !important;margin:0 0 8px 0;font-weight:600;">Setup Steps:</p>
                <ol style="font-size:13px;color:#a0a0a0 !important;margin:0;padding-left:20px;">
                    <li>Go to Google Cloud Console</li>
                    <li>Create OAuth 2.0 credentials</li>
                    <li>Set GOOGLE_CLIENT_ID & GOOGLE_CLIENT_SECRET env vars</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Demo mode
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:24px;margin-bottom:20px;">
            <h4 style="color:#f0f0f0 !important;margin:0 0 12px 0;font-size:16px;">Demo Mode</h4>
            <p style="font-size:14px;color:#a0a0a0 !important;margin:0;">
                Enter your Google email to simulate sign-in.
            </p>
        </div>
        """, unsafe_allow_html=True)

        email = st.text_input("Google Email", placeholder="yourname@gmail.com", key="google_email_input")
        name = st.text_input("Display Name", placeholder="Your Name", key="google_name_input")

        if st.button("Continue with Google (Demo)", use_container_width=True, type="primary"):
            if email and name:
                if '@' in email:
                    # Simulate Google OAuth by creating/updating user
                    google_id = f"google_{email.replace('@', '_').replace('.', '_')}"
                    success, user_role, student_id, username = UserOperations.create_or_update_google_user(
                        google_id=google_id,
                        email=email,
                        name=name,
                        role=role
                    )

                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_role = user_role
                        st.session_state.student_id = student_id
                        st.session_state.username = username
                        st.session_state.page = f"{user_role}_dashboard"
                        st.success(f"Welcome, {name}!")
                        st.rerun()
                    else:
                        st.error(f"Failed to sign in: {username}")
                else:
                    st.error("Please enter a valid email address")
            else:
                st.warning("Please enter both email and name")

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        if st.button("Back to Login", use_container_width=True):
            st.session_state.page = f"{st.session_state.get('google_login_role', 'student')}_login"
            st.rerun()


def show_student_dashboard():
    """Show student dashboard"""

    student = StudentOperations.get_student(st.session_state.student_id)
    stats = AttendanceOperations.get_student_attendance_stats(st.session_state.student_id)

    # Sidebar
    with st.sidebar:
        student_name = student.name if student else 'Student'
        student_initial = student_name[0].upper()
        st.markdown(f"""
        <div style="padding:8px 4px 16px;">
            <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#70E6ED,#4dd4db);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:#0a0a0a;margin-bottom:10px;">{student_initial}</div>
            <div style="font-size:15px;font-weight:700;color:#f0f0f0;">{student_name}</div>
            <div style="font-size:11px;color:#444;font-family:'JetBrains Mono',monospace;margin-top:2px;">{st.session_state.student_id}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        if st.button("📊  Dashboard", use_container_width=True):
            st.session_state.page = 'student_dashboard'
            st.rerun()
        if st.button("📷  Mark Attendance", use_container_width=True):
            st.session_state.page = 'mark_attendance'
            st.rerun()
        if st.button("👤  My Profile", use_container_width=True):
            st.session_state.page = 'profile'
            st.rerun()
        if st.button("📅  Attendance History", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()
        st.markdown("---")
        if st.button("🚪  Logout", use_container_width=True):
            for key in ['logged_in', 'user_role', 'student_id', 'username']:
                st.session_state[key] = None if key != 'logged_in' else False
            st.session_state.page = 'role_select'
            st.rerun()

    # Header
    st.markdown(f"""
    <div class="header-bar">
        <h2 style="margin:0;">Welcome, {student.name if student else 'Student'}</h2>
        <p style="margin:5px 0 0 0;font-family:'JetBrains Mono',monospace;font-size:13px;">{datetime.now().strftime('%A, %B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{stats["total"]}</div><div class="stat-label">Total Classes</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#4ade80;">{stats["present"]}</div><div class="stat-label">Present</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#f87171;">{stats["absent"]}</div><div class="stat-label">Absent</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#70E6ED;">{stats["percentage"]}%</div><div class="stat-label">Attendance</div></div>', unsafe_allow_html=True)

    # Recent attendance
    st.markdown('<div class="section-title">Recent Attendance</div>', unsafe_allow_html=True)
    records = AttendanceOperations.get_student_attendance(st.session_state.student_id)[:5]

    if records:
        for record in records:
            status_class = "status-present" if record.status == 'Present' else "status-absent"
            time_str = record.time_in.strftime('%H:%M') if record.time_in else 'N/A'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <strong style="color:#f0f0f0;">{record.date.strftime('%B %d, %Y')}</strong><br>
                    <span style="font-size:12px;color:#555;font-family:'JetBrains Mono',monospace;">{time_str}</span>
                </div>
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
        color = "#4ade80" if already_marked else "#70E6ED"
        st.markdown(f"""
        <div class="stat-card">
            <span class="material-symbols-outlined" style="font-size:36px;color:{color};margin-bottom:12px;display:block;">{'check_circle' if already_marked else 'schedule'}</span>
            <div class="stat-value" style="color:{color};font-size:28px;">{status}</div>
            <div class="stat-label">Today's Status</div>
        </div>
        """, unsafe_allow_html=True)

        # Show face registration status
        if not already_marked:
            face_status = "Registered" if face_registered else "Not Registered"
            face_color = "#4ade80" if face_registered else "#f87171"
            st.markdown(f"""
            <div class="stat-card" style="margin-top:16px;">
                <span class="material-symbols-outlined" style="font-size:28px;color:{face_color};margin-bottom:10px;display:block;">{'face' if face_registered else 'no_accounts'}</span>
                <div class="stat-value" style="color:{face_color};font-size:20px;">{face_status}</div>
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
        face_status_color = "#4ade80" if (student and student.face_encoding) else "#f87171"
        face_status_text = "Face Registered" if (student and student.face_encoding) else "No Face Data"
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-avatar">{initial}</div>
            <h3 style="font-size:20px;margin-bottom:6px;">{student.name if student else 'Student'}</h3>
            <p style="color:#888;font-size:13px;margin:0 0 4px;font-family:'JetBrains Mono',monospace;">{student.student_id if student else ''}</p>
            <p style="color:#70E6ED;font-size:13px;margin:0 0 16px;">{student.department if student else ''}</p>
            <span style="background:rgba({('74,222,128' if (student and student.face_encoding) else '248,113,113')},0.12);color:{face_status_color};padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600;border:1px solid rgba({('74,222,128' if (student and student.face_encoding) else '248,113,113')},0.2);">{face_status_text}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
        if student:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="attendance-row" style="padding:12px 16px;margin:6px 0;">
                    <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Email</div>
                    <div style="font-size:14px;color:#f0f0f0;">{student.email or '—'}</div>
                </div>
                <div class="attendance-row" style="padding:12px 16px;margin:6px 0;">
                    <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Phone</div>
                    <div style="font-size:14px;color:#f0f0f0;">{student.phone or '—'}</div>
                </div>
                <div class="attendance-row" style="padding:12px 16px;margin:6px 0;">
                    <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Department</div>
                    <div style="font-size:14px;color:#f0f0f0;">{student.department or '—'}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="attendance-row" style="padding:12px 16px;margin:6px 0;">
                    <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Batch</div>
                    <div style="font-size:14px;color:#f0f0f0;">{student.batch or '—'}</div>
                </div>
                <div class="attendance-row" style="padding:12px 16px;margin:6px 0;">
                    <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Semester</div>
                    <div style="font-size:14px;color:#f0f0f0;">{student.semester or '—'}</div>
                </div>
                <div class="attendance-row" style="padding:12px 16px;margin:6px 0;">
                    <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Section</div>
                    <div style="font-size:14px;color:#f0f0f0;">{student.section or '—'}</div>
                </div>
                """, unsafe_allow_html=True)


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
            time_out = record.time_out.strftime('%H:%M:%S') if record.time_out else '—'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <strong style="color:#f0f0f0;">{record.date.strftime('%A, %B %d, %Y')}</strong><br>
                    <span style="font-size:12px;color:#555;font-family:'JetBrains Mono',monospace;">In: {time_in} &nbsp;|&nbsp; Out: {time_out}</span>
                </div>
                <span class="{status_class}">{record.status}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No attendance records")


def show_admin_dashboard():
    """Show admin dashboard"""

    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 4px 20px;">
            <div style="font-size:20px;font-weight:800;letter-spacing:-0.5px;background:linear-gradient(135deg,#70E6ED,#CAF291);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">AttendEase</div>
            <div style="font-size:11px;color:#444;text-transform:uppercase;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-top:2px;">Admin Panel</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("📊  Dashboard", use_container_width=True):
            st.session_state.page = 'admin_dashboard'
        if st.button("👥  All Students", use_container_width=True):
            st.session_state.page = 'admin_students'
            st.rerun()
        if st.button("➕  Register Student", use_container_width=True):
            st.session_state.page = 'admin_register'
            st.rerun()
        if st.button("📷  Add Face Images", use_container_width=True):
            st.session_state.page = 'admin_capture'
            st.rerun()
        if st.button("🧠  Train Model", use_container_width=True):
            st.session_state.page = 'admin_train'
            st.rerun()
        if st.button("✅  Mark Attendance", use_container_width=True):
            st.session_state.page = 'admin_mark'
            st.rerun()
        st.markdown("---")
        if st.button("🚪  Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = 'role_select'
            st.rerun()

    st.markdown(f"""
    <div class="header-bar">
        <h2 style="margin:0;">Admin Dashboard</h2>
        <p style="margin:5px 0 0 0;font-family:'JetBrains Mono',monospace;font-size:13px;">{datetime.now().strftime('%A, %B %d, %Y')}</p>
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
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#4ade80;">{today_attendance}</div><div class="stat-label">Present Today</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#f87171;">{absent}</div><div class="stat-label">Absent Today</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#70E6ED;">{rate}%</div><div class="stat-label">Attendance Rate</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Today\'s Attendance</div>', unsafe_allow_html=True)
    records = AttendanceOperations.get_daily_attendance()

    if records:
        for record in records:
            student = StudentOperations.get_student(record.student_id)
            time_str = record.time_in.strftime('%H:%M:%S') if record.time_in else 'N/A'
            st.markdown(f"""
            <div class="attendance-row" style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <strong style="color:#f0f0f0;">{student.name if student else record.student_id}</strong><br>
                    <span style="font-size:12px;color:#666;font-family:'JetBrains Mono',monospace;">{record.student_id}</span>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:12px;color:#555;font-family:'JetBrains Mono',monospace;">{time_str}</span>
                    <span class="status-present">{record.status}</span>
                </div>
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
                <div>
                    <strong style="color:#f0f0f0;font-size:15px;">{student.name}</strong><br>
                    <span style="font-size:12px;color:#666;font-family:'JetBrains Mono',monospace;">{student.student_id}</span>
                    <span style="font-size:12px;color:#555;margin-left:8px;">| {student.department or 'N/A'}</span>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:14px;color:#70E6ED;font-weight:700;font-family:'JetBrains Mono',monospace;">{stats['percentage']}%</span>
                    <span class="{face_status}">{face_text}</span>
                </div>
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
    <div class="stat-card" style="text-align:left;margin-bottom:20px;padding:20px 24px;">
        <div style="display:flex;gap:32px;flex-wrap:wrap;">
            <div>
                <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Student ID</div>
                <div style="font-size:14px;color:#f0f0f0;font-family:'JetBrains Mono',monospace;">{student.student_id}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Face Status</div>
                <div style="font-size:14px;color:{'#4ade80' if student.face_encoding else '#f87171'};">{'Registered' if student.face_encoding else 'Not Registered'}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Images</div>
                <div style="font-size:14px;color:#70E6ED;font-family:'JetBrains Mono',monospace;">{student.image_count or 0}</div>
            </div>
        </div>
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
        <div class="stat-card" style="text-align:left;margin:15px 0;padding:18px 24px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                <div>
                    <div style="font-size:16px;font-weight:700;color:#f0f0f0;">{student.name}</div>
                    <div style="font-size:12px;color:#666;font-family:'JetBrains Mono',monospace;margin-top:4px;">{selected_id}</div>
                </div>
                <div style="display:flex;gap:20px;">
                    <div style="text-align:center;">
                        <div style="font-size:22px;font-weight:700;color:#70E6ED;font-family:'JetBrains Mono',monospace;">{existing}</div>
                        <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;">Current</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:22px;font-weight:700;color:#CAF291;font-family:'JetBrains Mono',monospace;">5</div>
                        <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;">Min. Required</div>
                    </div>
                </div>
            </div>
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
    <div style="text-align:center;padding:48px 0 24px;">
        <div style="width:52px;height:52px;border-radius:13px;background:#70E6ED;margin:0 auto 18px;display:flex;align-items:center;justify-content:center;font-size:22px;">⚡</div>
        <h1 style="font-size:32px;font-weight:800;color:#fff;margin:0 0 8px;">Quick Attendance</h1>
        <p style="font-size:14px;color:#666;margin:0;">Face scan to mark your attendance — no login required</p>
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
        elif page == 'forgot_password':
            show_forgot_password()
        elif page == 'google_login':
            show_google_login()
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
