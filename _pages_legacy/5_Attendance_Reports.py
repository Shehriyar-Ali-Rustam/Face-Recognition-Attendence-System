"""
Attendance Reports Page
View and export attendance reports
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.operations import StudentOperations, AttendanceOperations
from utils.export import ExportManager, generate_attendance_summary_report
from utils.helpers import format_date, format_time, get_date_range, calculate_attendance_percentage

# Page configuration
st.set_page_config(
    page_title="Attendance Reports",
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
    .report-table {
        background-color: #ffffff;
        border-radius: 8px;
        overflow: hidden;
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
    .filter-section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def show_summary_stats(start_date: date, end_date: date):
    """Display summary statistics"""
    summary = AttendanceOperations.get_attendance_summary(start_date, end_date)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{summary['total']}</div>
            <div class="stat-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #28a745;">{summary['present']}</div>
            <div class="stat-label">Present</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #ffc107;">{summary['late']}</div>
            <div class="stat-label">Late</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        rate = (summary['present'] / max(summary['total'], 1)) * 100
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{rate:.1f}%</div>
            <div class="stat-label">Attendance Rate</div>
        </div>
        """, unsafe_allow_html=True)


def show_attendance_table(start_date: date, end_date: date,
                          student_filter: str = None):
    """Display attendance records in a table"""
    records = AttendanceOperations.get_attendance_report(start_date, end_date)

    if not records:
        st.info("No attendance records found for the selected period.")
        return []

    # Prepare data for display
    data = []
    for attendance, student in records:
        if student_filter and student_filter not in [student.student_id, 'All']:
            continue

        data.append({
            'Date': str(attendance.date),
            'Student ID': attendance.student_id,
            'Name': student.name,
            'Department': student.department or 'N/A',
            'Time In': format_time(attendance.time_in),
            'Time Out': format_time(attendance.time_out) if attendance.time_out else '-',
            'Status': attendance.status,
            'Confidence': f"{attendance.confidence_score:.0%}" if attendance.confidence_score else 'N/A'
        })

    if not data:
        st.info("No records match the filter criteria.")
        return []

    # Display as dataframe
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    return data


def show_student_attendance_summary():
    """Show attendance summary per student"""
    students = StudentOperations.get_all_students()

    if not students:
        st.info("No students registered.")
        return

    # Get date range for this month
    today = date.today()
    start_of_month = today.replace(day=1)

    data = []
    for student in students:
        records = AttendanceOperations.get_student_attendance(
            student.student_id, start_of_month, today
        )
        present_days = len([r for r in records if r.status == 'Present'])
        late_days = len([r for r in records if r.status == 'Late'])
        total_days = (today - start_of_month).days + 1

        data.append({
            'Student ID': student.student_id,
            'Name': student.name,
            'Department': student.department or 'N/A',
            'Present': present_days,
            'Late': late_days,
            'Total Days': total_days,
            'Attendance %': f"{calculate_attendance_percentage(present_days + late_days, total_days):.1f}%"
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_daily_breakdown(target_date: date):
    """Show detailed breakdown for a specific day"""
    records = AttendanceOperations.get_daily_attendance(target_date)

    st.markdown(f"**Date: {format_date(target_date)}**")

    if not records:
        st.info("No attendance records for this day.")
        return

    st.markdown(f"**Total Present: {len(records)}**")

    for record in records:
        student = StudentOperations.get_student(record.student_id)
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown(f"**{student.name if student else 'Unknown'}**")
            st.caption(f"ID: {record.student_id}")
        with col2:
            st.markdown(f"In: {format_time(record.time_in)}")
            if record.time_out:
                st.markdown(f"Out: {format_time(record.time_out)}")
        with col3:
            status_color = "#28a745" if record.status == 'Present' else "#ffc107"
            st.markdown(f'<span style="color: {status_color}">{record.status}</span>',
                        unsafe_allow_html=True)

        st.divider()


def export_report(data: list, format_type: str):
    """Export attendance report"""
    if not data:
        st.warning("No data to export.")
        return

    export_manager = ExportManager()

    if format_type == 'Excel':
        filepath = export_manager.export_to_excel(data, f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        if filepath:
            with open(filepath, 'rb') as f:
                st.download_button(
                    label="Download Excel",
                    data=f,
                    file_name=Path(filepath).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    elif format_type == 'CSV':
        filepath = export_manager.export_to_csv(data)
        if filepath:
            with open(filepath, 'rb') as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name=Path(filepath).name,
                    mime="text/csv"
                )

    elif format_type == 'PDF':
        filepath = export_manager.export_to_pdf(data, title="Attendance Report")
        if filepath:
            with open(filepath, 'rb') as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=Path(filepath).name,
                    mime="application/pdf"
                )


def main():
    st.markdown('<p class="main-header">Attendance Reports</p>', unsafe_allow_html=True)

    # Tabs for different report views
    tab1, tab2, tab3 = st.tabs(["Daily Report", "Period Report", "Student Summary"])

    with tab1:
        st.markdown('<p class="sub-header">Daily Attendance Report</p>', unsafe_allow_html=True)

        # Date selector
        selected_date = st.date_input(
            "Select Date",
            value=date.today(),
            max_value=date.today()
        )

        show_daily_breakdown(selected_date)

    with tab2:
        st.markdown('<p class="sub-header">Period Report</p>', unsafe_allow_html=True)

        # Filter section
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                period = st.selectbox(
                    "Quick Select",
                    options=['Custom', 'Today', 'This Week', 'This Month', 'This Year']
                )

            with col2:
                if period == 'Custom':
                    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=7))
                else:
                    start_date, _ = get_date_range(period.lower().replace('this ', ''))
                    st.date_input("Start Date", value=start_date, disabled=True)

            with col3:
                if period == 'Custom':
                    end_date = st.date_input("End Date", value=date.today())
                else:
                    _, end_date = get_date_range(period.lower().replace('this ', ''))
                    st.date_input("End Date", value=end_date, disabled=True)

            # Student filter
            students = StudentOperations.get_all_students()
            student_options = ['All'] + [s.student_id for s in students]
            student_filter = st.selectbox("Filter by Student", options=student_options)

        # Get date range
        if period != 'Custom':
            start_date, end_date = get_date_range(period.lower().replace('this ', ''))

        # Show summary
        st.markdown("---")
        show_summary_stats(start_date, end_date)

        st.markdown("---")

        # Show table and get data for export
        data = show_attendance_table(start_date, end_date,
                                     student_filter if student_filter != 'All' else None)

        # Export options
        if data:
            st.markdown("---")
            st.markdown("**Export Report**")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Export to Excel", use_container_width=True):
                    export_report(data, 'Excel')

            with col2:
                if st.button("Export to CSV", use_container_width=True):
                    export_report(data, 'CSV')

            with col3:
                if st.button("Export to PDF", use_container_width=True):
                    export_report(data, 'PDF')

    with tab3:
        st.markdown('<p class="sub-header">Student Attendance Summary</p>', unsafe_allow_html=True)
        st.markdown("Monthly attendance summary for all students:")
        show_student_attendance_summary()


if __name__ == "__main__":
    main()
