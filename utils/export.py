"""
Export Module
Export attendance data to Excel and PDF
"""

import pandas as pd
from datetime import datetime, date
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import EXPORTS_DIR

logger = logging.getLogger(__name__)


class ExportManager:
    """Handles data export to various formats"""

    def __init__(self):
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def export_to_excel(self, data: list, filename: str = None,
                        sheet_name: str = "Attendance") -> str:
        """Export data to Excel file"""
        try:
            if filename is None:
                filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            filepath = EXPORTS_DIR / filename

            df = pd.DataFrame(data)
            df.to_excel(filepath, sheet_name=sheet_name, index=False)

            logger.info(f"Exported to Excel: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            return None

    def export_to_csv(self, data: list, filename: str = None) -> str:
        """Export data to CSV file"""
        try:
            if filename is None:
                filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            filepath = EXPORTS_DIR / filename

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            logger.info(f"Exported to CSV: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return None

    def export_to_pdf(self, data: list, filename: str = None,
                      title: str = "Attendance Report") -> str:
        """Export data to PDF file"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            if filename is None:
                filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            filepath = EXPORTS_DIR / filename

            # Create document
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            elements = []

            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center
            )

            # Title
            elements.append(Paragraph(title, title_style))
            elements.append(Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles['Normal']
            ))
            elements.append(Spacer(1, 20))

            if data:
                # Create table
                df = pd.DataFrame(data)
                table_data = [df.columns.tolist()] + df.values.tolist()

                table = Table(table_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ]))

                elements.append(table)

            # Build PDF
            doc.build(elements)

            logger.info(f"Exported to PDF: {filepath}")
            return str(filepath)

        except ImportError:
            logger.error("reportlab not installed. Install with: pip install reportlab")
            return None
        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            return None

    def prepare_attendance_data(self, attendance_records: list) -> list:
        """Prepare attendance records for export"""
        data = []
        for record in attendance_records:
            if hasattr(record, '__iter__') and len(record) == 2:
                attendance, student = record
                data.append({
                    'Student ID': attendance.student_id,
                    'Name': student.name,
                    'Department': student.department or 'N/A',
                    'Date': str(attendance.date),
                    'Time In': str(attendance.time_in) if attendance.time_in else 'N/A',
                    'Time Out': str(attendance.time_out) if attendance.time_out else 'N/A',
                    'Status': attendance.status,
                    'Confidence': f"{attendance.confidence_score:.2%}" if attendance.confidence_score else 'N/A'
                })
            else:
                # Single attendance record
                data.append({
                    'Student ID': record.student_id,
                    'Date': str(record.date),
                    'Time In': str(record.time_in) if record.time_in else 'N/A',
                    'Time Out': str(record.time_out) if record.time_out else 'N/A',
                    'Status': record.status,
                    'Confidence': f"{record.confidence_score:.2%}" if record.confidence_score else 'N/A'
                })
        return data

    def prepare_student_data(self, students: list) -> list:
        """Prepare student records for export"""
        data = []
        for student in students:
            data.append({
                'Student ID': student.student_id,
                'Name': student.name,
                'Email': student.email or 'N/A',
                'Phone': student.phone or 'N/A',
                'Department': student.department or 'N/A',
                'Batch': student.batch or 'N/A',
                'Images': student.image_count,
                'Status': 'Active' if student.is_active else 'Inactive',
                'Registered': str(student.created_at.date()) if student.created_at else 'N/A'
            })
        return data


def generate_attendance_summary_report(start_date: date, end_date: date,
                                        summary: dict) -> dict:
    """Generate a summary report dictionary"""
    return {
        'report_period': f"{start_date} to {end_date}",
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_records': summary.get('total', 0),
        'present': summary.get('present', 0),
        'late': summary.get('late', 0),
        'absent': summary.get('absent', 0),
        'attendance_rate': f"{(summary.get('present', 0) / max(summary.get('total', 1), 1)) * 100:.1f}%"
    }
