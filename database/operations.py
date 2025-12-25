"""
Database Operations
CRUD operations for Face Recognition Attendance System
"""

from datetime import datetime, date, timedelta
from sqlalchemy import func, and_, or_
from sqlalchemy.exc import IntegrityError
import json
import hashlib
import logging

from .models import get_session, User, Student, Attendance, TrainingLog, SystemLog

logger = logging.getLogger(__name__)


class UserOperations:
    """CRUD operations for users"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def create_user(username: str, password: str, role: str = 'student',
                    student_id: str = None) -> tuple:
        """Create a new user"""
        session = get_session()
        try:
            hashed_pw = UserOperations.hash_password(password)
            user = User(
                username=username,
                password=hashed_pw,
                role=role,
                student_id=student_id
            )
            session.add(user)
            session.commit()
            return True, "User created successfully"
        except IntegrityError:
            session.rollback()
            return False, "Username already exists"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def authenticate(username: str, password: str) -> tuple:
        """Authenticate a user"""
        session = get_session()
        try:
            hashed_pw = UserOperations.hash_password(password)
            user = session.query(User).filter(
                and_(
                    User.username == username,
                    User.password == hashed_pw,
                    User.is_active == True
                )
            ).first()

            if user:
                user.last_login = datetime.now()
                session.commit()
                return True, user.role, user.student_id
            return False, None, None
        except Exception as e:
            return False, None, None
        finally:
            session.close()

    @staticmethod
    def get_user(username: str):
        """Get user by username"""
        session = get_session()
        try:
            return session.query(User).filter(User.username == username).first()
        finally:
            session.close()

    @staticmethod
    def update_password(username: str, new_password: str) -> tuple:
        """Update user password"""
        session = get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return False, "User not found"
            user.password = UserOperations.hash_password(new_password)
            session.commit()
            return True, "Password updated"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()


class StudentOperations:
    """CRUD operations for students"""

    @staticmethod
    def create_student(student_id: str, name: str, email: str = None,
                       phone: str = None, department: str = None,
                       batch: str = None, semester: str = None,
                       section: str = None, address: str = None,
                       profile_picture: str = None) -> tuple:
        """Create a new student record"""
        session = get_session()
        try:
            student = Student(
                student_id=student_id,
                name=name,
                email=email,
                phone=phone,
                department=department,
                batch=batch,
                semester=semester,
                section=section,
                address=address,
                profile_picture=profile_picture
            )
            session.add(student)
            session.commit()
            return True, "Student registered successfully"
        except IntegrityError:
            session.rollback()
            return False, "Student ID or email already exists"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def get_student(student_id: str) -> Student:
        """Get a student by ID"""
        session = get_session()
        try:
            return session.query(Student).filter(
                Student.student_id == student_id
            ).first()
        finally:
            session.close()

    @staticmethod
    def get_all_students(active_only: bool = True) -> list:
        """Get all students"""
        session = get_session()
        try:
            query = session.query(Student)
            if active_only:
                query = query.filter(Student.is_active == True)
            return query.order_by(Student.name).all()
        finally:
            session.close()

    @staticmethod
    def update_student(student_id: str, **kwargs) -> tuple:
        """Update student information"""
        session = get_session()
        try:
            student = session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            if not student:
                return False, "Student not found"

            for key, value in kwargs.items():
                if hasattr(student, key):
                    setattr(student, key, value)

            student.updated_at = datetime.now()
            session.commit()
            return True, "Student updated successfully"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def update_face_encoding(student_id: str, encoding: list, image_count: int) -> tuple:
        """Update student's face encoding"""
        session = get_session()
        try:
            student = session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            if not student:
                return False, "Student not found"

            student.face_encoding = json.dumps(encoding)
            student.image_count = image_count
            student.updated_at = datetime.now()
            session.commit()
            return True, "Face encoding updated"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def delete_student(student_id: str, soft_delete: bool = True) -> tuple:
        """Delete a student"""
        session = get_session()
        try:
            student = session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            if not student:
                return False, "Student not found"

            if soft_delete:
                student.is_active = False
                student.updated_at = datetime.now()
            else:
                session.delete(student)

            session.commit()
            return True, "Student deleted successfully"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def get_students_with_encodings() -> list:
        """Get all students who have face encodings"""
        session = get_session()
        try:
            return session.query(Student).filter(
                and_(
                    Student.is_active == True,
                    Student.face_encoding.isnot(None)
                )
            ).all()
        finally:
            session.close()

    @staticmethod
    def search_students(query: str) -> list:
        """Search students by name or ID"""
        session = get_session()
        try:
            return session.query(Student).filter(
                and_(
                    Student.is_active == True,
                    or_(
                        Student.name.ilike(f'%{query}%'),
                        Student.student_id.ilike(f'%{query}%')
                    )
                )
            ).all()
        finally:
            session.close()

    @staticmethod
    def get_student_count() -> int:
        """Get total number of active students"""
        session = get_session()
        try:
            return session.query(func.count(Student.id)).filter(
                Student.is_active == True
            ).scalar()
        finally:
            session.close()


class AttendanceOperations:
    """CRUD operations for attendance"""

    @staticmethod
    def mark_attendance(student_id: str, confidence_score: float = None,
                        status: str = 'Present', notes: str = None) -> tuple:
        """Mark attendance for a student"""
        session = get_session()
        try:
            today = date.today()
            current_time = datetime.now().time()

            existing = session.query(Attendance).filter(
                and_(
                    Attendance.student_id == student_id,
                    Attendance.date == today
                )
            ).first()

            if existing:
                existing.time_out = current_time
                session.commit()
                return True, "Time out recorded"

            attendance = Attendance(
                student_id=student_id,
                date=today,
                time_in=current_time,
                status=status,
                confidence_score=confidence_score,
                notes=notes
            )
            session.add(attendance)
            session.commit()
            return True, "Attendance marked successfully"
        except IntegrityError:
            session.rollback()
            return False, "Attendance already marked for today"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def check_attendance_exists(student_id: str, check_date: date = None) -> bool:
        """Check if attendance exists for a student on a given date"""
        session = get_session()
        try:
            if check_date is None:
                check_date = date.today()

            existing = session.query(Attendance).filter(
                and_(
                    Attendance.student_id == student_id,
                    Attendance.date == check_date
                )
            ).first()
            return existing is not None
        finally:
            session.close()

    @staticmethod
    def get_daily_attendance(target_date: date = None) -> list:
        """Get all attendance records for a specific date"""
        session = get_session()
        try:
            if target_date is None:
                target_date = date.today()

            return session.query(Attendance).filter(
                Attendance.date == target_date
            ).order_by(Attendance.time_in).all()
        finally:
            session.close()

    @staticmethod
    def get_student_attendance(student_id: str, start_date: date = None,
                                end_date: date = None) -> list:
        """Get attendance records for a specific student"""
        session = get_session()
        try:
            query = session.query(Attendance).filter(
                Attendance.student_id == student_id
            )

            if start_date:
                query = query.filter(Attendance.date >= start_date)
            if end_date:
                query = query.filter(Attendance.date <= end_date)

            return query.order_by(Attendance.date.desc()).all()
        finally:
            session.close()

    @staticmethod
    def get_student_attendance_stats(student_id: str) -> dict:
        """Get attendance statistics for a student"""
        session = get_session()
        try:
            total = session.query(func.count(Attendance.id)).filter(
                Attendance.student_id == student_id
            ).scalar()

            present = session.query(func.count(Attendance.id)).filter(
                and_(
                    Attendance.student_id == student_id,
                    Attendance.status == 'Present'
                )
            ).scalar()

            late = session.query(func.count(Attendance.id)).filter(
                and_(
                    Attendance.student_id == student_id,
                    Attendance.status == 'Late'
                )
            ).scalar()

            return {
                'total': total or 0,
                'present': present or 0,
                'late': late or 0,
                'absent': (total or 0) - (present or 0) - (late or 0),
                'percentage': round((present or 0) / max(total or 1, 1) * 100, 1)
            }
        finally:
            session.close()

    @staticmethod
    def get_attendance_summary(start_date: date, end_date: date) -> dict:
        """Get attendance summary statistics"""
        session = get_session()
        try:
            total_records = session.query(func.count(Attendance.id)).filter(
                and_(
                    Attendance.date >= start_date,
                    Attendance.date <= end_date
                )
            ).scalar()

            present_count = session.query(func.count(Attendance.id)).filter(
                and_(
                    Attendance.date >= start_date,
                    Attendance.date <= end_date,
                    Attendance.status == 'Present'
                )
            ).scalar()

            late_count = session.query(func.count(Attendance.id)).filter(
                and_(
                    Attendance.date >= start_date,
                    Attendance.date <= end_date,
                    Attendance.status == 'Late'
                )
            ).scalar()

            return {
                'total': total_records or 0,
                'present': present_count or 0,
                'late': late_count or 0,
                'absent': (total_records or 0) - (present_count or 0) - (late_count or 0)
            }
        finally:
            session.close()

    @staticmethod
    def get_attendance_report(start_date: date, end_date: date) -> list:
        """Get detailed attendance report with student info"""
        session = get_session()
        try:
            results = session.query(
                Attendance, Student
            ).join(
                Student, Attendance.student_id == Student.student_id
            ).filter(
                and_(
                    Attendance.date >= start_date,
                    Attendance.date <= end_date
                )
            ).order_by(Attendance.date.desc(), Attendance.time_in).all()

            return results
        finally:
            session.close()

    @staticmethod
    def update_attendance(attendance_id: int, **kwargs) -> tuple:
        """Update an attendance record"""
        session = get_session()
        try:
            attendance = session.query(Attendance).filter(
                Attendance.id == attendance_id
            ).first()
            if not attendance:
                return False, "Attendance record not found"

            for key, value in kwargs.items():
                if hasattr(attendance, key):
                    setattr(attendance, key, value)

            session.commit()
            return True, "Attendance updated successfully"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def delete_attendance(attendance_id: int) -> tuple:
        """Delete an attendance record"""
        session = get_session()
        try:
            attendance = session.query(Attendance).filter(
                Attendance.id == attendance_id
            ).first()
            if not attendance:
                return False, "Attendance record not found"

            session.delete(attendance)
            session.commit()
            return True, "Attendance record deleted"
        except Exception as e:
            session.rollback()
            return False, f"Error: {str(e)}"
        finally:
            session.close()

    @staticmethod
    def get_today_attendance_count() -> int:
        """Get today's attendance count"""
        session = get_session()
        try:
            return session.query(func.count(Attendance.id)).filter(
                Attendance.date == date.today()
            ).scalar() or 0
        finally:
            session.close()


class TrainingLogOperations:
    """Operations for training logs"""

    @staticmethod
    def create_training_log(num_students: int, num_images: int,
                            model_path: str = None, accuracy: float = None,
                            notes: str = None) -> int:
        """Create a new training log entry"""
        session = get_session()
        try:
            log = TrainingLog(
                num_students=num_students,
                num_images=num_images,
                model_path=model_path,
                accuracy=accuracy,
                notes=notes
            )
            session.add(log)
            session.commit()
            return log.id
        except Exception as e:
            session.rollback()
            return None
        finally:
            session.close()

    @staticmethod
    def get_latest_training() -> TrainingLog:
        """Get the most recent training log"""
        session = get_session()
        try:
            return session.query(TrainingLog).order_by(
                TrainingLog.training_date.desc()
            ).first()
        finally:
            session.close()

    @staticmethod
    def get_all_training_logs() -> list:
        """Get all training logs"""
        session = get_session()
        try:
            return session.query(TrainingLog).order_by(
                TrainingLog.training_date.desc()
            ).all()
        finally:
            session.close()


class SystemLogOperations:
    """Operations for system logs"""

    @staticmethod
    def log_activity(level: str, module: str, message: str,
                     user_action: str = None):
        """Log a system activity"""
        session = get_session()
        try:
            log = SystemLog(
                level=level,
                module=module,
                message=message,
                user_action=user_action
            )
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
        finally:
            session.close()

    @staticmethod
    def get_recent_logs(limit: int = 100) -> list:
        """Get recent system logs"""
        session = get_session()
        try:
            return session.query(SystemLog).order_by(
                SystemLog.timestamp.desc()
            ).limit(limit).all()
        finally:
            session.close()
