"""
Database Models
Face Recognition Attendance System
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float,
    Boolean, Text, ForeignKey, Date, Time, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from pathlib import Path

# Database path
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "attendance.db"

Base = declarative_base()


class Student(Base):
    """Student registration model"""
    __tablename__ = 'students'

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    department = Column(String(100), nullable=True)
    batch = Column(String(50), nullable=True)
    semester = Column(String(20), nullable=True)
    section = Column(String(10), nullable=True)
    address = Column(Text, nullable=True)
    profile_picture = Column(Text, nullable=True)
    face_encoding = Column(Text, nullable=True)
    image_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    attendance_records = relationship("Attendance", back_populates="student")

    def __repr__(self):
        return f"<Student(id={self.student_id}, name={self.name})>"


class User(Base):
    """User authentication model"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default='student')
    student_id = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(username={self.username}, role={self.role})>"


class Attendance(Base):
    """Attendance records model"""
    __tablename__ = 'attendance'

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), ForeignKey('students.student_id'), nullable=False)
    date = Column(Date, nullable=False)
    time_in = Column(Time, nullable=True)
    time_out = Column(Time, nullable=True)
    status = Column(String(20), default='Present')
    confidence_score = Column(Float, nullable=True)
    verification_method = Column(String(50), default='face_recognition')
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        UniqueConstraint('student_id', 'date', name='unique_daily_attendance'),
    )

    student = relationship("Student", back_populates="attendance_records")

    def __repr__(self):
        return f"<Attendance(student={self.student_id}, date={self.date}, status={self.status})>"


class TrainingLog(Base):
    """Model training history"""
    __tablename__ = 'training_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_date = Column(DateTime, default=datetime.now)
    num_students = Column(Integer, default=0)
    num_images = Column(Integer, default=0)
    model_path = Column(String(255), nullable=True)
    accuracy = Column(Float, nullable=True)
    status = Column(String(50), default='completed')
    notes = Column(Text, nullable=True)


class SystemLog(Base):
    """System activity logs"""
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    level = Column(String(20), default='INFO')
    module = Column(String(100), nullable=True)
    message = Column(Text, nullable=True)
    user_action = Column(String(100), nullable=True)


def init_database():
    """Initialize the database and create all tables"""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f'sqlite:///{DATABASE_PATH}', echo=False)
    Base.metadata.create_all(engine)

    # Create default admin if not exists
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        admin = session.query(User).filter(User.username == 'admin').first()
        if not admin:
            import hashlib
            hashed_pw = hashlib.sha256('admin123'.encode()).hexdigest()
            admin = User(username='admin', password=hashed_pw, role='admin')
            session.add(admin)
            session.commit()
    except:
        session.rollback()
    finally:
        session.close()

    return engine


def get_session():
    """Get a database session"""
    engine = init_database()
    Session = sessionmaker(bind=engine)
    return Session()
