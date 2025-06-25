from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import enum


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)

    full_name = Column(String, nullable=False)  # Add this
    phone_number = Column(String, nullable=True)

    has_uploaded_cv = Column(Boolean, default=False)
    cv_filename = Column(String, nullable=True)
    cv_path = Column(String, nullable=True)  # Full path to CV file
    cv_upload_date = Column(DateTime(timezone=True), nullable=True)

    cv_analyses = relationship("CVAnalysis", back_populates="user", cascade="all, delete-orphan")

    has_preferences_data = Column(Boolean, default=False)

    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=False)  # False until email verified
    is_verified = Column(Boolean, default=False)



    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())



class OTP(Base):
    __tablename__ = "otps"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True, nullable=False)
    otp_code = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False)


class CVAnalysis(Base):
    __tablename__ = "cv_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    analysis_data = Column(Text, nullable=False)  # JSON stored as text
    model_used = Column(String(50), default="gpt-4o-mini")
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Changed to use func.now()
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # Changed to use func.now()

    # Relationship
    user = relationship("User", back_populates="cv_analyses")


class SalaryRange(enum.Enum):
    RANGE_5K = "5000"
    RANGE_10K = "10000"
    RANGE_15K = "15000"
    RANGE_20K = "20000"


class JobType(enum.Enum):
    FULL_TIME = "full-time"
    PART_TIME = "part-time"
    CONTRACT = "contract"
    FREELANCE = "freelance"


class RelocationPreference(enum.Enum):
    YES_WITH_SUPPORT = "yes-with-support"
    YES_WITHOUT_SUPPORT = "yes-without-support"
    NO = "no"


class RemotePreference(enum.Enum):
    HYBRID = "hybrid"
    REMOTE_ONLY = "remote"
    ONSITE_ONLY = "onsite"


class ExperienceLevel(enum.Enum):
    ZERO_TO_TWO = "0-2"
    THREE_TO_FIVE = "3-5"
    FIVE_PLUS = "5+"
    TEN_PLUS = "10+"


class NoticePeriod(enum.Enum):
    IMMEDIATE = "immediate"
    ONE_WEEK = "1-week"
    TWO_WEEKS = "2-weeks"
    ONE_MONTH = "1-month"
    TWO_MONTHS = "2-months"


class WorkPreferences(Base):
    __tablename__ = "work_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Work preference fields
    expected_salary = Column(Enum(SalaryRange), nullable=False)
    job_type = Column(Enum(JobType), nullable=False)
    desired_role = Column(String, nullable=False)  # Store as string for flexibility
    industries = Column(String, nullable=False)  # Can store comma-separated values or JSON
    willing_to_relocate = Column(Enum(RelocationPreference), nullable=False)
    remote_work_preference = Column(Enum(RemotePreference), nullable=False)
    years_of_experience = Column(Enum(ExperienceLevel), nullable=False)
    notice_period = Column(Enum(NoticePeriod), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship
    user = relationship("User", backref="work_preferences", uselist=False)