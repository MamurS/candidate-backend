import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# Get database URL from environment or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    settings.database_url  # Use your settings as fallback
)

# Render uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create database engine using the DATABASE_URL variable
engine = create_engine(DATABASE_URL)  # <-- Use DATABASE_URL here, not settings.database_url

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()