from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import random
import string

from .schemas.schemas import TokenData
from . import models, schemas
from .database import get_db
from .config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def create_otp(db: Session, email: str):
    """Create and store OTP in database"""
    otp_code = generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.otp_expire_minutes)

    # Invalidate any existing OTPs for this email
    db.query(models.OTP).filter(
        models.OTP.email == email,
        models.OTP.is_used == False
    ).update({"is_used": True})

    # Create new OTP
    db_otp = models.OTP(
        email=email,
        otp_code=otp_code,
        expires_at=expires_at
    )
    db.add(db_otp)
    db.commit()
    db.refresh(db_otp)

    return otp_code


def verify_otp(db: Session, email: str, otp_code: str):
    """Verify OTP from database"""
    otp = db.query(models.OTP).filter(
        models.OTP.email == email,
        models.OTP.otp_code == otp_code,
        models.OTP.is_used == False
    ).first()

    if not otp:
        return False

    # Use timezone-aware datetime for comparison
    if datetime.now(timezone.utc) > otp.expires_at.replace(tzinfo=timezone.utc):
        return False

    # Mark OTP as used
    otp.is_used = True
    db.commit()

    return True


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

