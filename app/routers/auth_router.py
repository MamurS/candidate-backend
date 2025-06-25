from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from pydantic import EmailStr
from .. import models, schemas, auth, email_utils
from ..auth import create_access_token, verify_password
from ..database import get_db
from ..config import settings
from ..models import User
from ..schemas.schemas import UserLoginResponse, UserResponse, UserLogin, LoginSuccessResponse, TokenResponse, MessageResponse, UserRegister, Token, OTPVerify

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=MessageResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user and send OTP"""

    # Check if user already exists
    existing_user = db.query(models.User).filter(
        models.User.email == user_data.email
    ).first()

    if existing_user:
        if existing_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            # User exists but not verified, update password and resend OTP
            existing_user.hashed_password = auth.get_password_hash(user_data.password)
            existing_user.hashed_password = auth.get_password_hash(user_data.password)
            existing_user.full_name = user_data.full_name  # Add this
            existing_user.phone_number = user_data.phone_number

            db.commit()

            # Generate new OTP
            otp_code = auth.create_otp(db, user_data.email)

            # Send OTP email
            await email_utils.send_otp_email(user_data.email, otp_code)

            return {"message": "Registration updated. Please check your email for OTP."}

    # Create new user
    hashed_password = auth.get_password_hash(user_data.password)
    db_user = models.User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,  # Add this
        phone_number=user_data.phone_number
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Generate OTP
    otp_code = auth.create_otp(db, user_data.email)

    # Send OTP email
    await email_utils.send_otp_email(user_data.email, otp_code)

    return {"message": "Registration successful. Please check your email for OTP."}


@router.post("/login", response_model=LoginSuccessResponse)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login user and return token with CV status"""
    # Find user
    user = db.query(User).filter(User.email == user_credentials.email).first()

    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email first"
        )

    # Check if first login
    is_first_login = False

    # Update last login
    #user.last_login = datetime.utcnow()
    db.commit()

    # Create access token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )

    # Prepare user response
    user_response = UserLoginResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        phone_number=user.phone_number,
        is_active=user.is_active,
        is_verified=user.is_verified,
        has_uploaded_cv=user.has_uploaded_cv,
        is_first_login=is_first_login,
        created_at=user.created_at
    )

    # Prepare token response
    token_response = TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )

    return LoginSuccessResponse(
        status="success",
        data=token_response,
        message="Login successful"
    )


@router.post("/verify-otp", response_model=Token)
async def verify_otp(otp_data: OTPVerify, db: Session = Depends(get_db)):
    """Verify OTP and activate user account"""

    # Verify OTP
    if not auth.verify_otp(db, otp_data.email, otp_data.otp_code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )

    # Activate user
    user = db.query(models.User).filter(models.User.email == otp_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user.is_active = True
    user.is_verified = True
    db.commit()

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/resend-otp", response_model=MessageResponse)
async def resend_otp(email: EmailStr, db: Session = Depends(get_db)):
    """Resend OTP to email"""

    # Check if user exists
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )

    # Generate new OTP
    otp_code = auth.create_otp(db, email)

    # Send OTP email
    await email_utils.send_otp_email(email, otp_code)

    return {"message": "OTP resent. Please check your email."}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: models.User = Depends(auth.get_current_active_user)):
    """Get current user information"""
    return current_user

