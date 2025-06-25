from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str  # Add this
    phone_number: Optional[str] = None

    @validator('password')
    def validate_password(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserLoginResponse(BaseModel):
    """Extended user response for login with additional fields"""
    id: int
    email: str
    full_name: str
    phone_number: Optional[str] = None
    is_active: bool
    is_verified: bool
    has_uploaded_cv: bool
    is_first_login: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserLoginResponse



class LoginSuccessResponse(BaseModel):
    """Standard API response for login"""
    status: str = "success"
    data: TokenResponse
    message: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    phone_number: Optional[str] = None
    is_active: bool
    is_verified: bool
    has_uploaded_cv: bool
    has_preferences_data: bool# Add this
    cv_filename: Optional[str] = None  # Add this
    cv_upload_date: Optional[datetime] = None  # Add this
    last_login: Optional[datetime] = None  # Add this
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class OTPVerify(BaseModel):
    email: EmailStr
    otp_code: str = Field(..., min_length=6, max_length=6)


class MessageResponse(BaseModel):
    message: str

