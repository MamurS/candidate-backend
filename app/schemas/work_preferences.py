from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class WorkPreferencesBase(BaseModel):
    expected_salary: str = Field(..., pattern="^(5000|10000|15000|20000)$")
    job_type: str = Field(..., pattern="^(full-time|part-time|contract|freelance)$")
    desired_role: str = Field(..., min_length=1, max_length=255)
    industries: str = Field(..., min_length=1, max_length=500)
    willing_to_relocate: str = Field(..., pattern="^(yes-with-support|yes-without-support|no)$")
    remote_work_preference: str = Field(..., pattern="^(hybrid|remote|onsite)$")
    years_of_experience: str = Field(..., pattern="^(0-2|3-5|5\\+|10\\+)$")
    notice_period: Optional[str] = Field(None, pattern="^(immediate|1-week|2-weeks|1-month|2-months)$")

class WorkPreferencesCreate(WorkPreferencesBase):
    pass

class WorkPreferencesUpdate(WorkPreferencesBase):
    pass

class WorkPreferencesResponse(WorkPreferencesBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True