from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional

from ..database import get_db
from ..models import WorkPreferences, User, SalaryRange, JobType, RelocationPreference, RemotePreference, \
    ExperienceLevel, NoticePeriod
from ..schemas.work_preferences import WorkPreferencesCreate, WorkPreferencesUpdate, WorkPreferencesResponse
from ..auth import get_current_user  # Assuming you have this authentication dependency

router = APIRouter(
    prefix="/work-preferences",
    tags=["work-preferences"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=WorkPreferencesResponse, status_code=status.HTTP_201_CREATED)
async def create_work_preferences(
        preferences: WorkPreferencesCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Create or update work preferences for the authenticated user.
    """
    # Convert string values to Enum types
    preferences_data = preferences.dict()
    preferences_data['expected_salary'] = SalaryRange(preferences_data['expected_salary'])
    preferences_data['job_type'] = JobType(preferences_data['job_type'])
    preferences_data['willing_to_relocate'] = RelocationPreference(preferences_data['willing_to_relocate'])
    preferences_data['remote_work_preference'] = RemotePreference(preferences_data['remote_work_preference'])
    preferences_data['years_of_experience'] = ExperienceLevel(preferences_data['years_of_experience'])
    if preferences_data.get('notice_period'):
        preferences_data['notice_period'] = NoticePeriod(preferences_data['notice_period'])

    try:
        # Check if work preferences already exist for this user
        existing_preferences = db.query(WorkPreferences).filter(
            WorkPreferences.user_id == current_user.id
        ).first()

        if existing_preferences:
            # Update existing preferences
            for key, value in preferences_data.items():
                setattr(existing_preferences, key, value)
            db.commit()
            db.refresh(existing_preferences)
            result = existing_preferences
        else:
            # Create new preferences
            db_preferences = WorkPreferences(
                user_id=current_user.id,
                **preferences_data
            )
            db.add(db_preferences)
            db.commit()
            db.refresh(db_preferences)
            result = db_preferences

        # Update user's has_preferences_data flag
        current_user.has_preferences_data = True
        db.commit()
        db.refresh(current_user)

        return result

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save work preferences: {str(e)}"
        )




@router.get("/", response_model=Optional[WorkPreferencesResponse])
async def get_work_preferences(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Get work preferences for the authenticated user.
    """
    preferences = db.query(WorkPreferences).filter(
        WorkPreferences.user_id == current_user.id
    ).first()

    if not preferences:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Work preferences not found"
        )

    return preferences
