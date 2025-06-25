#!/usr/bin/env python3
"""
CV Analyzer Router - Simple version that uses the existing CVAnalyzer module

Place this file in: routers/cv_router.py
"""

import os
import tempfile
import json
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from ..auth import get_current_user
from ..config import settings

from sqlalchemy.orm import Session
from sqlalchemy import desc

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from pydantic import BaseModel

# Import the existing CV analyzer module
# Adjust the import path based on where cv_analyzer.py is located
from ..cv_analyzer import CVAnalyzer
from ..database import get_db
from ..models import User, CVAnalysis

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/cv",
    tags=["CV Analysis"]
)

# Initialize CV analyzer once
try:
    cv_analyzer = CVAnalyzer(api_key=settings.openai_api_key)
    analyzer_available = True
except Exception as e:
    logger.error(f"Failed to initialize CV Analyzer: {e}")
    cv_analyzer = None
    analyzer_available = False


class AnalysisResponse(BaseModel):
    """Response model for CV analysis"""
    success: bool
    profile: dict
    message: str = "CV analyzed successfully"

class CVDataResponse(BaseModel):
    """Response model for CV data retrieval"""
    id: int
    filename: str
    profile: dict
    model_used: str
    created_at: datetime
    updated_at: datetime


@router.get("/health")
async def health_check(
    current_user: User = Depends(get_current_user),
):
    """Check if CV analyzer is ready"""
    return {
        "status": "healthy" if analyzer_available else "unhealthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "analyzer_ready": analyzer_available
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_cv(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
        model: Optional[str] = Form(default="gpt-4o-mini")
):
    """
    Analyze uploaded CV file.

    Simply saves the uploaded file temporarily and calls the existing analyze function.
    """
    # Check if analyzer is available
    if not analyzer_available:
        raise HTTPException(
            status_code=503,
            detail="CV Analyzer service unavailable. Check OpenAI API key."
        )

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Create temporary file and analyze
    temp_file = None
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Call the existing analyze function
        profile = cv_analyzer.analyze(temp_path, model=model)

        # Convert to dict (handling date serialization)
        profile_dict = json.loads(
            json.dumps(profile.model_dump(), default=str)
        )

        current_user.has_uploaded_cv = True
        current_user.cv_filename = file.filename
        current_user.cv_upload_date = datetime.utcnow()

        db.commit()

        cv_analysis = CVAnalysis(
            user_id=current_user.id,
            filename=file.filename,
            analysis_data=json.dumps(profile_dict),
            model_used=model
        )
        db.add(cv_analysis)

        current_user.has_uploaded_cv = True
        current_user.cv_filename = file.filename
        current_user.cv_upload_date = datetime.utcnow()

        db.commit()

        return AnalysisResponse(
            success=True,
            profile=profile_dict
        )

    except ValueError as e:
        # Handle CV analyzer ValueError (e.g., "Input file must be a PDF")
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        # This shouldn't happen with temp files, but just in case
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        # Handle other errors
        error_msg = str(e)

        # Provide user-friendly error messages for common issues
        if "OPENAI_API_KEY" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )
        elif "extract text" in error_msg.lower():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from PDF. File may be corrupted or image-based."
            )
        elif "rate limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="OpenAI API rate limit exceeded. Please try again later."
            )
        else:
            logger.error(f"CV analysis failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {error_msg}"
            )

    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass


@router.get("/data")
async def get_cv_data(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
) -> Dict[str, Any]:  # Return dict instead of using response_model
    """
    Get the most recent CV analysis data for the authenticated user.
    """
    # Query for the most recent CV analysis
    cv_analysis = db.query(CVAnalysis).filter(
        CVAnalysis.user_id == current_user.id
    ).order_by(desc(CVAnalysis.created_at)).first()

    if not cv_analysis:
        raise HTTPException(
            status_code=404,
            detail="No CV analysis found. Please upload and analyze a CV first."
        )

    # Parse the stored JSON data
    try:
        profile_data = json.loads(cv_analysis.analysis_data)
        print(profile_data)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse CV data for user {current_user.id}")
        raise HTTPException(
            status_code=500,
            detail="Error parsing stored CV data"
        )


    # Return as dict to avoid Pydantic issues
    return {
        "id": cv_analysis.id,
        "filename": cv_analysis.filename,
        "profile": profile_data,
        "model_used": cv_analysis.model_used,
        "created_at": cv_analysis.created_at.isoformat() if cv_analysis.created_at else None,
        "updated_at": cv_analysis.updated_at.isoformat() if cv_analysis.updated_at else None
    }

