#!/usr/bin/env python3
"""
FastAPI Job Search API
Provides REST endpoints for job searching and matching
"""

from fastapi import FastAPI,APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
import time
import uuid
from enum import Enum

# Import the existing classes
try:
    from ..vacancy_grabber_v2 import FindAJobGrabber
    from ..job_matcher import GeneralJobMatcher
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure vacancy_grabber_v2.py and job_matcher.py are in the same directory")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/jobs",
    tags=["job-search"],
    responses={404: {"description": "Not found"}},
)


# Global cache for job search results (in production, use Redis or similar)
search_cache = {}


# Enums for valid options
class MatcherModel(str, Enum):
    FAST = "paraphrase-MiniLM-L3-v2"
    BALANCED = "all-MiniLM-L6-v2"
    BEST = "all-mpnet-base-v2"


class DaysPosted(int, Enum):
    ONE = 1
    THREE = 3
    SEVEN = 7
    FOURTEEN = 14


# Pydantic models for request/response
class CVProfile(BaseModel):
    full_name: str
    email: str
    summary: Optional[str] = None
    skills: List[str] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    languages: List[str] = []


class JobSearchRequest(BaseModel):
    cv_profile: CVProfile
    role: str = Field(..., description="Job role/title to search for")
    max_jobs: int = Field(50, ge=1, le=200, description="Maximum number of jobs to scrape")
    min_match_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum match score (0-1)")
    days_posted: Optional[DaysPosted] = Field(None, description="Filter jobs posted within X days")
    matcher_model: MatcherModel = Field(MatcherModel.BALANCED, description="Embedding model to use")
    use_async: bool = Field(True, description="Use asynchronous scraping")


class JobMatch(BaseModel):
    job_details: Dict[str, Any]
    overall_score: float
    semantic_similarity: float
    keyword_match_score: float
    tfidf_similarity: float
    matched_keywords: List[str]
    common_skills: List[str]
    match_percentage: float
    match_explanation: str


class JobSearchResponse(BaseModel):
    search_id: str
    timestamp: datetime
    candidate_name: str
    search_role: str
    total_jobs_scraped: int
    total_matches: int
    execution_time: float
    matches: List[JobMatch]
    match_distribution: Dict[str, int]
    average_match_score: Optional[float]


class SearchStatusResponse(BaseModel):
    search_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[JobSearchResponse] = None


# Helper functions
def calculate_match_distribution(matches: List[Dict]) -> Dict[str, int]:
    """Calculate distribution of match quality"""
    distribution = {
        "excellent": sum(1 for m in matches if m['overall_score'] >= 0.8),
        "good": sum(1 for m in matches if 0.65 <= m['overall_score'] < 0.8),
        "moderate": sum(1 for m in matches if 0.5 <= m['overall_score'] < 0.65),
        "fair": sum(1 for m in matches if m['overall_score'] < 0.5)
    }
    return distribution


async def run_job_search_async(
        search_id: str,
        cv_profile: Dict,
        job_preferences: Dict,
        max_jobs: int,
        min_match_score: float,
        days_posted: Optional[int],
        use_async: bool,
        matcher_model: str
) -> None:
    """Run job search asynchronously and store results in cache"""
    try:
        # Update status
        search_cache[search_id] = {
            "status": "processing",
            "progress": "Initializing components...",
            "result": None
        }

        start_time = time.time()

        # Initialize components
        grabber = FindAJobGrabber()
        matcher = GeneralJobMatcher(model_name=matcher_model)

        # Step 1: Scrape jobs
        search_cache[search_id]["progress"] = f"Scraping jobs for '{job_preferences['desired_role']}'..."

        if use_async and max_jobs > 20:
            vacancies = grabber.grab_vacancies_async(
                job_preferences,
                max_results=max_jobs,
                days_posted=days_posted
            )
        else:
            vacancies = grabber.grab_vacancies(
                job_preferences,
                max_results=max_jobs,
                days_posted=days_posted
            )

        if not vacancies:
            search_cache[search_id] = {
                "status": "completed",
                "progress": "No jobs found",
                "result": {
                    "search_id": search_id,
                    "timestamp": datetime.now(),
                    "candidate_name": cv_profile.get('full_name', 'Unknown'),
                    "search_role": job_preferences['desired_role'],
                    "total_jobs_scraped": 0,
                    "total_matches": 0,
                    "execution_time": time.time() - start_time,
                    "matches": [],
                    "match_distribution": calculate_match_distribution([]),
                    "average_match_score": None
                }
            }
            return

        # Step 2: Match jobs
        search_cache[search_id]["progress"] = f"Matching {len(vacancies)} jobs with CV..."

        matches = matcher.match_jobs_batch(
            cv_profile,
            vacancies,
            min_score=min_match_score
        )

        execution_time = time.time() - start_time

        # Calculate average score
        avg_score = sum(m['overall_score'] for m in matches) / len(matches) if matches else None

        # Store results
        search_cache[search_id] = {
            "status": "completed",
            "progress": "Search completed",
            "result": {
                "search_id": search_id,
                "timestamp": datetime.now(),
                "candidate_name": cv_profile.get('full_name', 'Unknown'),
                "search_role": job_preferences['desired_role'],
                "total_jobs_scraped": len(vacancies),
                "total_matches": len(matches),
                "execution_time": execution_time,
                "matches": matches,
                "match_distribution": calculate_match_distribution(matches),
                "average_match_score": avg_score
            }
        }

    except Exception as e:
        logger.error(f"Error in job search {search_id}: {e}")
        search_cache[search_id] = {
            "status": "failed",
            "progress": f"Error: {str(e)}",
            "result": None
        }





@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


@router.post("/search", response_model=SearchStatusResponse)
async def search_jobs(request: JobSearchRequest, background_tasks: BackgroundTasks):
    """
    Asynchronous job search endpoint
    Returns immediately with a search ID, results can be fetched later
    """
    # Generate search ID
    search_id = str(uuid.uuid4())

    # Convert CV profile to dict
    cv_profile_dict = request.cv_profile.dict()

    # Build job preferences
    job_preferences = {
        'desired_role': request.role,
        'job_type': 'FULL_TIME',
        'remote_work_preference': 'FLEXIBLE'
    }

    # Initialize search status
    search_cache[search_id] = {
        "status": "queued",
        "progress": "Search queued",
        "result": None
    }

    # Add to background tasks
    background_tasks.add_task(
        run_job_search_async,
        search_id,
        cv_profile_dict,
        job_preferences,
        request.max_jobs,
        request.min_match_score,
        request.days_posted,
        request.use_async,
        request.matcher_model.value
    )

    return SearchStatusResponse(
        search_id=search_id,
        status="queued",
        progress="Search has been queued for processing"
    )


@router.get("/search/{search_id}", response_model=SearchStatusResponse)
async def get_search_results(search_id: str):
    """Get search results by ID"""
    if search_id not in search_cache:
        raise HTTPException(status_code=404, detail="Search ID not found")

    search_data = search_cache[search_id]

    return SearchStatusResponse(
        search_id=search_id,
        status=search_data["status"],
        progress=search_data["progress"],
        result=search_data["result"]
    )


@router.post("/search/sync", response_model=JobSearchResponse)
async def search_jobs_sync(request: JobSearchRequest):
    """
    Synchronous job search endpoint
    Waits for results and returns them immediately
    Warning: This can take 30-60 seconds for large searches
    """
    start_time = time.time()

    # Convert CV profile to dict
    cv_profile_dict = request.cv_profile.dict()

    # Build job preferences
    job_preferences = {
        'desired_role': request.role,
        'job_type': 'FULL_TIME',
        'remote_work_preference': 'FLEXIBLE'
    }

    try:
        # Initialize components
        grabber = FindAJobGrabber()
        matcher = GeneralJobMatcher(model_name=request.matcher_model.value)

        # Scrape jobs
        if request.use_async and request.max_jobs > 20:
            vacancies = grabber.grab_vacancies_async(
                job_preferences,
                max_results=request.max_jobs,
                days_posted=request.days_posted
            )
        else:
            vacancies = grabber.grab_vacancies(
                job_preferences,
                max_results=request.max_jobs,
                days_posted=request.days_posted
            )

        if not vacancies:
            return JobSearchResponse(
                search_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                candidate_name=cv_profile_dict.get('full_name', 'Unknown'),
                search_role=request.role,
                total_jobs_scraped=0,
                total_matches=0,
                execution_time=time.time() - start_time,
                matches=[],
                match_distribution=calculate_match_distribution([]),
                average_match_score=None
            )

        # Match jobs
        matches = matcher.match_jobs_batch(
            cv_profile_dict,
            vacancies,
            min_score=request.min_match_score
        )

        execution_time = time.time() - start_time

        # Calculate average score
        avg_score = sum(m['overall_score'] for m in matches) / len(matches) if matches else None

        return JobSearchResponse(
            search_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            candidate_name=cv_profile_dict.get('full_name', 'Unknown'),
            search_role=request.role,
            total_jobs_scraped=len(vacancies),
            total_matches=len(matches),
            execution_time=execution_time,
            matches=matches,
            match_distribution=calculate_match_distribution(matches),
            average_match_score=avg_score
        )

    except Exception as e:
        logger.error(f"Error in synchronous search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.delete("/search/{search_id}")
async def delete_search_results(search_id: str):
    """Delete search results from cache"""
    if search_id not in search_cache:
        raise HTTPException(status_code=404, detail="Search ID not found")

    del search_cache[search_id]
    return {"message": f"Search {search_id} deleted successfully"}
