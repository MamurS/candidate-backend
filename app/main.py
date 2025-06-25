import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .routers import auth_router, cv_router, work_preferences, job_matches

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="FastAPI Auth System",
    description="Authentication system with email OTP verification",
    version="1.0.0"
)

# Add CORS middleware
origins = [
    "http://localhost:3000",  # React default
    "http://localhost:8000",  # FastAPI default
    "http://localhost:5173",  # Vite default
]

# Add production origins from environment variable if available
if os.getenv("FRONTEND_URL"):
    origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router.router)
app.include_router(cv_router.router)
app.include_router(work_preferences.router)
app.include_router(job_matches.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI Auth System"}

@app.get("/health")
def health_check():
    try:
        # Test database connection
        from .database import engine
        from sqlalchemy import text
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }