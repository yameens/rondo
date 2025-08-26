from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Database - Supabase
    database_url: str = "postgresql://rondo:rondo@localhost:5432/rondo"
    
    # Redis - Railway/Render
    redis_url: str = "redis://localhost:6379/0"
    
    # Storage - Supabase
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    s3_bucket: str = "rondo-files"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_endpoint: Optional[str] = None
    s3_region: str = "us-east-1"
    
    # Audio Processing
    max_audio_duration: int = 600  # 10 minutes in seconds
    sample_rate: int = 44100
    onset_tolerance_ms: int = 500  # More realistic tolerance (0.5 seconds)
    pitch_tolerance: int = 1  # Allow 1 semitone difference
    
    # File Upload
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_audio_extensions: list = [".wav", ".mp3", ".flac"]
    allowed_score_extensions: list = [".xml", ".musicxml", ".mxl", ".mei", ".pdf"]
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Deployment
    environment: str = "development"  # development, staging, production
    debug: bool = True
    
    class Config:
        env_file = ".env"


settings = Settings()
