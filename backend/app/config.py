"""
Configuration settings for the application.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = "sqlite:///./piano_analysis.db"
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_accept_content: list = ["json"]
    celery_timezone: str = "UTC"
    celery_enable_utc: bool = True
    
    # Audio processing
    max_audio_file_size: int = 100 * 1024 * 1024  # 100MB
    audio_temp_dir: str = "/tmp/audio_processing"
    
    # Feature extraction
    target_sr: int = 22050
    hop_length: int = 512
    
    class Config:
        env_file = ".env"


settings = Settings()


def get_celery_config():
    """Get Celery configuration dictionary."""
    return {
        'broker_url': settings.celery_broker_url,
        'result_backend': settings.celery_result_backend,
        'task_serializer': settings.celery_task_serializer,
        'result_serializer': settings.celery_result_serializer,
        'accept_content': settings.celery_accept_content,
        'timezone': settings.celery_timezone,
        'enable_utc': settings.celery_enable_utc,
        'task_track_started': True,
        'task_time_limit': 30 * 60,  # 30 minutes
        'task_soft_time_limit': 25 * 60,  # 25 minutes
        'worker_prefetch_multiplier': 1,
        'task_acks_late': True,
        'worker_disable_rate_limits': False,
        'task_compression': 'gzip',
        'result_compression': 'gzip',
        'result_expires': 3600,  # 1 hour
    }
