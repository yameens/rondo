"""
Celery worker application.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from app.services.jobs import celery_app

# Import tasks to register them
from app.services.jobs import (
    analyze_reference_performance,
    analyze_student_performance,
    process_audio_file,
    cleanup_temp_files
)

if __name__ == '__main__':
    celery_app.start()
