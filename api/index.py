"""
Vercel serverless function entry point for the FastAPI backend.
This file serves as the main entry point for all API routes on Vercel.
"""
import os
import sys

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend'))

from backend.app.main import app

# Vercel expects the ASGI app to be available as 'app'
# This is the entry point for Vercel's Python runtime
