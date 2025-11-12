"""
Vercel serverless function entry point for the FastAPI backend.
This file serves as the main entry point for all API routes on Vercel.
"""
import os
import sys
from pathlib import Path

# Get the project root directory
current_dir = Path(__file__).parent
project_root = current_dir.parent
backend_dir = project_root / "backend"

# Add paths to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

try:
    # Try to import the full backend first
    from backend.app.main import app
    print("✓ Successfully imported full backend")
except ImportError as e:
    print(f"⚠️ Full backend import failed: {e}")
    try:
        # Fallback to simplified version
        from simple_main import app
        print("✓ Using simplified backend for Vercel")
    except ImportError as e2:
        print(f"❌ Simplified backend import failed: {e2}")
        # Last resort: create minimal FastAPI app
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="Piano Analysis API - Minimal Mode")
        
        @app.get("/")
        async def root():
            return JSONResponse({
                "status": "minimal_mode",
                "message": "Running in minimal mode due to import errors",
                "original_error": str(e),
                "fallback_error": str(e2),
                "paths": sys.path[:5]
            })
        
        @app.get("/health")
        async def health():
            return JSONResponse({
                "status": "limited", 
                "mode": "minimal",
                "errors": [str(e), str(e2)]
            })

# Vercel expects the ASGI app to be available as 'app'
