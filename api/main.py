from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional
import tempfile
import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
import sys
import re
import librosa
import yt_dlp
import time
import glob

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import our analysis modules
try:
    from score import parse_score_musicxml
    from audio import load_audio_feats
    from align import score_beatchroma_from_events, dtw_align, pitch_to_chroma
except ImportError:
    # Fallback for legacy analysis
    parse_score_musicxml = None
    load_audio_feats = None

# Import new analysis module
try:
    from app.analysis import analyze_audio_pair
    from app.utils import standardize_audio_to_wav, process_youtube_url
    NEW_ANALYSIS_AVAILABLE = True
except ImportError:
    analyze_audio_pair = None
    standardize_audio_to_wav = None
    process_youtube_url = None
    NEW_ANALYSIS_AVAILABLE = False

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/piano_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# TTL Cleanup function
def cleanup_old_audio_files():
    """Clean up audio files older than 2 hours from /tmp directory."""
    try:
        current_time = time.time()
        ttl_seconds = 2 * 60 * 60  # 2 hours
        
        # Look for patterns like piano_*.wav, report_*.csv, report_*.json
        patterns = [
            '/tmp/piano_*.wav',
            '/tmp/report_*.csv', 
            '/tmp/report_*.json',
            '/tmp/*_analysis_issues.csv'
        ]
        
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    # Check if file is older than TTL
                    if os.path.exists(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > ttl_seconds:
                            os.remove(file_path)
                            logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
        
        logger.info("Audio file cleanup completed")
    except Exception as e:
        logger.error(f"Error during audio file cleanup: {e}")

# Validation constants
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB
MAX_DURATION_SECONDS = 600  # 10 minutes
ALLOWED_AUDIO_EXTENSIONS = ['.mp3']
ALLOWED_CONTENT_TYPES = ['audio/mpeg', 'audio/mp3']

# YouTube URL regex pattern
YOUTUBE_URL_PATTERN = re.compile(
    r'^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)[\w-]+(&[\w=]*)?$'
)

def create_error_response(code: str, message: str, status_code: int = 400) -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}}
    )

def validate_audio_file(file: UploadFile, file_type: str = "audio") -> None:
    """
    Validate audio file format, content type, and size.
    
    Args:
        file: The uploaded file
        file_type: Description of file type for error messages
        
    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BadRequest", "message": f"{file_type} file is required"}}
        )
    
    # Check file extension
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BadRequest", "message": f"{file_type} file must be MP3 format (received: {file.filename})"}}
        )
    
    # Check content type
    content_type = getattr(file, 'content_type', None)
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BadRequest", "message": f"{file_type} file must have audio/mpeg content type (received: {content_type})"}}
        )
    
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE_BYTES:
        size_mb = file.size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail={"error": {"code": "TooLarge", "message": f"{file_type} file too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_BYTES/(1024*1024):.0f}MB)"}}
        )

def validate_audio_duration(file_path: str, file_type: str = "Audio") -> None:
    """
    Validate audio file duration.
    
    Args:
        file_path: Path to the audio file
        file_type: Description of file type for error messages
        
    Raises:
        HTTPException: If duration exceeds limit
    """
    try:
        duration = librosa.get_duration(path=file_path)
        if duration > MAX_DURATION_SECONDS:
            duration_min = duration / 60
            max_min = MAX_DURATION_SECONDS / 60
            raise HTTPException(
                status_code=413,
                detail={"error": {"code": "TooLarge", "message": f"{file_type} too long: {duration_min:.1f} minutes (max: {max_min:.0f} minutes)"}}
            )
    except Exception as e:
        if "duration" in str(e).lower() or "too long" in str(e).lower():
            raise  # Re-raise duration-related errors
        logger.warning(f"Could not validate duration for {file_path}: {e}")

def validate_youtube_url(url: str) -> None:
    """
    Validate YouTube URL format and availability.
    
    Args:
        url: YouTube URL to validate
        
    Raises:
        HTTPException: If URL is invalid or video unavailable
    """
    if not url or not url.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BadRequest", "message": "YouTube URL is required"}}
        )
    
    # Basic format validation
    if not YOUTUBE_URL_PATTERN.match(url.strip()):
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "BadRequest", "message": "Invalid YouTube URL format. Please provide a valid YouTube video link."}}
        )
    
    # Test availability with yt-dlp simulation
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'simulate': True,
            'format': 'bestaudio',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Check duration if available
            if 'duration' in info and info['duration']:
                if info['duration'] > MAX_DURATION_SECONDS:
                    duration_min = info['duration'] / 60
                    max_min = MAX_DURATION_SECONDS / 60
                    raise HTTPException(
                        status_code=413,
                        detail={"error": {"code": "TooLarge", "message": f"YouTube video too long: {duration_min:.1f} minutes (max: {max_min:.0f} minutes)"}}
                    )
                    
    except yt_dlp.DownloadError as e:
        error_msg = str(e)
        if "private" in error_msg.lower() or "unavailable" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "BadRequest", "message": "YouTube video is private, unavailable, or restricted."}}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "BadRequest", "message": f"Cannot access YouTube video: {error_msg}"}}
            )
    except Exception as e:
        logger.error(f"YouTube validation error for {url}: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "ProcessingError", "message": "Failed to validate YouTube URL. Please check the link and try again."}}
        )

app = FastAPI(title="Piano Performance Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for audio preview
app.mount("/static", StaticFiles(directory="/tmp"), name="static")

# Startup event to clean old files
@app.on_event("startup")
async def startup_event():
    """Run cleanup on startup."""
    cleanup_old_audio_files()

# Job storage (in production, use Redis or database)
job_storage: Dict[str, Dict[str, Any]] = {}

# Job status constants
JOB_STATUS_PENDING = "pending"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

async def process_analysis_task(
    job_id: str,
    user_wav_path: str,
    ref_wav_path: str,
    is_youtube_ref: bool = False
):
    """
    Background task to process audio analysis.
    
    Args:
        job_id: Unique job identifier
        user_wav_path: Path to user audio file (always a file)
        ref_wav_path: Path to reference audio file or YouTube URL
        is_youtube_ref: Whether ref_wav_path is a YouTube URL
    """
    logger.info(f"Starting analysis task for job {job_id}")
    
    try:
        # Update job status
        job_storage[job_id]["status"] = JOB_STATUS_PROCESSING
        job_storage[job_id]["started_at"] = datetime.now().isoformat()
        
        # Process reference audio if it's a YouTube URL
        final_ref_path = ref_wav_path
        if is_youtube_ref and NEW_ANALYSIS_AVAILABLE:
            logger.info(f"Processing YouTube URL for reference for job {job_id}")
            final_ref_path, duration = process_youtube_url(ref_wav_path)
            logger.info(f"YouTube reference audio processed: {duration:.2f}s duration")
        
        # User audio is always a file
        final_user_path = user_wav_path
        
        # Create standardized audio files for preview
        import shutil
        user_preview_filename = f"piano_user_{job_id}.wav"
        ref_preview_filename = f"piano_ref_{job_id}.wav"
        user_preview_path = f"/tmp/{user_preview_filename}"
        ref_preview_path = f"/tmp/{ref_preview_filename}"
        
        # Copy standardized files to predictable names for serving
        try:
            shutil.copy2(final_user_path, user_preview_path)
            shutil.copy2(final_ref_path, ref_preview_path)
            logger.info(f"Audio preview files created: {user_preview_path}, {ref_preview_path}")
        except Exception as e:
            logger.warning(f"Failed to create audio preview files: {e}")
            user_preview_path = None
            ref_preview_path = None
        
        # Run the analysis
        if NEW_ANALYSIS_AVAILABLE:
            logger.info(f"Running new analysis for job {job_id}")
            result = analyze_audio_pair(final_user_path, final_ref_path)
            logger.info(f"Analysis completed for job {job_id}")
        else:
            logger.warning(f"New analysis not available for job {job_id}, using placeholder")
            result = {
                "summary": {"message": "New analysis module not available"},
                "per_frame": {},
                "issues": [],
                "midi_estimates": {"available": False}
            }
        
        # Add audio preview URLs to result
        if user_preview_path and os.path.exists(user_preview_path):
            result["user_audio_url"] = f"/static/{user_preview_filename}"
        if ref_preview_path and os.path.exists(ref_preview_path):
            result["ref_audio_url"] = f"/static/{ref_preview_filename}"
        
        # Save results to files
        report_json_path = f"/tmp/report_{job_id}.json"
        report_csv_path = f"/tmp/report_{job_id}.csv"
        
        # Save JSON report
        with open(report_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"JSON report saved: {report_json_path}")
        
        # Save CSV report (issues)
        if NEW_ANALYSIS_AVAILABLE and result.get('issues'):
            from app.analysis import save_issues_to_csv
            save_issues_to_csv(result['issues'], report_csv_path)
            logger.info(f"CSV report saved: {report_csv_path}")
        else:
            # Create empty CSV if no issues
            with open(report_csv_path, 'w') as f:
                f.write("start_s,end_s,type,severity,note\n")
            logger.info(f"Empty CSV report created: {report_csv_path}")
        
        # Update job as completed
        job_storage[job_id].update({
            "status": JOB_STATUS_COMPLETED,
            "completed_at": datetime.now().isoformat(),
            "result": result,
            "json_path": report_json_path,
            "csv_path": report_csv_path
        })
        
        logger.info(f"Analysis task completed successfully for job {job_id}")
        
    except Exception as e:
        logger.error(f"Analysis task failed for job {job_id}: {str(e)}", exc_info=True)
        
        # Update job as failed
        job_storage[job_id].update({
            "status": JOB_STATUS_FAILED,
            "failed_at": datetime.now().isoformat(),
            "error": str(e)
        })
    
    finally:
        # Clean up temporary files
        if is_youtube_ref and final_ref_path != ref_wav_path:
            try:
                if os.path.exists(final_ref_path):
                    os.unlink(final_ref_path)
                logger.info(f"Cleaned up temporary reference file: {final_ref_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary reference file {final_ref_path}: {e}")


@app.get("/")
async def root():
    return {"message": "Piano Performance Analysis API"}

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    user_audio: UploadFile = File(..., description="User's performance audio file (.wav, .mp3)"),
    reference_audio: UploadFile = File(..., description="Reference performance audio file (.wav, .mp3) or dummy if youtube_url provided"),
    youtube_url: Optional[str] = Form(None, description="YouTube URL as alternative to reference_audio file")
):
    """
    Start asynchronous analysis of piano performance.
    
    Args:
        user_audio: User's performance audio file (required)
        reference_audio: Reference performance audio file (ignored if youtube_url provided)
        youtube_url: Optional YouTube URL for reference performance
    
    Returns:
        Job ID for polling results (HTTP 202)
    """
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Starting new analysis job {job_id}")
    
    try:
        # Validate user audio (always required)
        if not user_audio or not user_audio.filename:
            return create_error_response(
                "BadRequest",
                "User audio file is required"
            )
        validate_audio_file(user_audio, "User audio")
        
        # Validate reference source (either file or YouTube URL)
        if youtube_url:
            if not NEW_ANALYSIS_AVAILABLE:
                logger.error(f"YouTube URL provided but new analysis not available for job {job_id}")
                return create_error_response(
                    "ProcessingError",
                    "YouTube URL processing requires new analysis module",
                    500
                )
            
            # Validate YouTube URL
            validate_youtube_url(youtube_url)
            ref_source = youtube_url
            is_youtube_ref = True
            logger.info(f"Job {job_id} using YouTube URL for reference: {youtube_url}")
        else:
            # Validate reference audio file
            if not reference_audio or not reference_audio.filename:
                return create_error_response(
                    "BadRequest",
                    "Either reference_audio file or youtube_url must be provided"
                )
            
            validate_audio_file(reference_audio, "Reference audio")
            is_youtube_ref = False
        
        # Handle reference audio processing
        if is_youtube_ref:
            ref_source = youtube_url
            # Reference will be processed in background task
        else:
            # Save reference audio to temp file
            ref_temp_path = f"/tmp/ref_{job_id}.wav"
            
            if NEW_ANALYSIS_AVAILABLE:
                # Use new audio processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                    ref_content = await reference_audio.read()
                    
                    # Validate file size
                    if len(ref_content) > MAX_FILE_SIZE_BYTES:
                        size_mb = len(ref_content) / (1024 * 1024)
                        os.unlink(tmp_file.name)
                        return create_error_response(
                            "TooLarge",
                            f"Reference audio file too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_BYTES/(1024*1024):.0f}MB)",
                            413
                        )
                    
                    tmp_file.write(ref_content)
                    tmp_file.flush()
                    
                    # Standardize reference audio
                    ref_temp_path, _ = standardize_audio_to_wav(tmp_file.name, ref_temp_path)
                    os.unlink(tmp_file.name)
                    
                    # Validate duration
                    validate_audio_duration(ref_temp_path, "Reference audio")
            else:
                # Fallback: save as-is
                ref_content = await reference_audio.read()
                
                # Validate file size
                if len(ref_content) > MAX_FILE_SIZE_BYTES:
                    size_mb = len(ref_content) / (1024 * 1024)
                    return create_error_response(
                        "TooLarge",
                        f"Reference audio file too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_BYTES/(1024*1024):.0f}MB)",
                        413
                    )
                
                with open(ref_temp_path, 'wb') as f:
                    f.write(ref_content)
                    
                # Validate duration
                validate_audio_duration(ref_temp_path, "Reference audio")
            
            ref_source = ref_temp_path
        
        # Handle user audio (always a file upload)
        user_temp_path = f"/tmp/user_{job_id}.wav"
        
        if NEW_ANALYSIS_AVAILABLE:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                user_content = await user_audio.read()
                
                # Validate file size
                if len(user_content) > MAX_FILE_SIZE_BYTES:
                    size_mb = len(user_content) / (1024 * 1024)
                    os.unlink(tmp_file.name)
                    return create_error_response(
                        "TooLarge",
                        f"User audio file too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_BYTES/(1024*1024):.0f}MB)",
                        413
                    )
                
                tmp_file.write(user_content)
                tmp_file.flush()
                
                # Standardize user audio
                user_temp_path, _ = standardize_audio_to_wav(tmp_file.name, user_temp_path)
                os.unlink(tmp_file.name)
                
                # Validate duration
                validate_audio_duration(user_temp_path, "User audio")
        else:
            user_content = await user_audio.read()
            
            # Validate file size
            if len(user_content) > MAX_FILE_SIZE_BYTES:
                size_mb = len(user_content) / (1024 * 1024)
                return create_error_response(
                    "TooLarge",
                    f"User audio file too large: {size_mb:.1f}MB (max: {MAX_FILE_SIZE_BYTES/(1024*1024):.0f}MB)",
                    413
                )
            
            with open(user_temp_path, 'wb') as f:
                f.write(user_content)
                
            # Validate duration
            validate_audio_duration(user_temp_path, "User audio")
        
        user_source = user_temp_path
        
        # Initialize job record
        job_storage[job_id] = {
            "job_id": job_id,
            "status": JOB_STATUS_PENDING,
            "created_at": datetime.now().isoformat(),
            "user_source": os.path.basename(user_source),
            "reference_source": ref_source if is_youtube_ref else os.path.basename(ref_source),
            "is_youtube_ref": is_youtube_ref
        }
        
        # Queue background task
        background_tasks.add_task(
            process_analysis_task,
            job_id=job_id,
            user_wav_path=user_source,
            ref_wav_path=ref_source,
            is_youtube_ref=is_youtube_ref
        )
        
        logger.info(f"Analysis job {job_id} queued successfully")
        
        # Return 202 Accepted with job ID
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": JOB_STATUS_PENDING,
                "message": "Analysis started. Use GET /analyze/{job_id} to check progress."
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to start analysis job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")


@app.get("/analyze/{job_id}")
async def get_analysis_result(job_id: str):
    """
    Get analysis results for a job.
    
    Args:
        job_id: Job identifier from POST /analyze
        
    Returns:
        Job status and results (if completed)
    """
    logger.info(f"Checking status for job {job_id}")
    
    try:
        # Check if job exists
        if job_id not in job_storage:
            logger.warning(f"Job {job_id} not found")
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = job_storage[job_id]
        status = job["status"]
        
        # Base response
        response = {
            "job_id": job_id,
            "status": status,
            "created_at": job["created_at"]
        }
        
        if status == JOB_STATUS_PENDING:
            response["message"] = "Analysis is queued"
            
        elif status == JOB_STATUS_PROCESSING:
            response["message"] = "Analysis is in progress"
            response["started_at"] = job.get("started_at")
            
        elif status == JOB_STATUS_COMPLETED:
            response["message"] = "Analysis completed successfully"
            response.update({
                "started_at": job.get("started_at"),
                "completed_at": job["completed_at"],
                "result": job["result"],
                "csv_url": f"/tmp/report_{job_id}.csv"
            })
            
        elif status == JOB_STATUS_FAILED:
            response["message"] = "Analysis failed"
            response.update({
                "failed_at": job["failed_at"],
                "error": job["error"]
            })
        
        logger.info(f"Job {job_id} status: {status}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking job status: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
