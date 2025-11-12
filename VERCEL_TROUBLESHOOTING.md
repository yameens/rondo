# Vercel Deployment Troubleshooting Guide

## 🚨 Common Errors and Fixes

### 1. **FUNCTION_INVOCATION_FAILED** (500 Error)

**Cause**: Python import errors or missing dependencies

**Fix**: 
```bash
# Check the simplified API is working
curl https://your-app.vercel.app/api/health

# If that fails, check the minimal mode
curl https://your-app.vercel.app/
```

**Solution**: The project now has fallback modes:
1. **Full Backend** (with audio processing) - may fail on Vercel
2. **Simplified Backend** (basic API only) - Vercel compatible  
3. **Minimal Mode** (error reporting) - always works

### 2. **FUNCTION_INVOCATION_TIMEOUT** (504 Error)

**Cause**: Heavy audio processing libraries taking too long to import

**Fix**: Use the simplified backend without librosa/scipy
- ✅ Current setup automatically falls back to simplified mode
- ✅ Audio processing is disabled for Vercel compatibility

### 3. **FUNCTION_PAYLOAD_TOO_LARGE** (413 Error)

**Cause**: Audio files too large for Vercel functions

**Fix**: Reduce file size limits
```bash
# In Vercel environment variables, set:
MAX_FILE_SIZE_BYTES=5242880  # 5MB instead of 20MB
```

### 4. **DEPLOYMENT_NOT_READY_REDIRECTING** (303 Error)

**Cause**: Deployment still building

**Fix**: Wait for deployment to complete, then try again

### 5. **NO_RESPONSE_FROM_FUNCTION** (502 Error)

**Cause**: Function crashed during startup

**Fix**: Check Vercel function logs:
```bash
vercel logs --follow
```

## 🔧 Quick Fixes Applied

### ✅ **Fixed vercel.json Configuration**
```json
{
  "buildCommand": "cd web && npm run build",
  "outputDirectory": "web/.next", 
  "installCommand": "cd web && npm install",
  "functions": {
    "api/index.py": {
      "runtime": "python3.9"
    }
  },
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "/api/index.py"
    }
  ]
}
```

### ✅ **Simplified Dependencies**
Removed heavy libraries that cause Vercel timeouts:
- ❌ `librosa` (large audio processing library)
- ❌ `scipy` (scientific computing - too heavy)
- ❌ `music21` (music analysis - complex dependencies)
- ❌ `celery` (not needed in serverless)

Kept essential libraries:
- ✅ `fastapi` (web framework)
- ✅ `sqlalchemy` (database)
- ✅ `numpy` (basic math)
- ✅ `pydantic` (data validation)

### ✅ **Fallback Import System**
```python
try:
    from backend.app.main import app  # Full backend
except ImportError:
    try:
        from simple_main import app   # Simplified backend
    except ImportError:
        # Minimal FastAPI app for debugging
```

### ✅ **Simplified API Endpoints**
Working endpoints in simplified mode:
- `GET /` - Status and info
- `GET /health` - Health check
- `GET /api/scores` - List scores (mock data)
- `POST /api/performances/upload` - Upload audio
- `GET /api/analysis/{id}` - Mock analysis results
- `GET /api/envelopes/{score_id}` - Mock envelope data

## 🧪 Testing Your Deployment

### 1. **Check Basic Functionality**
```bash
# Test root endpoint
curl https://your-app.vercel.app/

# Test health check
curl https://your-app.vercel.app/health

# Test API info
curl https://your-app.vercel.app/api/info
```

### 2. **Test API Endpoints**
```bash
# List scores
curl https://your-app.vercel.app/api/scores

# Get specific score
curl https://your-app.vercel.app/api/scores/1

# Test file upload (with a small audio file)
curl -X POST https://your-app.vercel.app/api/performances/upload \
  -F "score_id=1" \
  -F "role=student" \
  -F "source=test" \
  -F "audio=@test.mp3"
```

### 3. **Check Frontend**
```bash
# Test frontend build
curl https://your-app.vercel.app/

# Test API integration
# Open browser developer tools and check network requests
```

## 🔍 Debugging Steps

### 1. **Check Vercel Logs**
```bash
# Install Vercel CLI
npm i -g vercel

# View logs
vercel logs --follow

# View specific function logs
vercel logs api/index.py
```

### 2. **Check Environment Variables**
```bash
# List environment variables
vercel env ls

# Add missing variables
vercel env add DATABASE_URL
```

### 3. **Local Testing**
```bash
# Test simplified backend locally
cd api
python simple_main.py

# Test in browser
open http://localhost:8000/docs
```

## 🚀 Deployment Commands

### **Redeploy with Fixes**
```bash
# From project root
vercel --prod

# Force new deployment
vercel --prod --force
```

### **Environment Setup**
```bash
# Set required environment variables
vercel env add DATABASE_URL
# Enter your PostgreSQL connection string

# Optional: Set file size limits
vercel env add MAX_FILE_SIZE_BYTES
# Enter: 5242880 (for 5MB limit)
```

## 📊 Expected Behavior

### ✅ **Working Deployment Should Show:**

1. **Root endpoint** (`/`):
```json
{
  "message": "Piano Performance Analysis API - Vercel Deployment",
  "version": "1.0.0", 
  "status": "running",
  "docs": "/docs"
}
```

2. **Health check** (`/health`):
```json
{
  "status": "healthy",
  "message": "API is running successfully on Vercel",
  "environment": "production"
}
```

3. **API info** (`/api/info`):
```json
{
  "deployment": {
    "platform": "Vercel",
    "runtime": "Python 3.9"
  },
  "features": {
    "audio_upload": true,
    "basic_analysis": true,
    "full_audio_processing": false
  }
}
```

## 🆘 If All Else Fails

### **Emergency Minimal Deployment**
If the API still fails, the system will fall back to a minimal mode that:
- ✅ Always works on Vercel
- ✅ Shows error information
- ✅ Helps debug the issue
- ✅ Provides basic endpoints

### **Contact Information**
If you're still having issues:
1. Check the Vercel function logs: `vercel logs`
2. Test the endpoints listed above
3. Verify environment variables are set
4. Try redeploying: `vercel --prod --force`

## 📈 Next Steps After Successful Deployment

1. **Add Database**: Set up PostgreSQL and add `DATABASE_URL`
2. **Test File Uploads**: Try uploading small audio files
3. **Monitor Performance**: Check Vercel analytics
4. **Scale Up**: Add more features as needed

The current setup prioritizes **getting deployed successfully** over full functionality. Once deployed, you can gradually add back features and optimize performance.
