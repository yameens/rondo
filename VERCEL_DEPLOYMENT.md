# Vercel Deployment Guide

This guide explains how to deploy the Piano Performance Analysis system to Vercel as a full-stack application.

## 🚀 Quick Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/bruceruan07/rondo.audiotoaudio)

## 📋 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Database**: PostgreSQL database (recommended: Vercel Postgres, Supabase, or PlanetScale)

## 🏗️ Architecture Overview

The deployment consists of:
- **Frontend**: Next.js app served from `/web` directory
- **Backend API**: FastAPI serverless functions from `/api` directory
- **Database**: PostgreSQL for production data storage
- **File Storage**: Vercel's temporary file system for audio processing

## 📦 Project Structure

```
rondo.audiotoaudio/
├── vercel.json              # Vercel configuration
├── api/                     # Backend API (serverless functions)
│   ├── index.py            # Main FastAPI entry point
│   └── requirements.txt    # Python dependencies
├── backend/                 # Backend application code
│   └── app/                # FastAPI application
├── web/                    # Frontend Next.js application
│   ├── package.json        # Node.js dependencies
│   └── next.config.js      # Next.js configuration
└── env.example             # Environment variables template
```

## 🔧 Setup Instructions

### 1. Database Setup

#### Option A: Vercel Postgres (Recommended)

1. Go to your Vercel dashboard
2. Create a new Postgres database
3. Copy the connection string

#### Option B: External Database (Supabase, PlanetScale, etc.)

1. Create a PostgreSQL database
2. Get the connection string in the format:
   ```
   postgresql://username:password@host:port/database
   ```

### 2. Environment Variables

In your Vercel project settings, add these environment variables:

#### Required Variables

```bash
# Database
DATABASE_URL=postgresql://username:password@host:port/database

# Frontend API URL (automatically set by Vercel)
NEXT_PUBLIC_API_URL=https://your-app.vercel.app/api
```

#### Optional Variables

```bash
# Audio Processing
AUDIO_TEMP_DIR=/tmp
MAX_FILE_SIZE_BYTES=20971520
MAX_DURATION_SECONDS=600
TARGET_SAMPLE_RATE=22050

# Celery (not used in serverless, but kept for compatibility)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### 3. Deploy to Vercel

#### Method 1: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy from project root
cd /path/to/rondo.audiotoaudio
vercel

# Follow the prompts:
# - Link to existing project or create new
# - Set up environment variables
# - Deploy
```

#### Method 2: GitHub Integration

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository
3. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `web`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
4. Add environment variables
5. Deploy

### 4. Database Initialization

After deployment, the database tables will be created automatically on first API request. You can also manually initialize:

1. Visit `https://your-app.vercel.app/api/db/tables` to verify tables exist
2. Use the `/api/health` endpoint to check database connectivity

## 🔄 Development Workflow

### Local Development

```bash
# Install dependencies
cd web && npm install
cd ../api && pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with your local database URL

# Run frontend (from web directory)
npm run dev

# Run backend (from project root)
cd backend && python -m uvicorn app.main:app --reload --port 8001
```

### Testing Before Deploy

```bash
# Build frontend
cd web && npm run build

# Test API locally
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

# Test full integration
curl http://localhost:3000/api/health
```

## 📊 Monitoring & Debugging

### Vercel Dashboard

- **Functions**: Monitor API performance and errors
- **Analytics**: Track frontend performance
- **Logs**: View real-time application logs

### Health Checks

- **API Health**: `GET /api/health`
- **Database**: `GET /api/db/tables`
- **Frontend**: Visit your domain

### Common Issues

#### 1. Database Connection Errors

```bash
# Check environment variables
vercel env ls

# Test database URL format
# Should be: postgresql://user:pass@host:port/db
```

#### 2. Import Errors

```bash
# Check Python path in api/index.py
# Ensure backend modules are properly imported
```

#### 3. File Upload Issues

```bash
# Check file size limits in environment
MAX_FILE_SIZE_BYTES=20971520  # 20MB
```

#### 4. CORS Issues

```bash
# Backend automatically allows all origins
# Check frontend API_BASE_URL configuration
```

## 🚀 Performance Optimization

### Backend

- **Cold Starts**: First request may be slow (~2-3s)
- **Memory**: Functions have 1GB RAM limit
- **Timeout**: 10-second execution limit for Hobby plan

### Frontend

- **Static Generation**: Most pages are statically generated
- **Image Optimization**: Automatic with Next.js
- **Bundle Size**: Optimized with tree shaking

### Database

- **Connection Pooling**: Use connection pooling for better performance
- **Indexes**: Ensure proper indexes on frequently queried columns

## 🔐 Security Considerations

### Environment Variables

- Never commit `.env` files
- Use Vercel's environment variable system
- Rotate database credentials regularly

### API Security

- CORS is configured for all origins (adjust for production)
- File upload validation is in place
- SQL injection protection via SQLAlchemy ORM

### Database

- Use SSL connections (enabled by default)
- Regular backups (automatic with Vercel Postgres)
- Monitor for unusual activity

## 📈 Scaling Considerations

### Current Limits

- **Vercel Functions**: 10-second timeout, 1GB RAM
- **File Uploads**: 20MB limit (configurable)
- **Database**: Depends on your provider's plan

### Scaling Options

1. **Upgrade Vercel Plan**: Higher limits and better performance
2. **Database Scaling**: Upgrade database plan as needed
3. **CDN**: Vercel's global CDN handles static assets
4. **Background Jobs**: Consider external job queue for heavy processing

## 🆘 Support & Troubleshooting

### Logs

```bash
# View function logs
vercel logs

# View build logs
vercel logs --build
```

### Debug Mode

Add to environment variables:
```bash
DEBUG=1
LOG_LEVEL=DEBUG
```

### Common Commands

```bash
# Redeploy
vercel --prod

# Check deployment status
vercel ls

# View environment variables
vercel env ls

# Pull environment to local
vercel env pull .env.local
```

## 🎯 Next Steps

After successful deployment:

1. **Custom Domain**: Add your custom domain in Vercel settings
2. **Analytics**: Enable Vercel Analytics for insights
3. **Monitoring**: Set up uptime monitoring
4. **Backups**: Configure database backup strategy
5. **CI/CD**: Set up automated testing and deployment

## 📚 Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Deployment](https://nextjs.org/docs/deployment)
- [FastAPI on Vercel](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [Vercel Postgres](https://vercel.com/docs/storage/vercel-postgres)

---

**Need Help?** Check the [GitHub Issues](https://github.com/bruceruan07/rondo.audiotoaudio/issues) or create a new issue for deployment-specific problems.
