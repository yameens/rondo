# AI Piano Teacher - Docker Deployment Guide

## 🎹 Complete AI Music Teacher System

This is a comprehensive AI-powered piano teaching system with real-time performance analysis, expressive feedback, and interactive practice sessions.

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM available
- 2GB free disk space

### One-Command Startup
```bash
./start.sh
```

That's it! The script will:
- ✅ Check Docker availability
- ✅ Build all services
- ✅ Start the complete system
- ✅ Run health checks
- ✅ Display service URLs

### Manual Startup
```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🏗️ System Architecture

### Services
- **Frontend** (Port 3000): Next.js React application with clean, modern UI
- **Backend** (Port 8001): FastAPI with full audio processing capabilities
- **Database** (Port 5432): PostgreSQL for data persistence
- **Redis** (Port 6379): Celery task queue for background processing
- **Worker**: Celery worker for audio analysis tasks

### Technology Stack
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, SWR
- **Backend**: FastAPI, SQLAlchemy, Celery, librosa, music21
- **Database**: PostgreSQL with JSON fields for flexible data
- **Audio Processing**: librosa, soundfile, numpy, scipy
- **Containerization**: Docker & Docker Compose

## 🎯 Features

### 🎵 Core Functionality
- **Audio Upload & Analysis**: Upload piano performances for AI analysis
- **Real-time Feedback**: Tempo, pitch, dynamics, and timing analysis
- **Expressive Features**: Beat-wise tempo, loudness, articulation, pedal, balance
- **Reference Comparisons**: Compare against professional performances
- **Practice Sessions**: Structured, goal-oriented practice

### 🤖 AI Capabilities
- **Performance Scoring**: Overall and detailed performance metrics
- **Envelope Analysis**: Statistical comparison with reference performances
- **Personalized Recommendations**: AI-driven practice suggestions
- **Progress Tracking**: Long-term improvement analytics

### 💻 User Interface
- **Clean, Modern Design**: Inspired by professional music software
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Interactive Visualizations**: Feature curves, heatmaps, progress charts
- **Intuitive Navigation**: Easy-to-use interface for all skill levels

## 📊 API Endpoints

### Core APIs
- `GET /health` - Health check
- `GET /api/scores` - List available scores
- `POST /api/performances/student` - Upload student performance
- `POST /api/performances/reference` - Upload reference performance
- `GET /api/envelopes/{score_id}` - Get performance envelopes
- `POST /api/expressive-score/{perf_id}` - Get expressive analysis

### Documentation
- **Interactive API Docs**: http://localhost:8001/docs
- **OpenAPI Schema**: http://localhost:8001/openapi.json

## 🧪 Testing

### Automated API Testing
```bash
# Run comprehensive API tests
python3 test_apis.py

# Run tests with startup script
./start.sh --test
```

### Manual Testing
1. **Upload Performance**: http://localhost:3000/upload
2. **Browse Scores**: http://localhost:3000/scores
3. **Practice Sessions**: http://localhost:3000/practice
4. **API Documentation**: http://localhost:8001/docs

## 🔧 Configuration

### Environment Variables
Create `.env` file (or copy from `env.example`):
```bash
# Database
DATABASE_URL=postgresql://piano_user:piano_pass@db:5432/piano_analysis

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Audio Processing
MAX_FILE_SIZE_BYTES=20971520
MAX_DURATION_SECONDS=600
TARGET_SAMPLE_RATE=22050

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Volume Mounts
- `./backend:/app` - Backend code (hot reload)
- `./web:/app` - Frontend code (hot reload)
- `./scores:/app/scores` - MusicXML score files
- `audio_uploads:/app/uploads` - Uploaded audio files
- `postgres_data:/var/lib/postgresql/data` - Database persistence
- `redis_data:/data` - Redis persistence

## 📁 Project Structure

```
rondo.audiotoaudio/
├── docker-compose.yml          # Multi-service orchestration
├── start.sh                    # One-command startup script
├── test_apis.py               # Comprehensive API testing
├── backend/                   # FastAPI backend
│   ├── Dockerfile            # Backend container
│   ├── app/                  # Application code
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # Database models
│   │   ├── analysis.py      # Audio analysis engine
│   │   ├── services/        # Business logic
│   │   └── api/             # API routes
│   └── requirements.txt     # Python dependencies
├── web/                      # Next.js frontend
│   ├── Dockerfile           # Frontend container
│   ├── src/                 # Source code
│   │   ├── app/            # App router pages
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom hooks
│   │   └── lib/            # Utilities
│   └── package.json        # Node.js dependencies
└── api/                     # Legacy API (maintained for compatibility)
```

## 🎮 Usage Guide

### 1. Upload a Performance
1. Go to http://localhost:3000/upload
2. Select a score (e.g., "Chopin Nocturne Op. 9 No. 2")
3. Upload an audio file or record live
4. Click "Analyze Performance"
5. View detailed feedback and scores

### 2. Practice Sessions
1. Go to http://localhost:3000/practice
2. Review your practice goals
3. Select a recommended session
4. Follow structured practice guidance
5. Track your progress over time

### 3. Browse Scores
1. Go to http://localhost:3000/scores
2. Explore available pieces
3. View difficulty levels and details
4. Start practice sessions directly

## 🔍 Monitoring & Debugging

### Service Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f worker
```

### Health Checks
```bash
# Backend API
curl http://localhost:8001/health

# Frontend
curl http://localhost:3000

# Database
docker-compose exec db pg_isready -U piano_user -d piano_analysis
```

### Performance Monitoring
- **Backend**: Built-in FastAPI metrics
- **Database**: PostgreSQL query logs
- **Celery**: Task monitoring via Redis
- **Frontend**: Next.js built-in analytics

## 🛠️ Development

### Hot Reload
Both frontend and backend support hot reload:
- **Backend**: Code changes in `./backend` automatically reload
- **Frontend**: Code changes in `./web` trigger rebuild

### Adding New Features
1. **Backend**: Add routes in `backend/app/api/`
2. **Frontend**: Add pages in `web/src/app/`
3. **Database**: Update models in `backend/app/models.py`
4. **Tests**: Add tests in `test_apis.py`

### Database Migrations
```bash
# Access database
docker-compose exec db psql -U piano_user -d piano_analysis

# View tables
docker-compose exec backend python -c "from app.database import engine; print(engine.table_names())"
```

## 🚨 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker is running
docker info

# Check port conflicts
netstat -tulpn | grep :3000
netstat -tulpn | grep :8001

# Rebuild containers
docker-compose down
docker-compose up --build
```

#### Database Connection Issues
```bash
# Check database logs
docker-compose logs db

# Reset database
docker-compose down -v
docker-compose up -d db
```

#### Audio Processing Errors
```bash
# Check worker logs
docker-compose logs worker

# Restart worker
docker-compose restart worker
```

### Performance Issues
- **Slow Audio Processing**: Increase worker concurrency
- **Memory Usage**: Monitor with `docker stats`
- **Disk Space**: Clean up with `docker system prune`

## 📈 Production Deployment

### Security Considerations
- Change default passwords in `.env`
- Use SSL certificates for HTTPS
- Configure firewall rules
- Enable database backups
- Monitor system resources

### Scaling
- **Horizontal**: Add more worker containers
- **Vertical**: Increase container resources
- **Database**: Use managed PostgreSQL service
- **Storage**: Use object storage for audio files

## 🎯 Next Steps

1. **Add More Scores**: Place MusicXML files in `./scores/`
2. **Customize UI**: Modify components in `web/src/components/`
3. **Extend Analysis**: Add features in `backend/app/analysis.py`
4. **Deploy Production**: Use Docker Swarm or Kubernetes

---

## 🆘 Support

- **Logs**: Check service logs for detailed error information
- **API Docs**: Visit http://localhost:8001/docs for API reference
- **Testing**: Run `python3 test_apis.py` for system validation

**Your AI Piano Teacher is ready to help students improve their musical skills!** 🎹✨
