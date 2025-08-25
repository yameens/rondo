# Rondo - Classical Performance Analysis Tool

A comprehensive web application that analyzes where a classical musician's performance diverges from the score using advanced AI transcription and alignment algorithms.

## 🎵 Features

- **Score Ingestion**: MusicXML/MXL/MEI parsing, PDF OMR conversion using Audiveris
- **Audio Processing**: AI-powered transcription from WAV/MP3 to symbolic notation using Basic Pitch
- **Alignment Analysis**: Score-to-performance alignment with configurable accuracy metrics
- **Visual Feedback**: Color-coded notation with accuracy overlays using Verovio
- **Export Options**: CSV reports and detailed analysis metrics
- **Real-time Processing**: Background job processing with progress tracking

## 🏗️ Architecture

- **Frontend**: Next.js/React with Tailwind CSS and Verovio for notation rendering
- **Backend**: FastAPI with async processing and Celery job queues
- **Audio Processing**: Basic Pitch (Spotify) for transcription
- **OMR**: Audiveris for PDF/image to MusicXML conversion
- **Database**: PostgreSQL for metadata and analysis results
- **Cache**: Redis for job queues and session management
- **Storage**: Local filesystem (S3-ready for production)

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**
- **Python 3.9+**
- **Node.js 18+**
- **4GB+ RAM** (8GB recommended for transcription)

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd rondo

# Run the startup script
./start.sh
```

The startup script will:
- Start PostgreSQL and Redis containers
- Install Python and Node.js dependencies
- Launch all services
- Open the application at http://localhost:3000

### Option 2: Manual Setup

```bash
# 1. Start database services
docker-compose up -d postgres redis

# 2. Install backend dependencies
cd backend
pip install -r requirements.txt

# 3. Install frontend dependencies
cd ../frontend
npm install

# 4. Start backend (Terminal 1)
cd ../backend
uvicorn app.main:app --reload --port 8000

# 5. Start frontend (Terminal 2)
cd ../frontend
npm run dev

# 6. Start worker (Terminal 3)
cd ../backend
celery -A app.worker worker --loglevel=info
```

### Option 3: Docker Production

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 📊 Usage

1. **Upload Files**: Upload a score (MusicXML/PDF) and audio recording (WAV/MP3)
2. **Configure Analysis**: Adjust timing and pitch tolerances
3. **Process**: The system will transcribe audio and align with the score
4. **Review Results**: View accuracy metrics and visual overlays
5. **Export**: Download CSV reports for detailed analysis

## 🎯 MVP Scope

- **Solo piano performances** (expandable to other instruments)
- **44.1kHz audio** (mono/stereo)
- **≤10 minutes per upload**
- **Common-practice notation**
- **Default tolerances**: ±50ms onset, exact pitch match

## 📁 Project Structure

```
rondo/
├── backend/                 # FastAPI Python application
│   ├── app/
│   │   ├── api/            # API routes and schemas
│   │   ├── services/       # Core processing services
│   │   ├── models.py       # Database models
│   │   ├── config.py       # Configuration settings
│   │   └── main.py         # Application entry point
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Backend container
├── frontend/               # Next.js React application
│   ├── app/               # Next.js app directory
│   ├── components/        # React components
│   ├── types/            # TypeScript definitions
│   ├── package.json      # Node.js dependencies
│   └── Dockerfile        # Frontend container
├── docker/               # Docker configurations
│   └── audiveris/        # OMR service container
├── tests/                # Test suite
├── docs/                 # Documentation
├── uploads/              # File upload directory
├── docker-compose.yml    # Service orchestration
└── start.sh             # Development startup script
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://rondo:rondo@localhost:5432/rondo

# Redis
REDIS_URL=redis://localhost:6379/0

# Audio Processing
MAX_AUDIO_DURATION=600
SAMPLE_RATE=44100
ONSET_TOLERANCE_MS=50
PITCH_TOLERANCE=0

# File Upload
MAX_FILE_SIZE=104857600
```

### Tolerance Settings

- **Onset Tolerance**: Timing tolerance in milliseconds (default: 50ms)
- **Pitch Tolerance**: Pitch tolerance in semitones (default: 0 = exact match)

## 📚 API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Full Documentation**: See `docs/API_DOCUMENTATION.md`

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest

# Run frontend tests
cd frontend
npm test
```

## 📊 Performance

- **Target**: <60 seconds for 5-minute files
- **Memory**: 4-8GB RAM recommended
- **Storage**: Local filesystem (S3-ready)
- **Concurrent Users**: Single instance (scalable)

## 🔍 Analysis Features

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correctly played notes
- **Precision/Recall**: F1 score metrics
- **Timing RMSE**: Root mean square error for timing
- **Per-measure Analysis**: Detailed breakdown by measure

### Error Classification
- **Correct**: Notes played accurately
- **Missed**: Notes in score but not played
- **Extra**: Notes played but not in score
- **Timing Errors**: Notes played with timing deviations
- **Pitch Errors**: Notes played with wrong pitch

### Visual Feedback
- **Color-coded Overlays**: Red (missed), Purple (extra), Orange (timing), Green (correct)
- **Measure Heatmaps**: Accuracy visualization by measure
- **Interactive Score**: Clickable notation with detailed information

## 🚀 Deployment

### Development
```bash
./start.sh
```

### Production
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Monitor logs
docker-compose logs -f
```

### Environment Setup
- **Production Database**: Use managed PostgreSQL service
- **File Storage**: Configure S3 or similar cloud storage
- **Load Balancing**: Use nginx or cloud load balancer
- **Monitoring**: Add Prometheus/Grafana for metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Documentation**: Check `docs/` directory
- **Known Issues**: See `docs/KNOWN_LIMITATIONS.md`
- **API Reference**: See `docs/API_DOCUMENTATION.md`
- **Issues**: Create GitHub issue with reproduction steps

## 🎯 Roadmap

### v1.1 (Next Release)
- [ ] Enhanced transcription models
- [ ] Real-time analysis updates
- [ ] Advanced score overlays
- [ ] User authentication
- [ ] Analysis history

### v1.2 (Future)
- [ ] Multi-instrument support
- [ ] Expressive analysis (rubato, dynamics)
- [ ] Mobile application
- [ ] Cloud deployment
- [ ] Advanced exports (annotated MusicXML, MIDI)

---

**Built with ❤️ for classical musicians and music educators**
