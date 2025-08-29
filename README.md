# Piano Performance Analysis MVP

A web application that analyzes piano performances by comparing user recordings against reference audio. Uses audio signal processing and machine learning to provide feedback on tempo, pitch accuracy, dynamics, and timing.

## Features

- 🎵 **Audio Analysis**: Compare user performance against reference recordings
- 📊 **Performance Metrics**: Tempo, pitch accuracy, dynamics, and overall scoring
- 🎯 **Issue Detection**: Identify specific problems with timestamps and explanations
- 📈 **Visual Charts**: Real-time analysis with tempo curves, chroma distance, and RMS dynamics
- 🎧 **Audio Playback**: Integrated player with clickable issue navigation
- 📱 **Modern UI**: Drag-and-drop file uploads, responsive design
- 🔗 **YouTube Support**: Direct analysis from YouTube URLs
- 🤖 **Optional MIDI Analysis**: Basic Pitch note transcription (when enabled)

## Quick Start

### Option 1: Local Development (No Docker)

#### Prerequisites
- Python 3.10+ 
- Node.js 18+
- FFmpeg
- System audio libraries

#### Backend Setup
```bash
# Navigate to API directory
cd api

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup
```bash
# In a new terminal, navigate to web directory
cd web

# Install dependencies
npm install

# Create environment file
cp .env.local.example .env.local  # Or create manually (see below)

# Start frontend
npm run dev
```

#### Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Docker Compose

#### Prerequisites
- Docker
- Docker Compose

#### Setup
```bash
# Clone and navigate to project
cd piano-analysis-mvp

# Build and start services
docker compose up --build

# Or run in background
docker compose up -d --build
```

#### Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

#### Docker Management
```bash
# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild after changes
docker compose up --build
```

## Environment Variables

### Frontend (.env.local)
```bash
# Backend API URL (required)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### Backend (Environment or .env)
```bash
# Enable Basic Pitch MIDI transcription (optional)
# Note: Requires compatible system (not ARM64 macOS)
BASIC_PITCH=true

# Python path (auto-configured in Docker)
PYTHONPATH=/app
```

## Usage

1. **Upload Reference Audio**: Drag and drop or click to upload a reference performance (MP3, WAV, M4A)
2. **Upload User Audio**: Either upload a file or paste a YouTube URL
3. **Analyze**: Click "Analyze Performance" and wait 10-40 seconds
4. **Review Results**: 
   - View overall scores (tempo, pitch, dynamics)
   - Browse detailed issue table
   - Click issues to jump to specific times in audio player
   - Download CSV report for detailed analysis
   - Examine performance charts

## API Endpoints

### Core Analysis
- `POST /analyze` - Submit analysis job
- `GET /analyze/{job_id}` - Get analysis results
- `GET /health` - Health check

### Static Files
- `GET /static/{filename}` - Audio preview files

## Testing

```bash
# Run all tests
pytest -q

# Run specific test modules
pytest tests/test_utils.py -v
pytest tests/test_analysis.py -v

# Generate coverage report
pytest --cov=app tests/
```

## Known Limitations

### Audio Analysis
- **Polyphony Transcription**: MIDI note detection works best on simple melodies; complex chords may be inaccurately transcribed
- **Chroma Distance Proxy**: Pitch accuracy uses chroma features as a proxy; may not detect all subtle intonation issues
- **DTW Warping Constraints**: Dynamic Time Warping alignment has constraints that may not handle extreme tempo variations
- **Mono Mix Dependency**: Analysis relies on mono audio mixing; stereo imaging and panning information is lost

### Technical Constraints
- **File Size Limits**: Maximum 20MB uploads, 10-minute duration
- **Processing Time**: Analysis typically takes 10-40 seconds depending on audio length
- **Basic Pitch Availability**: MIDI transcription requires compatible system architecture (not available on ARM64 macOS)
- **Memory Usage**: Large audio files may require significant memory for processing

### Accuracy Considerations
- **Reference Quality**: Analysis accuracy depends heavily on reference recording quality
- **Genre Limitations**: Optimized for solo piano; other instruments may produce unreliable results
- **Tempo Detection**: Works best with steady tempo; rubato and ritardando may be flagged as issues

## Architecture

```
piano-analysis-mvp/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints and job management
│   ├── requirements.txt   # Python dependencies
│   └── venv/             # Virtual environment
├── app/                   # Core analysis modules
│   ├── analysis.py       # Audio analysis and MIDI processing
│   └── utils.py          # Audio utilities and YouTube handling
├── web/                   # Next.js frontend
│   ├── src/app/page.tsx  # Main application page
│   ├── package.json      # Node.js dependencies
│   └── .env.local        # Environment configuration
├── tests/                 # Test suite
│   ├── test_utils.py     # Utility function tests
│   ├── test_analysis.py  # Analysis function tests
│   └── fixtures.py       # Test audio generation
├── backend/Dockerfile     # Backend container
├── frontend/Dockerfile    # Frontend container
└── docker-compose.yml    # Multi-service orchestration
```

## Next Steps Backlog

### Audio Analysis Improvements
- [ ] **Measure Detection**: Implement automatic measure/bar detection for more musical feedback
- [ ] **LUFS Normalization**: Add perceptual loudness normalization for better dynamics comparison
- [ ] **Advanced MIDI Matching**: Improve note onset/offset matching with better alignment algorithms
- [ ] **Polyphonic Analysis**: Enhanced chord recognition and voice separation

### User Experience
- [ ] **On-Score Overlays**: Visual score display with performance issues highlighted
- [ ] **Practice Recommendations**: AI-generated specific practice suggestions
- [ ] **Progress Tracking**: Historical performance comparison and improvement metrics
- [ ] **Real-time Analysis**: Live performance feedback during recording

### Technical Enhancements
- [ ] **Cloud Storage**: S3/GCS integration for audio file persistence
- [ ] **Database Integration**: PostgreSQL for job history and user data
- [ ] **Authentication**: User accounts and session management
- [ ] **Caching Layer**: Redis for analysis result caching

### Platform Features
- [ ] **Mobile App**: React Native or Flutter mobile client
- [ ] **Batch Processing**: Multiple file analysis workflows
- [ ] **API Rate Limiting**: Production-ready request throttling
- [ ] **Monitoring**: Comprehensive logging and metrics dashboard

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- Open a GitHub issue
- Check existing documentation
- Review test cases for usage examples