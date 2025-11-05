# Piano Analysis Backend

SQLAlchemy + FastAPI backend for storing and managing piano performance data and expressive features.

## Features

- **Score Management**: Store musical scores with MusicXML paths and beat information
- **Performance Tracking**: Record student and reference performances with metadata
- **Expressive Envelopes**: Store statistical envelopes for tempo, loudness, articulation, pedal, and balance features
- **RESTful API**: Full CRUD operations with Pydantic validation
- **Database Integration**: SQLite by default, PostgreSQL support via configuration

## Models

### ScorePiece
- `id`: Primary key
- `slug`: Unique identifier (e.g., "chopin-op10-no1")
- `musicxml_path`: Path to MusicXML file
- `beats_json`: Array of normalized beat indices and durations
- `created_at`: Timestamp

### Performance
- `id`: Primary key
- `score_id`: Foreign key to ScorePiece
- `role`: Enum ('student' | 'reference')
- `source`: File path or URL
- `sr`: Sample rate
- `duration_s`: Duration in seconds
- `features_json`: Extracted expressive features
- `created_at`: Timestamp

### Envelope
- `id`: Primary key
- `score_id`: Foreign key to ScorePiece
- `feature`: Enum ('tempo' | 'loudness' | 'articulation' | 'pedal' | 'balance')
- `beats`: Array of beat positions
- `p20`, `median`, `p80`: Statistical envelope values
- `n_refs`: Number of reference performances used
- `created_at`: Timestamp
- Unique constraint on (score_id, feature)

## API Endpoints

### Score Pieces
- `POST /scores/` - Create score piece
- `GET /scores/` - List score pieces
- `GET /scores/{id}` - Get specific score piece

### Performances
- `POST /performances/` - Create performance
- `GET /performances/` - List performances (filterable by score_id)
- `GET /performances/{id}` - Get specific performance

### Envelopes
- `GET /scores/{id}/envelopes/` - Get all envelopes for a score
- `GET /scores/{id}/envelopes/{feature}` - Get specific feature envelope

## Quick Start

```bash
# Install dependencies (from project root)
cd api && source venv/bin/activate
pip install -r requirements.txt

# Start backend server
cd ../backend
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Visit http://localhost:8001/docs for interactive API documentation.

## Database Configuration

Default: SQLite (`piano_analysis.db`)

For PostgreSQL, set environment variable:
```bash
export DATABASE_URL="postgresql://user:password@localhost/piano_analysis"
```

## Integration

The backend integrates with the main analysis API (`api/main.py`) which automatically:
- Imports database modules if available
- Creates tables on startup
- Falls back to legacy mode if database unavailable

## Schema Examples

```python
# Create a score piece
{
    "slug": "chopin-op10-no1",
    "musicxml_path": "/scores/chopin_op10_no1.musicxml"
}

# Create a performance
{
    "score_id": 1,
    "role": "student",
    "source": "/recordings/student_performance.wav"
}

# Expressive features
{
    "tempo": {
        "beats": [0.0, 1.0, 2.0, 3.0],
        "values": [120.0, 125.0, 122.0, 118.0]
    },
    "loudness": {
        "beats": [0.0, 1.0, 2.0, 3.0], 
        "values": [0.5, 0.6, 0.55, 0.52]
    }
}
```
