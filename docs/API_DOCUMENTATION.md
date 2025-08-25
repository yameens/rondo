# Rondo API Documentation

## Overview

The Rondo API provides endpoints for uploading and analyzing classical music performances against scores. The API is built with FastAPI and provides RESTful endpoints for file upload, analysis management, and result retrieval.

## Base URL

- **Development**: `http://localhost:8000/api/v1`
- **Production**: `https://api.rondo.com/api/v1`

## Authentication

Currently, the API does not require authentication for MVP. Future versions will include JWT-based authentication.

## Endpoints

### 1. Upload Files

**POST** `/upload`

Upload a score file and audio recording for analysis.

#### Request

- **Content-Type**: `multipart/form-data`

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `score_file` | File | Yes | Score file (MusicXML, PDF, MXL, MEI) |
| `audio_file` | File | Yes | Audio recording (WAV, MP3, FLAC) |
| `onset_tolerance_ms` | Integer | No | Timing tolerance in milliseconds (default: 50) |
| `pitch_tolerance` | Integer | No | Pitch tolerance in semitones (default: 0) |

#### File Requirements

**Score Files:**
- **Formats**: `.xml`, `.musicxml`, `.mxl`, `.mei`, `.pdf`
- **Max Size**: 100MB
- **Content**: MusicXML, MEI, or PDF with musical notation

**Audio Files:**
- **Formats**: `.wav`, `.mp3`, `.flac`
- **Max Size**: 100MB
- **Duration**: ≤10 minutes
- **Sample Rate**: Will be resampled to 44.1kHz
- **Channels**: Will be converted to mono

#### Response

```json
{
  "id": "uuid-string",
  "status": "pending",
  "message": "Analysis started"
}
```

#### Example

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "score_file=@score.xml" \
  -F "audio_file=@performance.wav" \
  -F "onset_tolerance_ms=50" \
  -F "pitch_tolerance=0"
```

### 2. Get Analysis Status

**GET** `/analysis/{analysis_id}/status`

Get the current status of an analysis.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | String | Yes | Analysis ID from upload response |

#### Response

```json
{
  "id": "uuid-string",
  "status": "processing",
  "progress": 0.6,
  "error_message": null
}
```

#### Status Values

- `pending`: Analysis queued but not started
- `processing`: Analysis in progress
- `completed`: Analysis finished successfully
- `failed`: Analysis failed with error

### 3. Get Analysis Results

**GET** `/analysis/{analysis_id}`

Get complete analysis results.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | String | Yes | Analysis ID from upload response |

#### Response

```json
{
  "id": "uuid-string",
  "status": "completed",
  "progress": 1.0,
  "results": [
    {
      "score_event": {
        "pitch": 60,
        "onset_s": 1.0,
        "offset_s": 1.5,
        "velocity": 80,
        "measure": 1
      },
      "performance_event": {
        "pitch": 60,
        "onset_s": 1.05,
        "offset_s": 1.55,
        "velocity": 85,
        "measure": 1
      },
      "type": "matched",
      "accuracy_type": "correct",
      "onset_delta": 0.05,
      "pitch_delta": 0,
      "velocity_delta": 5,
      "cost": 5.0
    }
  ],
  "metrics": {
    "total_score_notes": 100,
    "total_performance_notes": 102,
    "correct_notes": 95,
    "missed_notes": 5,
    "extra_notes": 7,
    "timing_errors": 3,
    "pitch_errors": 2,
    "precision": 0.931,
    "recall": 0.950,
    "f1_score": 0.940,
    "timing_rmse": 0.023,
    "overall_accuracy": 0.950
  },
  "error_message": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:15Z"
}
```

### 4. Export CSV Report

**GET** `/analysis/{analysis_id}/export/csv`

Export analysis results as CSV.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `analysis_id` | String | Yes | Analysis ID from upload response |

#### Response

- **Content-Type**: `application/json`
- **Body**: CSV content and filename

```json
{
  "content": "Event_Type,Score_Pitch,Score_Onset,Score_Offset,Perf_Pitch,Perf_Onset,Perf_Offset,Accuracy_Type,Onset_Delta,Pitch_Delta,Velocity_Delta,Measure\nmatched,60,1.0,1.5,60,1.05,1.55,correct,0.05,0,5,1\n...",
  "filename": "rondo_analysis_uuid-string.csv"
}
```

### 5. Health Check

**GET** `/health`

Check API health status.

#### Response

```json
{
  "status": "healthy",
  "service": "rondo-api"
}
```

## Data Models

### Analysis Status

```typescript
interface AnalysisStatus {
  id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number; // 0.0 to 1.0
  error_message?: string;
}
```

### Analysis Results

```typescript
interface AnalysisData {
  id: string;
  status: string;
  progress: number;
  results?: AlignmentResult[];
  metrics?: AccuracyMetrics;
  error_message?: string;
  created_at: string;
  updated_at: string;
}
```

### Alignment Result

```typescript
interface AlignmentResult {
  score_event?: NoteEvent;
  performance_event?: NoteEvent;
  type: "matched" | "missed" | "extra";
  accuracy_type: "correct" | "missed" | "extra" | "timing_error" | "pitch_error";
  onset_delta?: number;
  pitch_delta?: number;
  velocity_delta?: number;
  cost: number;
}
```

### Note Event

```typescript
interface NoteEvent {
  pitch: number; // MIDI pitch (0-127)
  onset_s: number; // Onset time in seconds
  offset_s: number; // Offset time in seconds
  velocity: number; // MIDI velocity (0-127)
  confidence?: number; // Transcription confidence (0-1)
  duration_s: number; // Note duration in seconds
  midi_note: number; // MIDI note number
  measure?: number; // Measure number
  part_id?: string; // Part identifier
  type?: string; // Event type
}
```

### Accuracy Metrics

```typescript
interface AccuracyMetrics {
  total_score_notes: number;
  total_performance_notes: number;
  correct_notes: number;
  missed_notes: number;
  extra_notes: number;
  timing_errors: number;
  pitch_errors: number;
  precision: number; // 0.0 to 1.0
  recall: number; // 0.0 to 1.0
  f1_score: number; // 0.0 to 1.0
  timing_rmse: number; // Root mean square error in seconds
  overall_accuracy: number; // 0.0 to 1.0
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common Error Messages

| Error | Description | Solution |
|-------|-------------|----------|
| `File too large` | File exceeds 100MB limit | Reduce file size |
| `Unsupported format` | File format not supported | Use supported formats |
| `Audio duration too long` | Audio exceeds 10 minutes | Shorten audio file |
| `Analysis not found` | Invalid analysis ID | Check analysis ID |
| `Analysis not completed` | Results not ready | Wait for completion |

## Rate Limiting

Currently, no rate limiting is implemented. Future versions will include rate limiting based on user tiers.

## WebSocket Support

WebSocket support for real-time analysis updates is planned for future versions.

## SDKs and Libraries

### Python

```python
import requests

# Upload files
files = {
    'score_file': open('score.xml', 'rb'),
    'audio_file': open('performance.wav', 'rb')
}
data = {
    'onset_tolerance_ms': 50,
    'pitch_tolerance': 0
}

response = requests.post('http://localhost:8000/api/v1/upload', files=files, data=data)
analysis_id = response.json()['id']

# Poll for results
while True:
    status_response = requests.get(f'http://localhost:8000/api/v1/analysis/{analysis_id}/status')
    status = status_response.json()
    
    if status['status'] == 'completed':
        results = requests.get(f'http://localhost:8000/api/v1/analysis/{analysis_id}')
        print(results.json())
        break
    elif status['status'] == 'failed':
        print(f"Analysis failed: {status['error_message']}")
        break
    
    time.sleep(2)
```

### JavaScript

```javascript
// Upload files
const formData = new FormData();
formData.append('score_file', scoreFile);
formData.append('audio_file', audioFile);
formData.append('onset_tolerance_ms', '50');
formData.append('pitch_tolerance', '0');

const response = await fetch('/api/v1/upload', {
  method: 'POST',
  body: formData
});

const { id: analysisId } = await response.json();

// Poll for results
const pollResults = async () => {
  const statusResponse = await fetch(`/api/v1/analysis/${analysisId}/status`);
  const status = await statusResponse.json();
  
  if (status.status === 'completed') {
    const resultsResponse = await fetch(`/api/v1/analysis/${analysisId}`);
    const results = await resultsResponse.json();
    console.log(results);
  } else if (status.status === 'failed') {
    console.error(`Analysis failed: ${status.error_message}`);
  } else {
    setTimeout(pollResults, 2000);
  }
};

pollResults();
```

## Testing

### Postman Collection

A Postman collection is available for testing the API endpoints. Import the collection from the `docs/postman` directory.

### Example Test Data

Sample files for testing are available in the `tests/data` directory:
- `sample_score.xml`: Simple MusicXML score
- `sample_audio.wav`: 30-second piano recording

## Support

For API support:
1. Check the documentation
2. Review error messages
3. Test with sample data
4. Create an issue with reproduction steps
