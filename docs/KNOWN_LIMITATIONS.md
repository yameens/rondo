# Known Limitations - Rondo v1.0

This document outlines the current limitations and constraints of the Rondo classical performance analysis system.

## Audio Processing Limitations

### Transcription Accuracy
- **Basic Pitch Model**: Uses Spotify's Basic Pitch for transcription, which is optimized for general polyphonic audio but may not achieve the same accuracy as specialized piano transcription models
- **Noise Sensitivity**: Performance degrades significantly with background noise, room acoustics, or poor recording quality
- **Piano Focus**: Optimized for solo piano; ensemble recordings may require source separation preprocessing

### Audio Format Constraints
- **Duration Limit**: Maximum 10 minutes per upload (configurable)
- **Sample Rate**: Automatically resampled to 44.1kHz
- **Channels**: Automatically converted to mono
- **File Size**: Maximum 100MB per file

## Score Processing Limitations

### OMR (Optical Music Recognition)
- **PDF Quality**: Requires high-quality, clear scans for best results
- **Complex Notation**: May struggle with:
  - Handwritten scores
  - Very dense notation
  - Contemporary extended techniques
  - Non-standard symbols or annotations
- **Processing Time**: OMR can take 2-5 minutes for complex scores

### MusicXML Support
- **Format Variations**: Some MusicXML variants may not parse correctly
- **Extended Features**: Limited support for:
  - Microtonal notation
  - Complex tuplets across barlines
  - Advanced articulations
  - Custom symbols

## Alignment Limitations

### Timing Analysis
- **Rubato Handling**: Limited tolerance for expressive timing variations
- **Tempo Changes**: May struggle with extreme tempo fluctuations
- **Grace Notes**: Currently excluded from analysis by default
- **Ornaments**: Trills, mordents, and other ornaments are simplified

### Pitch Analysis
- **Enharmonic Equivalents**: Treated as different notes (configurable)
- **Octave Errors**: May not distinguish between octave displacements
- **Pedaling**: Basic pedal detection only (if supported by transcription model)

## Performance Constraints

### Processing Time
- **Target**: Under 60 seconds for 5-minute files
- **Actual**: May take 2-5 minutes depending on:
  - Audio length and complexity
  - Score complexity
  - Server load
  - Hardware specifications

### Resource Usage
- **Memory**: High memory usage during transcription (4-8GB recommended)
- **GPU**: Optional GPU acceleration for transcription
- **Storage**: Temporary files stored locally (not S3 in MVP)

## User Interface Limitations

### Score Visualization
- **Verovio Integration**: Basic integration with limited overlay capabilities
- **Real-time Updates**: No real-time score highlighting during analysis
- **Audio Synchronization**: Basic playback without synchronized cursor
- **Mobile Support**: Limited mobile optimization

### Export Features
- **CSV Export**: Basic note-level data only
- **MusicXML Export**: No annotated MusicXML export in MVP
- **MIDI Export**: Not implemented in v1.0

## Technical Limitations

### Database
- **Local Storage**: Uses local filesystem instead of S3 in MVP
- **No User Management**: No authentication or user accounts
- **Limited History**: No persistent analysis history

### Scalability
- **Single Instance**: No load balancing or horizontal scaling
- **Queue Management**: Basic Celery queue without priority handling
- **Error Recovery**: Limited error recovery and retry mechanisms

## Future Improvements

### Planned Enhancements
1. **Enhanced Transcription**: Integration with specialized piano transcription models
2. **Advanced OMR**: Support for more complex notation and handwritten scores
3. **Real-time Analysis**: Live analysis during performance
4. **Advanced Visualizations**: Interactive score overlays and audio synchronization
5. **Cloud Storage**: S3 integration for file storage
6. **User Management**: Authentication and analysis history
7. **Mobile App**: Native mobile application
8. **API Enhancements**: WebSocket support for real-time updates

### Research Areas
- **Multi-instrument Support**: Beyond solo piano
- **Expressive Analysis**: Rubato, dynamics, and articulation analysis
- **Pedal Detection**: Advanced pedal usage analysis
- **Performance Style**: Style classification and comparison
- **Learning Analytics**: Progress tracking and improvement suggestions

## Workarounds

### For Better Results
1. **Audio Quality**: Use high-quality recordings with minimal background noise
2. **Score Quality**: Ensure clear, high-resolution score scans
3. **Performance Style**: Avoid extreme rubato or tempo changes during analysis
4. **File Preparation**: Convert audio to WAV format for best results
5. **Score Format**: Use MusicXML when possible instead of PDF

### For Complex Pieces
1. **Segment Analysis**: Break long pieces into shorter segments
2. **Multiple Uploads**: Analyze different sections separately
3. **Manual Correction**: Review and correct OMR results before analysis
4. **Tolerance Adjustment**: Increase timing tolerance for expressive pieces

## Support

For issues related to these limitations or to request features, please:
1. Check the GitHub issues for known problems
2. Create a new issue with detailed reproduction steps
3. Include sample files (audio/score) when possible
4. Specify your use case and requirements
