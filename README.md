# AI Conspects Server

A comprehensive Python server for AI-powered note generation from audio recordings, supporting both Ollama and OpenRouter modes with Whisper Large for speech-to-text and Perplexity API for research.

## Features

- 🎤 **Audio Processing**: Whisper Large (large-v3) for high-quality speech-to-text
- 🤖 **Dual LLM Modes**: 
  - Ollama (local models like Llama 3.1)
  - OpenRouter (cloud-based models)
- 🔍 **Research Integration**: Perplexity API for information augmentation
- 📱 **iOS App Compatible**: RESTful API designed for mobile apps
- 🐳 **Docker Ready**: Complete containerization with Docker Compose
- 🗄️ **Vector Database**: PostgreSQL with pgvector for semantic search
- ⚡ **Async Processing**: Celery workers for background tasks
- 🔐 **Authentication**: JWT-based device authentication

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL with pgvector extension
- Redis
- Ollama (for local LLM mode)

### Environment Setup

1. Copy the environment file:
```bash
cp env.example .env
```

2. Edit `.env` with your configuration:
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ai_conspects
REDIS_URL=redis://localhost:6379

# AI Services
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
PERPLEXITY_API_KEY=your_perplexity_key

# Ollama (for local mode)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Authentication
JWT_SECRET_KEY=your_jwt_secret_key_here
```

### Docker Deployment

1. Start all services:
```bash
docker-compose up -d
```

2. The server will be available at:
   - **Mac**: http://localhost:8000
   - **Windows**: http://localhost:5000

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python main.py
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new device
- `POST /api/v1/auth/login` - Login with device ID
- `GET /api/v1/auth/me` - Get current user info

### Audio Processing
- `POST /api/v1/batches/submit` - Submit audio batch for processing
- `GET /api/v1/batches/{batch_id}/status` - Get processing status

### Results
- `GET /api/v1/results/{batch_id}/results` - Get processed notes
- `GET /api/v1/results/notes` - Get user's notes
- `GET /api/v1/results/search` - Search notes

### Chat
- `POST /api/v1/chat/query` - AI chat with notes
- `GET /api/v1/chat/conversations` - Get conversations

## Usage Examples

### Submit Audio for Processing

```bash
curl -X POST "http://localhost:8000/api/v1/batches/submit" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "123e4567-e89b-12d3-a456-426614174000",
    "device_id": "unique-device-identifier",
    "settings": {
      "language": "ru",
      "analysis_depth": "detailed",
      "include_research": true,
      "generate_tags": true,
      "llm_mode": "ollama",
      "ollama_model": "llama3.1:8b"
    },
    "recordings": [
      {
        "id": "recording-uuid-1",
        "filename": "recording_1234567890.m4a",
        "duration": 125.5,
        "created_at": "2025-01-18T14:30:00Z",
        "audio_data": "base64-encoded-audio-content"
      }
    ]
  }'
```

### Chat with AI

```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What did I study about SwiftUI?",
    "context": {
      "include_notes": true,
      "max_context_notes": 10
    }
  }'
```

## Configuration

### LLM Modes

#### Ollama Mode (Local)
- Set `llm_mode: "ollama"` in request settings
- Requires Ollama running locally
- Models: llama3.1:8b, mistral, etc.

#### OpenRouter Mode (Cloud)
- Set `llm_mode: "openrouter"` in request settings
- Requires OpenRouter API key
- Models: gpt-4-turbo, claude-3, etc.

### Platform Detection

The server automatically detects the platform and uses the appropriate port:
- **Mac**: Port 8000
- **Windows**: Port 5000

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER                          │
│  • Authentication & Authorization                          │
│  • Request Validation & Rate Limiting                      │
│  • API Endpoints & WebSocket Support                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 CELERY WORKERS                             │
│  • Audio Processing (Speech-to-Text)                       │
│  • LLM Processing (Note Generation)                        │
│  • Research & Enrichment                                   │
│  • Vector Embeddings Generation                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
      ┌───▼───┐   ┌───▼───┐   ┌───▼───┐
      │ REDIS │   │  S3   │   │  PG   │
      │ QUEUE │   │FILES  │   │VECTOR │
      └───────┘   └───────┘   └───────┘
```

## Development

### Project Structure

```
ai_conspects_server/
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── database.py            # Database models
├── schemas.py             # Pydantic schemas
├── auth.py                # Authentication
├── audio_processing.py    # Audio processing endpoints
├── results.py             # Results retrieval
├── chat.py                # AI chat endpoints
├── llm_services.py        # LLM service implementations
├── perplexity_service.py  # Perplexity API integration
├── workers/               # Celery workers
│   ├── celery_app.py
│   ├── audio_worker.py
│   ├── llm_worker.py
│   └── research_worker.py
├── docker-compose.yml     # Docker services
├── Dockerfile            # Container definition
└── requirements.txt      # Python dependencies
```

### Adding New Features

1. **New API Endpoints**: Add to appropriate router files
2. **New LLM Models**: Extend `llm_services.py`
3. **New Research Sources**: Extend `perplexity_service.py`
4. **New Background Tasks**: Add to `workers/` directory

## Monitoring

### Health Checks

- Server health: `GET /health`
- Database connectivity: Built into startup
- Redis connectivity: Built into startup

### Logging

All services use structured JSON logging for easy monitoring and debugging.

## Troubleshooting

### Common Issues

1. **Whisper model loading slowly**: First run downloads the model
2. **Ollama connection failed**: Ensure Ollama is running locally
3. **Database connection failed**: Check PostgreSQL is running
4. **Redis connection failed**: Check Redis is running

### Debug Mode

Set environment variable `DEBUG=true` for detailed logging.

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please create an issue in the repository.
