# 🚀 AI Conspects - Python Server Development Protocol

## 📋 Server Architecture

### 🏗️ Core Components

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

## 🗄️ Database Schema (PostgreSQL + pgvector)

### Core Tables

```sql
-- Users and authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

-- Processing batches
CREATE TABLE processing_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    estimated_time INTEGER,
    settings JSONB DEFAULT '{}',
    progress JSONB DEFAULT '{}',
    error_message TEXT
);

-- Audio recordings
CREATE TABLE audio_recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID REFERENCES processing_batches(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    s3_key VARCHAR(500),
    duration FLOAT NOT NULL,
    file_size BIGINT,
    format VARCHAR(10),
    sample_rate INTEGER,
    channels INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    is_processed BOOLEAN DEFAULT FALSE
);

-- Notes (main table)
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    source_recording_id UUID REFERENCES audio_recordings(id) ON DELETE SET NULL,
    title VARCHAR(500) NOT NULL,
    summary TEXT,
    content TEXT NOT NULL,
    type VARCHAR(20) DEFAULT 'recording',
    confidence FLOAT DEFAULT 1.0,
    reading_time INTEGER DEFAULT 5,
    view_count INTEGER DEFAULT 0,
    is_read BOOLEAN DEFAULT FALSE,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    -- Vector embedding for semantic search
    embedding VECTOR(1536)
);

-- Tags
CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(50) NOT NULL,
    color VARCHAR(20) DEFAULT 'blue',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, name)
);

-- Many-to-many relationship between notes and tags
CREATE TABLE note_tags (
    note_id UUID REFERENCES notes(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (note_id, tag_id)
);

-- Information sources
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    note_id UUID REFERENCES notes(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    url VARCHAR(1000),
    type VARCHAR(50) NOT NULL,
    relevance FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Note relationships
CREATE TABLE note_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    note_1_id UUID REFERENCES notes(id) ON DELETE CASCADE,
    note_2_id UUID REFERENCES notes(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    strength FLOAT DEFAULT 0.5,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(note_1_id, note_2_id)
);

-- Chat messages
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    conversation_id UUID,
    content TEXT NOT NULL,
    is_from_user BOOLEAN NOT NULL,
    message_type VARCHAR(20) DEFAULT 'text',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_notes_user_created ON notes(user_id, created_at DESC);
CREATE INDEX idx_notes_embedding ON notes USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_notes_title_search ON notes USING gin(to_tsvector('russian', title));
CREATE INDEX idx_notes_content_search ON notes USING gin(to_tsvector('russian', content));
CREATE INDEX idx_audio_recordings_user ON audio_recordings(user_id, created_at DESC);
CREATE INDEX idx_processing_batches_user ON processing_batches(user_id, created_at DESC);
```

## 🔧 Technology Stack

### Core Dependencies

```python
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
psycopg2-binary==2.9.9
pgvector==0.2.4
redis==5.0.1
celery==5.3.4
boto3==1.34.0
openai==1.3.7
whisper==1.1.10
sentence-transformers==2.2.2
httpx==0.25.2
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
```

## 🚀 API Endpoints (FastAPI)

### 1. Authentication

```python
# auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Device registration/login
@router.post("/register")
async def register_device(device_id: str):
    """Register new device"""
    user = await get_or_create_user(device_id)
    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(user.id)
    }

@router.post("/login")
async def login_device(device_id: str):
    """Login by device_id"""
    user = await get_user_by_device_id(device_id)
    if not user:
        raise HTTPException(status_code=404, detail="Device not found")
    
    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(user.id)
    }
```

### 2. Audio Processing

```python
# audio_processing.py
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
import uuid
from datetime import datetime

router = APIRouter(prefix="/batches", tags=["audio processing"])

class RecordingData(BaseModel):
    id: str
    filename: str
    duration: float
    audio_data: str  # Base64 encoded
    created_at: str
    metadata: Optional[dict] = None

class ProcessingSettings(BaseModel):
    language: str = "ru"
    analysis_depth: str = "detailed"
    include_research: bool = True
    generate_tags: bool = True
    max_notes_per_recording: int = 3
    include_related_topics: bool = True

class BatchSubmitRequest(BaseModel):
    batch_id: str
    device_id: str
    settings: ProcessingSettings
    recordings: List[RecordingData]

@router.post("/submit")
async def submit_batch(
    request: BatchSubmitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Submit audio recordings batch for processing"""
    
    # Request validation
    if not request.recordings:
        raise HTTPException(status_code=400, detail="No recordings provided")
    
    if len(request.recordings) > 10:
        raise HTTPException(status_code=400, detail="Too many recordings in batch")
    
    # Create batch in DB
    batch = ProcessingBatch(
        id=uuid.UUID(request.batch_id),
        user_id=current_user.id,
        status="accepted",
        settings=request.settings.dict(),
        estimated_time=len(request.recordings) * 2  # 2 minutes per recording
    )
    
    # Save recordings
    for recording_data in request.recordings:
        # Decode and save audio
        audio_bytes = base64.b64decode(recording_data.audio_data)
        s3_key = f"audio/{current_user.id}/{request.batch_id}/{recording_data.id}.m4a"
        
        # Upload to S3
        await upload_to_s3(audio_bytes, s3_key)
        
        # Save to DB
        recording = AudioRecording(
            id=uuid.UUID(recording_data.id),
            batch_id=batch.id,
            user_id=current_user.id,
            filename=recording_data.filename,
            s3_key=s3_key,
            duration=recording_data.duration,
            created_at=datetime.fromisoformat(recording_data.created_at.replace('Z', '+00:00'))
        )
        db.add(recording)
    
    db.add(batch)
    await db.commit()
    
    # Start background processing
    background_tasks.add_task(process_batch_async, batch.id)
    
    return {
        "batch_id": str(batch.id),
        "status": "accepted",
        "estimated_time": batch.estimated_time,
        "queue_position": await get_queue_position(batch.id),
        "processing_steps": [
            "audio_upload",
            "transcription", 
            "content_analysis",
            "research_augmentation",
            "note_generation",
            "relationship_mapping"
        ],
        "message": "Batch accepted for processing"
    }

@router.get("/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get batch processing status"""
    
    batch = await get_batch_by_id(uuid.UUID(batch_id), current_user.id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Calculate progress
    total_recordings = len(batch.recordings)
    processed_recordings = sum(1 for r in batch.recordings if r.is_processed)
    
    progress_percentage = (processed_recordings / total_recordings * 100) if total_recordings > 0 else 0
    
    return {
        "batch_id": str(batch.id),
        "status": batch.status,
        "progress": {
            "current_step": batch.progress.get("current_step", "pending"),
            "completed_steps": batch.progress.get("completed_steps", []),
            "progress_percentage": int(progress_percentage),
            "estimated_remaining": batch.estimated_time - batch.progress.get("elapsed_time", 0)
        },
        "recordings_processed": processed_recordings,
        "total_recordings": total_recordings,
        "created_at": batch.created_at.isoformat(),
        "updated_at": batch.updated_at.isoformat()
    }
```

### 3. Results Retrieval

```python
# results.py
@router.get("/{batch_id}/results")
async def get_batch_results(
    batch_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get processed notes results"""
    
    batch = await get_batch_by_id(uuid.UUID(batch_id), current_user.id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if batch.status != "completed":
        raise HTTPException(status_code=400, detail="Batch not completed yet")
    
    # Get all notes from batch
    notes = await get_notes_by_batch(batch.id)
    
    # Calculate statistics
    total_processing_time = (batch.updated_at - batch.created_at).total_seconds()
    avg_confidence = sum(note.confidence for note in notes) / len(notes) if notes else 0
    
    # Get note relationships
    relationships = await get_note_relationships([note.id for note in notes])
    
    return {
        "batch_id": str(batch.id),
        "status": batch.status,
        "processing_summary": {
            "total_recordings": len(batch.recordings),
            "notes_generated": len(notes),
            "total_processing_time": int(total_processing_time),
            "confidence_avg": round(avg_confidence, 2)
        },
        "notes": [format_note_response(note) for note in notes],
        "relationships": [format_relationship_response(rel) for rel in relationships]
    }

def format_note_response(note: Note) -> dict:
    """Format note for API response"""
    return {
        "id": str(note.id),
        "source_recording_id": str(note.source_recording_id) if note.source_recording_id else None,
        "title": note.title,
        "summary": note.summary,
        "content": note.content,
        "tags": [{"name": tag.name, "confidence": 0.9, "color": tag.color} for tag in note.tags],
        "metadata": {
            "reading_time": note.reading_time,
            "confidence": note.confidence,
            "word_count": len(note.content.split()),
            "complexity_level": "intermediate"  # Can add AI analysis
        },
        "sources": [{"title": s.title, "url": s.url, "type": s.type, "relevance": s.relevance} for s in note.sources],
        "related_topics": [rel.note_2.title for rel in note.relationships_1],
        "created_at": note.created_at.isoformat()
    }
```

### 4. AI Chat API

```python
# chat.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
from sentence_transformers import SentenceTransformer

router = APIRouter(prefix="/chat", tags=["ai chat"])

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[dict] = None
    settings: Optional[dict] = None

class ChatResponse(BaseModel):
    response: dict
    context_used: List[dict]
    suggested_actions: List[dict]
    conversation_id: str

@router.post("/query")
async def chat_query(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """Process AI chat query"""
    
    # Find relevant notes
    relevant_notes = await find_relevant_notes(
        request.message, 
        current_user.id,
        request.context
    )
    
    # Format context for LLM
    context_text = format_notes_for_context(relevant_notes)
    
    # Generate response via OpenAI
    response = await generate_ai_response(
        user_message=request.message,
        context=context_text,
        conversation_id=request.conversation_id
    )
    
    # Save messages to DB
    conversation_id = await save_chat_messages(
        user_id=current_user.id,
        user_message=request.message,
        ai_response=response["message"],
        conversation_id=request.conversation_id,
        related_notes=relevant_notes
    )
    
    return ChatResponse(
        response=response,
        context_used=[{"note_id": str(note.id), "title": note.title, "relevance": 0.9} for note in relevant_notes],
        suggested_actions=response.get("suggested_actions", []),
        conversation_id=conversation_id
    )

async def find_relevant_notes(
    query: str, 
    user_id: uuid.UUID, 
    context: Optional[dict] = None
) -> List[Note]:
    """Find relevant notes using vector search"""
    
    # Generate embedding for query
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = model.encode([query])[0]
    
    # Vector search in DB
    notes = await db.execute(
        select(Note)
        .where(Note.user_id == user_id)
        .order_by(Note.embedding.cosine_distance(query_embedding))
        .limit(10)
    )
    
    return notes.scalars().all()

async def generate_ai_response(
    user_message: str,
    context: str,
    conversation_id: Optional[str] = None
) -> dict:
    """Generate response via OpenAI"""
    
    system_prompt = f"""
    You are a personal AI assistant for working with notes and knowledge.
    
    User context:
    {context}
    
    Respond in Russian, be helpful and specific.
    If user asks about their notes, use the provided context.
    """
    
    response = await openai.ChatCompletion.acreate(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return {
        "message": response.choices[0].message.content,
        "confidence": 0.9
    }
```

## 🔄 Celery Workers (Async Processing)

### 1. Audio Processing Worker

```python
# workers/audio_worker.py
from celery import Celery
import whisper
import boto3
from io import BytesIO
import uuid

app = Celery('ai_conspects')

@app.task
def process_audio_recording(recording_id: str, s3_key: str):
    """Process audio recording: transcription and analysis"""
    
    # Load Whisper model
    model = whisper.load_model("large-v3")
    
    # Download audio from S3
    s3_client = boto3.client('s3')
    audio_obj = s3_client.get_object(Bucket='ai-conspects-audio', Key=s3_key)
    audio_data = audio_obj['Body'].read()
    
    # Transcription
    result = model.transcribe(BytesIO(audio_data), language='ru')
    
    # Clean and format text
    transcript = clean_transcript(result['text'])
    key_phrases = extract_key_phrases(transcript)
    
    # Save result
    await save_transcription_result(recording_id, {
        'transcript': transcript,
        'key_phrases': key_phrases,
        'confidence': result.get('confidence', 0.9),
        'language': result.get('language', 'ru')
    })
    
    return {
        'recording_id': recording_id,
        'transcript': transcript,
        'key_phrases': key_phrases,
        'confidence': result.get('confidence', 0.9)
    }

def clean_transcript(text: str) -> str:
    """Clean and format transcript"""
    # Remove extra spaces and characters
    text = ' '.join(text.split())
    # Remove repetitive phrases
    # Add punctuation
    return text

def extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from text"""
    # Simple implementation - can be improved with NLP libraries
    words = text.split()
    # Filter stop words and extract meaningful phrases
    return [phrase for phrase in words if len(phrase) > 3][:10]
```

### 2. LLM Processing Worker

```python
# workers/llm_worker.py
from celery import Celery
import openai
from sentence_transformers import SentenceTransformer
import uuid

app = Celery('ai_conspects')

@app.task
def generate_notes_from_transcript(recording_id: str, transcript: str, user_id: str):
    """Generate notes from transcript"""
    
    # Get user context
    user_context = await get_user_context(user_id)
    
    # Prompt for note generation
    prompt = f"""
    Analyze the following transcript and create a structured note:
    
    Transcript: {transcript}
    
    User context: {user_context}
    
    Create:
    1. Title (brief and descriptive)
    2. Brief summary (2-3 sentences)
    3. Main content in Markdown
    4. 3-5 relevant tags
    5. Connections to existing topics
    6. Reading time (estimate)
    """
    
    # Call OpenAI
    response = await openai.ChatCompletion.acreate(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000
    )
    
    # Parse response
    note_data = parse_llm_response(response.choices[0].message.content)
    
    # Create note in DB
    note = await create_note_from_llm_response(
        recording_id=recording_id,
        user_id=user_id,
        note_data=note_data
    )
    
    # Generate embeddings
    await generate_note_embeddings(note.id, note.content)
    
    return {
        'note_id': str(note.id),
        'title': note.title,
        'confidence': note.confidence
    }

def parse_llm_response(response_text: str) -> dict:
    """Parse LLM response into structured data"""
    # Implementation of LLM response parsing
    # Split into title, summary, content, tags
    lines = response_text.split('\n')
    
    title = ""
    summary = ""
    content = ""
    tags = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            title = line[2:]
        elif line.startswith('## '):
            current_section = line[3:].lower()
        elif line.startswith('- '):
            if current_section == 'теги':
                tags.append(line[2:])
        else:
            if current_section == 'резюме':
                summary += line + " "
            elif current_section == 'содержание':
                content += line + "\n"
    
    return {
        'title': title,
        'summary': summary.strip(),
        'content': content.strip(),
        'tags': tags
    }
```

### 3. Research Worker

```python
# workers/research_worker.py
from celery import Celery
import httpx
from bs4 import BeautifulSoup
import asyncio

app = Celery('ai_conspects')

@app.task
def research_and_augment_note(note_id: str, research_settings: dict):
    """Research and augment note with additional information"""
    
    note = await get_note_by_id(note_id)
    if not note:
        return
    
    if research_settings.get('include_research', True):
        # Web search
        search_results = await search_web(note.title, note.key_phrases)
        
        # Filter and rank sources
        relevant_sources = filter_relevant_sources(
            search_results, 
            note.content,
            min_relevance=0.7
        )
        
        # Add sources to note
        for source in relevant_sources:
            await create_source(note_id, source)
    
    # Find connections with existing notes
    related_notes = await find_related_notes(note.embedding, note.user_id)
    
    # Create relationships
    for related_note in related_notes:
        await create_note_relationship(note_id, related_note.id)
    
    return {
        'note_id': note_id,
        'sources_added': len(relevant_sources),
        'relationships_created': len(related_notes)
    }

async def search_web(query: str, key_phrases: List[str]) -> List[dict]:
    """Search information on the web"""
    # Use search API (e.g., SerpAPI, Bing Search API)
    search_terms = f"{query} {' '.join(key_phrases[:3])}"
    
    # Here will be real search via API
    # For now return mock
    return [
        {
            'title': f'Search result for {query}',
            'url': 'https://example.com',
            'snippet': 'Description of found information...',
            'relevance': 0.8
        }
    ]
```

## ⚙️ Configuration and Settings

### Environment Variables

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost:5432/ai_conspects
REDIS_URL=redis://localhost:6379
S3_BUCKET=ai-conspects-audio
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_openai_key
SERPAPI_KEY=your_serpapi_key
JWT_SECRET_KEY=your_jwt_secret
CELERY_BROKER_URL=redis://localhost:6379
CELERY_RESULT_BACKEND=redis://localhost:6379
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ai_conspects
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/ai_conspects
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  worker:
    build: .
    command: celery -A app.celery worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/ai_conspects
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

## 📊 Monitoring and Logging

### Logging

```python
# logging_config.py
import logging
from pythonjsonlogger import jsonlogger

def setup_logging():
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)
    return logger
```

### Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Counters
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
processing_time = Histogram('processing_time_seconds', 'Processing time')
active_batches = Gauge('active_batches', 'Number of active processing batches')

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    api_requests_total.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    processing_time.observe(process_time)
    
    return response
```

## 🚀 Deployment

### Production Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-conspects-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-conspects-api
  template:
    metadata:
      labels:
        app: ai-conspects-api
    spec:
      containers:
      - name: api
        image: ai-conspects:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
---
apiVersion: v1
kind: Service
metadata:
  name: ai-conspects-service
spec:
  selector:
    app: ai-conspects-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🎯 Next Steps

### Development Phases

1. **Environment Setup** - Install PostgreSQL, Redis, Python dependencies
2. **Database Migrations** - Alembic for schema management
3. **S3 Configuration** - For audio file storage
4. **OpenAI Integration** - For note generation
5. **Testing** - Unit and integration tests

### Key Features

- **Scalability** - Microservice architecture with Celery
- **Performance** - Vector search, caching, async processing
- **Security** - JWT authentication, data validation
- **Monitoring** - Complete logging and metrics

## 📝 API Examples

### Submit Batch for Processing

```bash
curl -X POST "http://localhost:8000/v1/batches/submit" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "123e4567-e89b-12d3-a456-426614174000",
    "device_id": "unique-device-identifier",
    "settings": {
      "language": "ru",
      "analysis_depth": "detailed",
      "include_research": true,
      "generate_tags": true
    },
    "recordings": [
      {
        "id": "recording-uuid-1",
        "filename": "recording_1234567890.m4a",
        "duration": 125.5,
        "created_at": "2025-09-18T14:30:00Z",
        "audio_data": "base64-encoded-audio-content"
      }
    ]
  }'
```

### Get Batch Status

```bash
curl -X GET "http://localhost:8000/v1/batches/{batch_id}/status" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Chat Query

```bash
curl -X POST "http://localhost:8000/v1/chat/query" \
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

This comprehensive protocol provides everything needed to develop a production-ready Python server for the AI Conspects application! 🚀
