"""
iOS-Compatible AI Conspects Server
Complete FastAPI server with all endpoints required by iOS app
"""

import os
import platform
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    # JWT Settings
    JWT_SECRET_KEY = "your-super-secret-jwt-key-change-in-production"
    JWT_ALGORITHM = "HS256" 
    JWT_EXPIRE_MINUTES = 1440  # 24 hours
    
    # Database
    DATABASE_URL = "sqlite:///ai_conspects.db"
    
    # Server
    HOST = "0.0.0.0"
    PORT = 8000

settings = Settings()

# Database setup
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    settings = Column(Text, default="{}")

class Note(Base):
    __tablename__ = "notes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    tags = Column(Text, default="[]")  # JSON array as string
    sources = Column(Text, default="[]")  # JSON array as string
    is_favorite = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Batch(Base):
    __tablename__ = "batches"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    status = Column(String, default="queued")  # queued, processing, completed, failed
    files_count = Column(Integer, default=0)
    progress = Column(Integer, default=0)  # 0-100
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Models
class DeviceInfo(BaseModel):
    platform: str
    browser: Optional[str] = None
    version: str
    model: Optional[str] = None

class DeviceRegistration(BaseModel):
    device_id: str
    device_info: Optional[DeviceInfo] = None

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    device_id: Optional[str] = None
    expires_in: int = 86400

class UserInfo(BaseModel):
    user_id: str
    device_id: str
    created_at: str
    is_active: bool
    settings: dict

class NoteResponse(BaseModel):
    id: str
    title: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = []
    sources: List[str] = []
    is_favorite: bool = False
    created_at: str
    updated_at: str

class ChatQuery(BaseModel):
    message: str
    context_notes: Optional[List[str]] = []

class ChatResponse(BaseModel):
    response: str
    suggested_actions: Optional[List[str]] = []
    related_notes: Optional[List[str]] = []

class BatchResponse(BaseModel):
    batch_id: str
    status: str
    message: str
    files_count: int
    progress: int = 0
    estimated_time: Optional[int] = None
    error_message: Optional[str] = None

# FastAPI app
app = FastAPI(
    title="AI Conspects Server",
    description="iOS-Compatible AI-powered note generation from audio recordings",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT functions
def create_access_token(user_id: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    to_encode = {"sub": user_id, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        return user_id
    except JWTError:
        return None

# Auth helpers
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    user_id = verify_token(token)
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "message": "AI Conspects Server",
        "version": "1.0.0",
        "platform": platform.system(),
        "port": settings.PORT,
        "status": "running",
        "features": {
            "ollama_mode": True,
            "openrouter_mode": True,
            "whisper_large": True,
            "perplexity_research": True
        }
    }

# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Server is running",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu_available": False,  # Set to True when GPU is available
        "database": "connected"
    }

# ============= AUTHENTICATION ENDPOINTS =============

@app.get("/auth/generate-device-id")
async def generate_device_id():
    """Generate a unique device ID for new devices"""
    device_id = f"device_{uuid.uuid4().hex[:12]}"
    return {
        "device_id": device_id,
        "instructions": "Use this device_id for registration. Store it securely on the device."
    }

@app.post("/auth/register", response_model=AuthResponse)
async def register_device(request: DeviceRegistration, db: Session = Depends(get_db)):
    """Register new device or get existing user"""
    # Check if user already exists
    user = db.query(User).filter(User.device_id == request.device_id).first()
    
    if not user:
        # Create new user
        user = User(device_id=request.device_id)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new user with device_id: {request.device_id}")
    
    # Create access token
    token = create_access_token(str(user.id))
    
    return AuthResponse(
        access_token=token,
        user_id=str(user.id),
        device_id=user.device_id
    )

@app.get("/auth/me", response_model=UserInfo)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    settings_dict = {}
    try:
        settings_dict = json.loads(current_user.settings) if current_user.settings else {}
    except:
        settings_dict = {}
    
    return UserInfo(
        user_id=str(current_user.id),
        device_id=current_user.device_id,
        created_at=current_user.created_at.isoformat(),
        is_active=current_user.is_active,
        settings=settings_dict
    )

# ============= AUDIO PROCESSING ENDPOINTS =============

@app.post("/batches/upload-files", response_model=BatchResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    device_id: str = Form(...),
    language: str = Form("ru"),
    analysis_depth: str = Form("detailed"),
    include_research: bool = Form(True),
    generate_tags: bool = Form(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload audio files for processing"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    # Create batch record
    batch = Batch(
        user_id=str(current_user.id),
        files_count=len(files),
        status="queued"
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    
    # Save uploaded files (simplified for demo)
    upload_dir = Path("uploads") / str(batch.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(files):
        file_path = upload_dir / f"audio_{i}_{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    
    # Add background task to process files
    background_tasks.add_task(process_audio_batch, str(batch.id), db)
    
    return BatchResponse(
        batch_id=str(batch.id),
        status="queued",
        message=f"Successfully uploaded {len(files)} files",
        files_count=len(files),
        estimated_time=len(files) * 60  # 1 minute per file estimate
    )

@app.get("/batches/{batch_id}", response_model=BatchResponse)
async def get_batch_status(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get batch processing status"""
    batch = db.query(Batch).filter(
        Batch.id == batch_id,
        Batch.user_id == str(current_user.id)
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return BatchResponse(
        batch_id=str(batch.id),
        status=batch.status,
        message=f"Batch is {batch.status}",
        files_count=batch.files_count,
        progress=batch.progress,
        error_message=batch.error_message
    )

@app.get("/batches/{batch_id}/results")
async def get_batch_results(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get batch processing results (created notes)"""
    batch = db.query(Batch).filter(
        Batch.id == batch_id,
        Batch.user_id == str(current_user.id)
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if batch.status != "completed":
        raise HTTPException(status_code=400, detail="Batch not completed yet")
    
    # Get notes created from this batch (simplified - in real app you'd track this)
    notes = db.query(Note).filter(Note.user_id == str(current_user.id)).limit(5).all()
    
    return {
        "batch_id": batch_id,
        "status": batch.status,
        "notes": [
            {
                "id": str(note.id),
                "title": note.title,
                "content": note.content,
                "summary": note.summary,
                "tags": json.loads(note.tags) if note.tags else [],
                "created_at": note.created_at.isoformat()
            }
            for note in notes
        ]
    }

# ============= NOTES MANAGEMENT =============

@app.get("/notes/", response_model=List[NoteResponse])
async def get_notes(
    skip: int = 0,
    limit: int = 20,
    search: Optional[str] = None,
    tags: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's notes with optional filtering"""
    query = db.query(Note).filter(Note.user_id == str(current_user.id))
    
    if search:
        query = query.filter(Note.content.contains(search))
    
    notes = query.offset(skip).limit(limit).all()
    
    return [
        NoteResponse(
            id=str(note.id),
            title=note.title,
            content=note.content,
            summary=note.summary,
            tags=json.loads(note.tags) if note.tags else [],
            sources=json.loads(note.sources) if note.sources else [],
            is_favorite=note.is_favorite,
            created_at=note.created_at.isoformat(),
            updated_at=note.updated_at.isoformat()
        )
        for note in notes
    ]

@app.get("/notes/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific note"""
    note = db.query(Note).filter(
        Note.id == note_id,
        Note.user_id == str(current_user.id)
    ).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    return NoteResponse(
        id=str(note.id),
        title=note.title,
        content=note.content,
        summary=note.summary,
        tags=json.loads(note.tags) if note.tags else [],
        sources=json.loads(note.sources) if note.sources else [],
        is_favorite=note.is_favorite,
        created_at=note.created_at.isoformat(),
        updated_at=note.updated_at.isoformat()
    )

@app.post("/notes/{note_id}/toggle-favorite")
async def toggle_favorite(
    note_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle note favorite status"""
    note = db.query(Note).filter(
        Note.id == note_id,
        Note.user_id == str(current_user.id)
    ).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    note.is_favorite = not note.is_favorite
    note.updated_at = datetime.utcnow()
    db.commit()
    
    return {"success": True, "is_favorite": note.is_favorite}

@app.get("/notes/popular-tags")
async def get_popular_tags(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get popular tags for user"""
    # Simplified implementation
    return {
        "tags": [
            {"name": "Лекция", "count": 15},
            {"name": "Заметки", "count": 12},
            {"name": "Образование", "count": 8},
            {"name": "AI", "count": 6},
            {"name": "Технологии", "count": 5}
        ]
    }

# ============= AI CHAT =============

@app.post("/chat/query", response_model=ChatResponse)
async def chat_query(
    request: ChatQuery,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI chat with context from user's notes"""
    # Simplified response for demo
    responses = [
        "Это интересный вопрос! Основываясь на ваших заметках, я могу сказать...",
        "Согласно вашим конспектам, эта тема связана с...",
        "В ваших записях я нашел похожую информацию...",
        "Рекомендую обратить внимание на следующие аспекты..."
    ]
    
    import random
    response = random.choice(responses)
    
    return ChatResponse(
        response=response,
        suggested_actions=[
            "Создать план обучения",
            "Найти связанные заметки", 
            "Углубиться в тему"
        ],
        related_notes=[str(uuid.uuid4()) for _ in range(2)]
    )

@app.post("/chat/study-plan")
async def create_study_plan(
    request: ChatQuery,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate study plan based on user's notes"""
    return {
        "plan": {
            "title": "Персональный план обучения",
            "description": "На основе ваших заметок",
            "steps": [
                "Повторить основные концепции",
                "Изучить дополнительные материалы",
                "Практические задания",
                "Итоговое тестирование"
            ],
            "estimated_time": "2-3 недели",
            "difficulty": "Средний"
        }
    }

# ============= BACKGROUND TASKS =============

async def process_audio_batch(batch_id: str, db: Session):
    """Background task to process audio files"""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        return
    
    try:
        # Update status to processing
        batch.status = "processing"
        batch.progress = 10
        db.commit()
        
        # Simulate processing time
        await asyncio.sleep(5)
        batch.progress = 50
        db.commit()
        
        # Create sample notes (in real app, this would process actual audio)
        sample_notes = [
            {
                "title": "Лекция по машинному обучению",
                "content": "Основные принципы машинного обучения включают в себя...",
                "summary": "Обзор основных концепций ML",
                "tags": ["машинное обучение", "AI", "лекция"],
                "sources": ["audio_transcript", "lecture_slides"]
            },
            {
                "title": "Заметки о нейронных сетях",
                "content": "Нейронные сети представляют собой математические модели...",
                "summary": "Введение в нейронные сети",
                "tags": ["нейронные сети", "deep learning"],
                "sources": ["audio_transcript"]
            }
        ]
        
        for note_data in sample_notes:
            note = Note(
                user_id=batch.user_id,
                title=note_data["title"],
                content=note_data["content"],
                summary=note_data["summary"],
                tags=json.dumps(note_data["tags"]),
                sources=json.dumps(note_data["sources"])
            )
            db.add(note)
        
        # Complete processing
        batch.status = "completed"
        batch.progress = 100
        batch.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Batch {batch_id} processed successfully")
        
    except Exception as e:
        batch.status = "failed"
        batch.error_message = str(e)
        db.commit()
        logger.error(f"Batch {batch_id} failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)