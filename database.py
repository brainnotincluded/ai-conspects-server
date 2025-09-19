"""
Database models and configuration for AI Conspects Server
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, String, Text, Float, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from config import settings

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    settings = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    
    # Relationships
    processing_batches = relationship("ProcessingBatch", back_populates="user")
    audio_recordings = relationship("AudioRecording", back_populates="user")
    notes = relationship("Note", back_populates="user")
    tags = relationship("Tag", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")

class ProcessingBatch(Base):
    __tablename__ = "processing_batches"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    estimated_time = Column(Integer)
    settings = Column(JSONB, default={})
    progress = Column(JSONB, default={})
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="processing_batches")
    audio_recordings = relationship("AudioRecording", back_populates="batch")

class AudioRecording(Base):
    __tablename__ = "audio_recordings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), ForeignKey("processing_batches.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    s3_key = Column(String(500))
    duration = Column(Float, nullable=False)
    file_size = Column(Integer)
    format = Column(String(10))
    sample_rate = Column(Integer)
    channels = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_processed = Column(Boolean, default=False)
    
    # Relationships
    batch = relationship("ProcessingBatch", back_populates="audio_recordings")
    user = relationship("User", back_populates="audio_recordings")
    notes = relationship("Note", back_populates="source_recording")

class Note(Base):
    __tablename__ = "notes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    source_recording_id = Column(UUID(as_uuid=True), ForeignKey("audio_recordings.id", ondelete="SET NULL"))
    title = Column(String(500), nullable=False)
    summary = Column(Text)
    content = Column(Text, nullable=False)
    type = Column(String(20), default="recording")
    confidence = Column(Float, default=1.0)
    reading_time = Column(Integer, default=5)
    view_count = Column(Integer, default=0)
    is_read = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    embedding = Column(Vector(1536))  # OpenAI embedding dimension
    
    # Relationships
    user = relationship("User", back_populates="notes")
    source_recording = relationship("AudioRecording", back_populates="notes")
    tags = relationship("Tag", secondary="note_tags", back_populates="notes")
    sources = relationship("Source", back_populates="note")
    relationships_1 = relationship("NoteRelationship", foreign_keys="NoteRelationship.note_1_id", back_populates="note_1")
    relationships_2 = relationship("NoteRelationship", foreign_keys="NoteRelationship.note_2_id", back_populates="note_2")

class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(50), nullable=False)
    color = Column(String(20), default="blue")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="tags")
    notes = relationship("Note", secondary="note_tags", back_populates="tags")

class NoteTag(Base):
    __tablename__ = "note_tags"
    
    note_id = Column(UUID(as_uuid=True), ForeignKey("notes.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)

class Source(Base):
    __tablename__ = "sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    note_id = Column(UUID(as_uuid=True), ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=False)
    url = Column(String(1000))
    type = Column(String(50), nullable=False)
    relevance = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    note = relationship("Note", back_populates="sources")

class NoteRelationship(Base):
    __tablename__ = "note_relationships"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    note_1_id = Column(UUID(as_uuid=True), ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    note_2_id = Column(UUID(as_uuid=True), ForeignKey("notes.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    strength = Column(Float, default=0.5)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    note_1 = relationship("Note", foreign_keys=[note_1_id], back_populates="relationships_1")
    note_2 = relationship("Note", foreign_keys=[note_2_id], back_populates="relationships_2")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(UUID(as_uuid=True))
    content = Column(Text, nullable=False)
    is_from_user = Column(Boolean, nullable=False)
    message_type = Column(String(20), default="text")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="chat_messages")

# Database initialization
async def init_db():
    """Initialize database tables"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise
