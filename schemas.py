"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

# Authentication Schemas
class DeviceRegistration(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str

# Audio Processing Schemas
class RecordingData(BaseModel):
    id: str
    filename: str
    duration: float
    audio_data: str = Field(..., description="Base64 encoded audio data")
    created_at: str
    metadata: Optional[Dict[str, Any]] = None

class ProcessingSettings(BaseModel):
    language: str = "ru"
    analysis_depth: str = "detailed"
    include_research: bool = True
    generate_tags: bool = True
    max_notes_per_recording: int = 3
    include_related_topics: bool = True
    llm_mode: str = "ollama"  # "ollama" or "openrouter"
    ollama_model: Optional[str] = None
    openrouter_model: Optional[str] = None

class BatchSubmitRequest(BaseModel):
    batch_id: str
    device_id: str
    settings: ProcessingSettings
    recordings: List[RecordingData]

class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    progress: Dict[str, Any]
    recordings_processed: int
    total_recordings: int
    created_at: str
    updated_at: str

class BatchResultsResponse(BaseModel):
    batch_id: str
    status: str
    processing_summary: Dict[str, Any]
    notes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

# Note Schemas
class NoteResponse(BaseModel):
    id: str
    source_recording_id: Optional[str]
    title: str
    summary: Optional[str]
    content: str
    tags: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    sources: List[Dict[str, Any]]
    related_topics: List[str]
    created_at: str

class TagResponse(BaseModel):
    name: str
    confidence: float
    color: str

class SourceResponse(BaseModel):
    title: str
    url: Optional[str]
    type: str
    relevance: float

# Chat Schemas
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: Dict[str, Any]
    context_used: List[Dict[str, Any]]
    suggested_actions: List[Dict[str, Any]]
    conversation_id: str

# LLM Mode Schemas
class LLMModeConfig(BaseModel):
    mode: str = Field(..., description="ollama or openrouter")
    model: str = Field(..., description="Model name")
    temperature: float = 0.7
    max_tokens: int = 2000

class OllamaConfig(LLMModeConfig):
    mode: str = "ollama"
    base_url: str = "http://localhost:11434"

class OpenRouterConfig(LLMModeConfig):
    mode: str = "openrouter"
    api_key: str

# Research Schemas
class ResearchRequest(BaseModel):
    query: str
    max_results: int = 5
    include_sources: bool = True

class ResearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    sources: List[SourceResponse]
    total_results: int

# Error Schemas
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
