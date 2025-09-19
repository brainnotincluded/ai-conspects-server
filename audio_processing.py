"""
Audio processing with Whisper Large for speech-to-text
"""

import whisper
import base64
import uuid
import asyncio
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from database import get_db, User, ProcessingBatch, AudioRecording, Note, Tag, Source
from schemas import BatchSubmitRequest, BatchStatusResponse, RecordingData, ProcessingSettings
from auth import get_current_user
from llm_services import get_llm_service
from perplexity_service import perplexity_service
from config import settings

router = APIRouter(prefix="/batches", tags=["audio processing"])
logger = logging.getLogger(__name__)

# Global Whisper model instance
whisper_model = None

def get_whisper_model():
    """Get or initialize Whisper model"""
    global whisper_model
    if whisper_model is None:
        logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
        whisper_model = whisper.load_model(settings.WHISPER_MODEL)
        logger.info("Whisper model loaded successfully")
    return whisper_model

async def process_audio_recording(recording_id: str, audio_data: bytes, user_id: str, settings: ProcessingSettings) -> Dict[str, Any]:
    """Process audio recording with Whisper and generate notes"""
    try:
        # Load Whisper model
        model = get_whisper_model()
        
        # Transcribe audio
        logger.info(f"Transcribing audio for recording {recording_id}")
        result = model.transcribe(BytesIO(audio_data), language=settings.language)
        
        transcript = clean_transcript(result['text'])
        confidence = result.get('confidence', 0.9)
        
        logger.info(f"Transcription completed for {recording_id}")
        
        # Generate notes using selected LLM mode
        llm_service = get_llm_service(
            mode=settings.llm_mode,
            model=settings.ollama_model if settings.llm_mode == "ollama" else settings.openrouter_model
        )
        
        # Get user context for better note generation
        user_context = await get_user_context(user_id)
        
        # Generate notes
        logger.info(f"Generating notes for {recording_id} using {settings.llm_mode}")
        note_data = await llm_service.generate_notes(
            transcript=transcript,
            context=user_context,
            settings=settings.dict()
        )
        
        # Research and augment if enabled
        research_data = None
        if settings.include_research and perplexity_service:
            logger.info(f"Researching additional information for {recording_id}")
            research_data = await perplexity_service.research_note_topic(
                note_title=note_data.get("title", ""),
                note_content=note_data.get("content", ""),
                key_phrases=note_data.get("key_points", [])
            )
        
        return {
            "recording_id": recording_id,
            "transcript": transcript,
            "confidence": confidence,
            "note_data": note_data,
            "research_data": research_data,
            "language": result.get('language', settings.language)
        }
        
    except Exception as e:
        logger.error(f"Error processing audio {recording_id}: {e}")
        return {
            "recording_id": recording_id,
            "error": str(e),
            "transcript": "",
            "confidence": 0.0
        }

def clean_transcript(text: str) -> str:
    """Clean and format transcript"""
    # Remove extra spaces and characters
    text = ' '.join(text.split())
    
    # Remove repetitive phrases (simple implementation)
    words = text.split()
    cleaned_words = []
    prev_word = ""
    repeat_count = 0
    
    for word in words:
        if word.lower() == prev_word.lower():
            repeat_count += 1
            if repeat_count < 2:  # Allow max 2 repetitions
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            repeat_count = 0
        prev_word = word
    
    return ' '.join(cleaned_words)

async def get_user_context(user_id: str) -> str:
    """Get user context for better note generation"""
    # This would typically fetch recent notes, tags, etc.
    # For now, return a simple context
    return f"User ID: {user_id} - Recent notes and context will be added here"

async def save_audio_to_storage(audio_data: bytes, s3_key: str) -> bool:
    """Save audio to S3 storage"""
    try:
        import boto3
        from config import settings
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION
        )
        
        s3_client.put_object(
            Bucket=settings.S3_BUCKET,
            Key=s3_key,
            Body=audio_data,
            ContentType='audio/m4a'
        )
        
        return True
    except Exception as e:
        logger.error(f"Error saving audio to S3: {e}")
        return False

async def create_note_from_processing(
    db: Session,
    user_id: str,
    recording_id: str,
    processing_result: Dict[str, Any]
) -> Note:
    """Create note from processing result"""
    note_data = processing_result.get("note_data", {})
    research_data = processing_result.get("research_data", {})
    
    # Create note
    note = Note(
        user_id=uuid.UUID(user_id),
        source_recording_id=uuid.UUID(recording_id),
        title=note_data.get("title", "Заметка из аудио"),
        summary=note_data.get("summary", ""),
        content=note_data.get("content", ""),
        confidence=processing_result.get("confidence", 0.9),
        reading_time=note_data.get("reading_time", 5)
    )
    
    db.add(note)
    db.flush()  # Get the ID
    
    # Create tags
    if note_data.get("tags"):
        for tag_name in note_data["tags"]:
            # Get or create tag
            tag = db.query(Tag).filter(
                Tag.user_id == uuid.UUID(user_id),
                Tag.name == tag_name
            ).first()
            
            if not tag:
                tag = Tag(
                    user_id=uuid.UUID(user_id),
                    name=tag_name,
                    color="blue"
                )
                db.add(tag)
                db.flush()
            
            # Associate tag with note
            note.tags.append(tag)
    
    # Add research sources
    if research_data.get("sources"):
        for source_data in research_data["sources"]:
            source = Source(
                note_id=note.id,
                title=source_data.get("title", "Unknown"),
                url=source_data.get("url", ""),
                type=source_data.get("type", "web"),
                relevance=source_data.get("relevance", 0.7)
            )
            db.add(source)
    
    return note

@router.post("/submit")
async def submit_batch(
    request: BatchSubmitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit audio recordings batch for processing"""
    
    # Request validation
    if not request.recordings:
        raise HTTPException(status_code=400, detail="No recordings provided")
    
    if len(request.recordings) > settings.MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Too many recordings in batch. Maximum: {settings.MAX_BATCH_SIZE}")
    
    # Validate recording durations
    for recording in request.recordings:
        if recording.duration > settings.MAX_RECORDING_DURATION:
            raise HTTPException(
                status_code=400, 
                detail=f"Recording {recording.id} exceeds maximum duration of {settings.MAX_RECORDING_DURATION} seconds"
            )
    
    # Create batch in DB
    batch = ProcessingBatch(
        id=uuid.UUID(request.batch_id),
        user_id=current_user.id,
        status="accepted",
        settings=request.settings.dict(),
        estimated_time=len(request.recordings) * 2  # 2 minutes per recording
    )
    
    # Save recordings and start processing
    for recording_data in request.recordings:
        # Decode and save audio
        try:
            audio_bytes = base64.b64decode(recording_data.audio_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio data for recording {recording_data.id}: {e}")
        
        s3_key = f"audio/{current_user.id}/{request.batch_id}/{recording_data.id}.m4a"
        
        # Save to S3 (async)
        background_tasks.add_task(save_audio_to_storage, audio_bytes, s3_key)
        
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
    db.commit()
    
    # Start background processing
    background_tasks.add_task(process_batch_async, str(batch.id), str(current_user.id))
    
    return {
        "batch_id": str(batch.id),
        "status": "accepted",
        "estimated_time": batch.estimated_time,
        "queue_position": 1,  # Simple implementation
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

async def process_batch_async(batch_id: str, user_id: str):
    """Process batch asynchronously"""
    db = SessionLocal()
    try:
        # Get batch
        batch = db.query(ProcessingBatch).filter(ProcessingBatch.id == uuid.UUID(batch_id)).first()
        if not batch:
            logger.error(f"Batch {batch_id} not found")
            return
        
        # Update status
        batch.status = "processing"
        batch.progress = {"current_step": "transcription", "completed_steps": []}
        db.commit()
        
        # Process each recording
        recordings = db.query(AudioRecording).filter(AudioRecording.batch_id == batch.id).all()
        
        for i, recording in enumerate(recordings):
            try:
                # Download audio from S3
                import boto3
                from config import settings
                
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.S3_ACCESS_KEY,
                    aws_secret_access_key=settings.S3_SECRET_KEY,
                    region_name=settings.S3_REGION
                )
                
                audio_obj = s3_client.get_object(Bucket=settings.S3_BUCKET, Key=recording.s3_key)
                audio_data = audio_obj['Body'].read()
                
                # Process audio
                processing_result = await process_audio_recording(
                    recording_id=str(recording.id),
                    audio_data=audio_data,
                    user_id=user_id,
                    settings=ProcessingSettings(**batch.settings)
                )
                
                if "error" not in processing_result:
                    # Create note from processing result
                    note = await create_note_from_processing(
                        db, user_id, str(recording.id), processing_result
                    )
                    
                    # Mark recording as processed
                    recording.is_processed = True
                    
                    logger.info(f"Successfully processed recording {recording.id}")
                else:
                    logger.error(f"Error processing recording {recording.id}: {processing_result['error']}")
                    recording.is_processed = False
                
                # Update progress
                batch.progress = {
                    "current_step": "note_generation",
                    "completed_steps": ["audio_upload", "transcription", "content_analysis"],
                    "progress_percentage": int((i + 1) / len(recordings) * 100)
                }
                db.commit()
                
            except Exception as e:
                logger.error(f"Error processing recording {recording.id}: {e}")
                recording.is_processed = False
                db.commit()
        
        # Mark batch as completed
        batch.status = "completed"
        batch.progress = {
            "current_step": "completed",
            "completed_steps": ["audio_upload", "transcription", "content_analysis", "research_augmentation", "note_generation"],
            "progress_percentage": 100
        }
        db.commit()
        
        logger.info(f"Batch {batch_id} processing completed")
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")
        batch.status = "failed"
        batch.error_message = str(e)
        db.commit()
    finally:
        db.close()

@router.get("/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get batch processing status"""
    
    batch = db.query(ProcessingBatch).filter(
        ProcessingBatch.id == uuid.UUID(batch_id),
        ProcessingBatch.user_id == current_user.id
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Calculate progress
    total_recordings = db.query(AudioRecording).filter(AudioRecording.batch_id == batch.id).count()
    processed_recordings = db.query(AudioRecording).filter(
        AudioRecording.batch_id == batch.id,
        AudioRecording.is_processed == True
    ).count()
    
    progress_percentage = (processed_recordings / total_recordings * 100) if total_recordings > 0 else 0
    
    return BatchStatusResponse(
        batch_id=str(batch.id),
        status=batch.status,
        progress={
            "current_step": batch.progress.get("current_step", "pending"),
            "completed_steps": batch.progress.get("completed_steps", []),
            "progress_percentage": int(progress_percentage),
            "estimated_remaining": batch.estimated_time - batch.progress.get("elapsed_time", 0)
        },
        recordings_processed=processed_recordings,
        total_recordings=total_recordings,
        created_at=batch.created_at.isoformat(),
        updated_at=batch.updated_at.isoformat()
    )
