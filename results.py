"""
Results retrieval endpoints for processed notes
"""

import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from database import get_db, User, ProcessingBatch, Note, Tag, Source, NoteRelationship
from schemas import BatchResultsResponse, NoteResponse, TagResponse, SourceResponse
from auth import get_current_user

router = APIRouter(prefix="/results", tags=["results"])

def format_note_response(note: Note) -> Dict[str, Any]:
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
            "word_count": len(note.content.split()) if note.content else 0,
            "complexity_level": "intermediate"  # Can add AI analysis
        },
        "sources": [{"title": s.title, "url": s.url, "type": s.type, "relevance": s.relevance} for s in note.sources],
        "related_topics": [rel.note_2.title for rel in note.relationships_1],
        "created_at": note.created_at.isoformat()
    }

def format_relationship_response(rel: NoteRelationship) -> Dict[str, Any]:
    """Format note relationship for API response"""
    return {
        "id": str(rel.id),
        "note_1_id": str(rel.note_1_id),
        "note_2_id": str(rel.note_2_id),
        "relationship_type": rel.relationship_type,
        "strength": rel.strength,
        "description": rel.description,
        "created_at": rel.created_at.isoformat()
    }

@router.get("/{batch_id}/results", response_model=BatchResultsResponse)
async def get_batch_results(
    batch_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get processed notes results"""
    
    batch = db.query(ProcessingBatch).filter(
        ProcessingBatch.id == uuid.UUID(batch_id),
        ProcessingBatch.user_id == current_user.id
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if batch.status != "completed":
        raise HTTPException(status_code=400, detail="Batch not completed yet")
    
    # Get all notes from batch
    notes = db.query(Note).filter(
        Note.user_id == current_user.id,
        Note.source_recording_id.in_(
            db.query(AudioRecording.id).filter(AudioRecording.batch_id == batch.id)
        )
    ).all()
    
    # Calculate statistics
    total_processing_time = (batch.updated_at - batch.created_at).total_seconds()
    avg_confidence = sum(note.confidence for note in notes) / len(notes) if notes else 0
    
    # Get note relationships
    note_ids = [note.id for note in notes]
    relationships = []
    if note_ids:
        relationships = db.query(NoteRelationship).filter(
            NoteRelationship.note_1_id.in_(note_ids)
        ).all()
    
    return BatchResultsResponse(
        batch_id=str(batch.id),
        status=batch.status,
        processing_summary={
            "total_recordings": len(batch.audio_recordings),
            "notes_generated": len(notes),
            "total_processing_time": int(total_processing_time),
            "confidence_avg": round(avg_confidence, 2)
        },
        notes=[format_note_response(note) for note in notes],
        relationships=[format_relationship_response(rel) for rel in relationships]
    )

@router.get("/notes")
async def get_user_notes(
    limit: int = 20,
    offset: int = 0,
    search: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's notes with optional search"""
    
    query = db.query(Note).filter(Note.user_id == current_user.id)
    
    if search:
        # Simple text search - can be enhanced with vector search
        query = query.filter(
            Note.title.contains(search) | 
            Note.content.contains(search) |
            Note.summary.contains(search)
        )
    
    notes = query.order_by(Note.created_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "notes": [format_note_response(note) for note in notes],
        "total": query.count(),
        "limit": limit,
        "offset": offset
    }

@router.get("/notes/{note_id}")
async def get_note(
    note_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific note by ID"""
    
    note = db.query(Note).filter(
        Note.id == uuid.UUID(note_id),
        Note.user_id == current_user.id
    ).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    return format_note_response(note)

@router.get("/notes/{note_id}/related")
async def get_related_notes(
    note_id: str,
    limit: int = 5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get notes related to a specific note"""
    
    note = db.query(Note).filter(
        Note.id == uuid.UUID(note_id),
        Note.user_id == current_user.id
    ).first()
    
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Get related notes through relationships
    relationships = db.query(NoteRelationship).filter(
        NoteRelationship.note_1_id == note.id
    ).limit(limit).all()
    
    related_notes = []
    for rel in relationships:
        related_note = db.query(Note).filter(Note.id == rel.note_2_id).first()
        if related_note:
            related_notes.append({
                "note": format_note_response(related_note),
                "relationship": format_relationship_response(rel)
            })
    
    return {
        "note_id": note_id,
        "related_notes": related_notes,
        "total_relationships": len(relationships)
    }

@router.get("/tags")
async def get_user_tags(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's tags"""
    
    tags = db.query(Tag).filter(Tag.user_id == current_user.id).all()
    
    return {
        "tags": [
            {
                "id": str(tag.id),
                "name": tag.name,
                "color": tag.color,
                "created_at": tag.created_at.isoformat(),
                "note_count": len(tag.notes)
            }
            for tag in tags
        ]
    }

@router.get("/search")
async def search_notes(
    q: str,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search notes by query"""
    
    # Simple text search - can be enhanced with vector search
    notes = db.query(Note).filter(
        Note.user_id == current_user.id,
        Note.title.contains(q) | 
        Note.content.contains(q) |
        Note.summary.contains(q)
    ).order_by(Note.created_at.desc()).limit(limit).all()
    
    return {
        "query": q,
        "notes": [format_note_response(note) for note in notes],
        "total": len(notes)
    }
