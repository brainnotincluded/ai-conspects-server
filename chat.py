"""
AI Chat API for conversational interaction with notes
"""

import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
import logging

from database import get_db, User, Note, ChatMessage
from schemas import ChatRequest, ChatResponse
from auth import get_current_user
from llm_services import get_llm_service
from config import settings

router = APIRouter(prefix="/chat", tags=["ai chat"])
logger = logging.getLogger(__name__)

# Global embedding model
embedding_model = None

def get_embedding_model():
    """Get or initialize embedding model"""
    global embedding_model
    if embedding_model is None and SentenceTransformer is not None:
        logger.info("Loading embedding model")
        embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("Embedding model loaded successfully")
    return embedding_model

async def find_relevant_notes(
    query: str, 
    user_id: str, 
    db: Session,
    context: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[Note]:
    """Find relevant notes using vector search"""
    
    try:
        # Generate embedding for query
        model = get_embedding_model()
        query_embedding = model.encode([query])[0]
        
        # Vector search in DB (simplified - would need proper vector search setup)
        # For now, use text search as fallback
        notes = db.query(Note).filter(
            Note.user_id == uuid.UUID(user_id),
            Note.title.contains(query) | 
            Note.content.contains(query) |
            Note.summary.contains(query)
        ).limit(limit).all()
        
        return notes
        
    except Exception as e:
        logger.error(f"Error finding relevant notes: {e}")
        # Fallback to recent notes
        return db.query(Note).filter(
            Note.user_id == uuid.UUID(user_id)
        ).order_by(Note.created_at.desc()).limit(5).all()

def format_notes_for_context(notes: List[Note]) -> str:
    """Format notes for LLM context"""
    if not notes:
        return "No relevant notes found."
    
    context_parts = []
    for note in notes:
        context_parts.append(f"Title: {note.title}")
        if note.summary:
            context_parts.append(f"Summary: {note.summary}")
        context_parts.append(f"Content: {note.content[:500]}...")
        context_parts.append("---")
    
    return "\n".join(context_parts)

async def save_chat_messages(
    db: Session,
    user_id: str,
    user_message: str,
    ai_response: str,
    conversation_id: Optional[str] = None,
    related_notes: List[Note] = None
) -> str:
    """Save chat messages to database"""
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Save user message
    user_msg = ChatMessage(
        user_id=uuid.UUID(user_id),
        conversation_id=uuid.UUID(conversation_id),
        content=user_message,
        is_from_user=True
    )
    db.add(user_msg)
    
    # Save AI response
    ai_msg = ChatMessage(
        user_id=uuid.UUID(user_id),
        conversation_id=uuid.UUID(conversation_id),
        content=ai_response,
        is_from_user=False
    )
    db.add(ai_msg)
    
    db.commit()
    
    return conversation_id

async def get_conversation_history(
    db: Session,
    conversation_id: str,
    user_id: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get conversation history"""
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.conversation_id == uuid.UUID(conversation_id),
        ChatMessage.user_id == uuid.UUID(user_id)
    ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
    
    return [
        {
            "content": msg.content,
            "is_from_user": msg.is_from_user,
            "created_at": msg.created_at.isoformat()
        }
        for msg in reversed(messages)
    ]

@router.post("/query", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process AI chat query"""
    
    try:
        # Find relevant notes
        relevant_notes = await find_relevant_notes(
            request.message, 
            str(current_user.id),
            db,
            request.context
        )
        
        # Format context for LLM
        context_text = format_notes_for_context(relevant_notes)
        
        # Get conversation history if conversation_id provided
        conversation_history = None
        if request.conversation_id:
            conversation_history = await get_conversation_history(
                db, request.conversation_id, str(current_user.id)
            )
        
        # Determine LLM mode from settings or request
        llm_mode = "ollama"  # Default
        if request.settings and "llm_mode" in request.settings:
            llm_mode = request.settings["llm_mode"]
        elif current_user.settings and "llm_mode" in current_user.settings:
            llm_mode = current_user.settings["llm_mode"]
        
        # Get LLM service
        llm_service = get_llm_service(mode=llm_mode)
        
        # Generate response
        response = await llm_service.generate_chat_response(
            user_message=request.message,
            context=context_text,
            conversation_history=conversation_history
        )
        
        # Save messages to DB
        conversation_id = await save_chat_messages(
            db=db,
            user_id=str(current_user.id),
            user_message=request.message,
            ai_response=response["message"],
            conversation_id=request.conversation_id,
            related_notes=relevant_notes
        )
        
        # Generate suggested actions
        suggested_actions = generate_suggested_actions(request.message, relevant_notes)
        
        return ChatResponse(
            response=response,
            context_used=[
                {
                    "note_id": str(note.id), 
                    "title": note.title, 
                    "relevance": 0.9
                } 
                for note in relevant_notes
            ],
            suggested_actions=suggested_actions,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}")

def generate_suggested_actions(message: str, relevant_notes: List[Note]) -> List[Dict[str, Any]]:
    """Generate suggested actions based on message and context"""
    
    actions = []
    
    # Add note-related actions
    if relevant_notes:
        actions.append({
            "type": "view_notes",
            "title": "View Related Notes",
            "description": f"View {len(relevant_notes)} related notes",
            "data": {"note_ids": [str(note.id) for note in relevant_notes]}
        })
    
    # Add search actions based on message content
    if any(word in message.lower() for word in ["search", "find", "look for"]):
        actions.append({
            "type": "search",
            "title": "Search Notes",
            "description": "Search through all your notes",
            "data": {"query": message}
        })
    
    # Add tag actions
    if any(word in message.lower() for word in ["tag", "categorize", "organize"]):
        actions.append({
            "type": "manage_tags",
            "title": "Manage Tags",
            "description": "Organize your notes with tags",
            "data": {}
        })
    
    # Add export actions
    if any(word in message.lower() for word in ["export", "download", "save"]):
        actions.append({
            "type": "export",
            "title": "Export Notes",
            "description": "Export your notes in various formats",
            "data": {}
        })
    
    return actions

@router.get("/conversations")
async def get_conversations(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's conversations"""
    
    # Get unique conversations
    conversations = db.query(ChatMessage.conversation_id).filter(
        ChatMessage.user_id == current_user.id
    ).distinct().limit(limit).all()
    
    conversation_list = []
    for conv in conversations:
        # Get last message for preview
        last_message = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conv.conversation_id,
            ChatMessage.user_id == current_user.id
        ).order_by(ChatMessage.created_at.desc()).first()
        
        # Get message count
        message_count = db.query(ChatMessage).filter(
            ChatMessage.conversation_id == conv.conversation_id,
            ChatMessage.user_id == current_user.id
        ).count()
        
        conversation_list.append({
            "conversation_id": str(conv.conversation_id),
            "last_message": last_message.content if last_message else "",
            "message_count": message_count,
            "last_updated": last_message.created_at.isoformat() if last_message else None
        })
    
    return {
        "conversations": conversation_list,
        "total": len(conversation_list)
    }

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get messages from a specific conversation"""
    
    messages = db.query(ChatMessage).filter(
        ChatMessage.conversation_id == uuid.UUID(conversation_id),
        ChatMessage.user_id == current_user.id
    ).order_by(ChatMessage.created_at.asc()).limit(limit).all()
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "id": str(msg.id),
                "content": msg.content,
                "is_from_user": msg.is_from_user,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ],
        "total": len(messages)
    }
