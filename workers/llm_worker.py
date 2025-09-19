"""
Celery worker for LLM processing tasks
"""

import json
from typing import Dict, Any
import logging
from workers.celery_app import celery_app
from llm_services import get_llm_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def generate_notes_task(self, recording_id: str, transcript: str, user_id: str, settings_dict: Dict[str, Any]):
    """Generate notes from transcript using LLM"""
    
    try:
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'initializing_llm', 'total': 100})
        
        # Get LLM service
        llm_mode = settings_dict.get('llm_mode', 'ollama')
        llm_service = get_llm_service(mode=llm_mode)
        
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'generating_notes', 'total': 100})
        
        # Get user context
        user_context = get_user_context(user_id)
        
        # Generate notes
        note_data = await llm_service.generate_notes(
            transcript=transcript,
            context=user_context,
            settings=settings_dict
        )
        
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'completed', 'total': 100})
        
        return {
            'recording_id': recording_id,
            'note_data': note_data,
            'user_id': user_id
        }
        
    except Exception as e:
        logger.error(f"Error generating notes for {recording_id}: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

def get_user_context(user_id: str) -> str:
    """Get user context for better note generation"""
    # This would typically fetch recent notes, tags, etc.
    # For now, return a simple context
    return f"User ID: {user_id} - Recent notes and context will be added here"

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response into structured data"""
    try:
        # Try to extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except:
        pass
    
    # Fallback parsing
    return create_fallback_response(response_text)

def create_fallback_response(text: str) -> Dict[str, Any]:
    """Create fallback response when parsing fails"""
    return {
        "title": "Заметка из аудио",
        "summary": text[:200] + "..." if len(text) > 200 else text,
        "content": text,
        "tags": ["аудио", "заметка"],
        "key_points": [],
        "reading_time": max(1, len(text.split()) // 200)
    }
