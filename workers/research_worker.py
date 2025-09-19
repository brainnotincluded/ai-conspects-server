"""
Celery worker for research and augmentation tasks
"""

from typing import Dict, Any, List
import logging
from workers.celery_app import celery_app
from perplexity_service import perplexity_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def research_note_task(self, note_id: str, note_title: str, note_content: str, key_phrases: List[str], user_id: str):
    """Research and augment note with additional information"""
    
    try:
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'researching', 'total': 100})
        
        if not perplexity_service:
            logger.warning("Perplexity service not available")
            return {
                'note_id': note_id,
                'sources': [],
                'additional_info': '',
                'research_available': False
            }
        
        # Research additional information
        research_data = await perplexity_service.research_note_topic(
            note_title=note_title,
            note_content=note_content,
            key_phrases=key_phrases
        )
        
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'completed', 'total': 100})
        
        return {
            'note_id': note_id,
            'sources': research_data.get('sources', []),
            'additional_info': research_data.get('additional_info', ''),
            'research_available': research_data.get('research_available', False)
        }
        
    except Exception as e:
        logger.error(f"Error researching note {note_id}: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True)
def find_related_notes_task(self, note_id: str, note_embedding: List[float], user_id: str):
    """Find and create relationships with other notes"""
    
    try:
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'finding_relationships', 'total': 100})
        
        # This would typically use vector similarity search
        # For now, return empty relationships
        relationships = []
        
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'completed', 'total': 100})
        
        return {
            'note_id': note_id,
            'relationships': relationships,
            'relationships_created': len(relationships)
        }
        
    except Exception as e:
        logger.error(f"Error finding relationships for note {note_id}: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
