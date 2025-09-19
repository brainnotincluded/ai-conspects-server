"""
Celery worker for audio processing tasks
"""

import whisper
import boto3
from io import BytesIO
from typing import Dict, Any
import logging
from workers.celery_app import celery_app
from config import settings

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_audio_task(self, recording_id: str, s3_key: str, user_id: str, settings_dict: Dict[str, Any]):
    """Process audio recording: transcription and analysis"""
    
    try:
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'loading_model', 'total': 100})
        
        # Load Whisper model
        model = whisper.load_model(settings.WHISPER_MODEL)
        
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'downloading_audio', 'total': 100})
        
        # Download audio from S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION
        )
        
        audio_obj = s3_client.get_object(Bucket=settings.S3_BUCKET, Key=s3_key)
        audio_data = audio_obj['Body'].read()
        
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'transcribing', 'total': 100})
        
        # Transcription
        result = model.transcribe(BytesIO(audio_data), language=settings_dict.get('language', 'ru'))
        
        # Clean and format text
        transcript = clean_transcript(result['text'])
        key_phrases = extract_key_phrases(transcript)
        
        # Update task status
        self.update_state(state='PROGRESS', meta={'current': 'completed', 'total': 100})
        
        return {
            'recording_id': recording_id,
            'transcript': transcript,
            'key_phrases': key_phrases,
            'confidence': result.get('confidence', 0.9),
            'language': result.get('language', 'ru')
        }
        
    except Exception as e:
        logger.error(f"Error processing audio {recording_id}: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

def clean_transcript(text: str) -> str:
    """Clean and format transcript"""
    # Remove extra spaces and characters
    text = ' '.join(text.split())
    
    # Remove repetitive phrases
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

def extract_key_phrases(text: str) -> list:
    """Extract key phrases from text"""
    # Simple implementation - can be improved with NLP libraries
    words = text.split()
    # Filter stop words and extract meaningful phrases
    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 'о', 'об', 'за', 'при', 'через', 'над', 'под', 'перед', 'после', 'между', 'среди', 'вокруг', 'около', 'близ', 'далеко', 'здесь', 'там', 'где', 'когда', 'как', 'что', 'кто', 'который', 'это', 'то', 'такой', 'такая', 'такое', 'такие', 'мой', 'моя', 'мое', 'мои', 'твой', 'твоя', 'твое', 'твои', 'его', 'ее', 'их', 'наш', 'наша', 'наше', 'наши', 'ваш', 'ваша', 'ваше', 'ваши'}
    
    meaningful_words = [word for word in words if len(word) > 3 and word.lower() not in stop_words]
    return meaningful_words[:10]  # Top 10 key phrases
