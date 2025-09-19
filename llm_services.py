"""
LLM Services for AI Conspects Server
Supports Ollama and OpenRouter modes
"""

import httpx
import openai
from typing import Dict, Any, Optional, List
from config import settings
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """Base class for LLM services"""
    
    def __init__(self):
        self.client = None
    
    async def generate_notes(self, transcript: str, context: str = "", settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate notes from transcript"""
        raise NotImplementedError
    
    async def generate_chat_response(self, message: str, context: str = "", conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate chat response"""
        raise NotImplementedError

class OllamaService(LLMService):
    """Ollama LLM service"""
    
    def __init__(self, model: str = None, base_url: str = None):
        super().__init__()
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate_notes(self, transcript: str, context: str = "", settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate notes using Ollama"""
        try:
            prompt = self._create_note_generation_prompt(transcript, context, settings)
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": settings.get("temperature", 0.7) if settings else 0.7,
                        "num_predict": settings.get("max_tokens", 2000) if settings else 2000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_ollama_response(result.get("response", ""))
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._create_fallback_response(transcript)
                
        except Exception as e:
            logger.error(f"Error in Ollama service: {e}")
            return self._create_fallback_response(transcript)
    
    async def generate_chat_response(self, message: str, context: str = "", conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate chat response using Ollama"""
        try:
            prompt = self._create_chat_prompt(message, context, conversation_history)
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "message": result.get("response", ""),
                    "confidence": 0.9
                }
            else:
                logger.error(f"Ollama chat API error: {response.status_code} - {response.text}")
                return {"message": "Sorry, I couldn't process your request.", "confidence": 0.1}
                
        except Exception as e:
            logger.error(f"Error in Ollama chat service: {e}")
            return {"message": "Sorry, I encountered an error.", "confidence": 0.1}
    
    def _create_note_generation_prompt(self, transcript: str, context: str, settings: Dict[str, Any] = None) -> str:
        """Create prompt for note generation"""
        return f"""
Analyze the following transcript and create a structured note in Russian:

Transcript: {transcript}

Context: {context}

Create a JSON response with the following structure:
{{
    "title": "Brief and descriptive title in Russian",
    "summary": "2-3 sentence summary in Russian",
    "content": "Main content in Markdown format in Russian",
    "tags": ["tag1", "tag2", "tag3"],
    "key_points": ["point1", "point2", "point3"],
    "reading_time": 5
}}

Focus on:
- Main topics and concepts
- Key insights and conclusions
- Important details and examples
- Practical applications if mentioned
- Connections to other topics

Respond only with valid JSON, no additional text.
"""
    
    def _create_chat_prompt(self, message: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Create prompt for chat"""
        history_text = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = "User" if msg.get("is_from_user") else "Assistant"
                history_text += f"{role}: {msg.get('content', '')}\n"
        
        return f"""
You are a helpful AI assistant for working with notes and knowledge. Respond in Russian.

Context from user's notes:
{context}

Conversation history:
{history_text}

User message: {message}

Provide a helpful, specific response based on the context and conversation history.
"""
    
    def _parse_ollama_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Ollama response into structured data"""
        try:
            import json
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        return self._create_fallback_response(response_text)
    
    def _create_fallback_response(self, text: str) -> Dict[str, Any]:
        """Create fallback response when parsing fails"""
        return {
            "title": "Заметка из аудио",
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "content": text,
            "tags": ["аудио", "заметка"],
            "key_points": [],
            "reading_time": max(1, len(text.split()) // 200)
        }

class OpenRouterService(LLMService):
    """OpenRouter LLM service"""
    
    def __init__(self, api_key: str = None, model: str = None):
        super().__init__()
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model = model or "openai/gpt-4-turbo-preview"
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    async def generate_notes(self, transcript: str, context: str = "", settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate notes using OpenRouter"""
        try:
            prompt = self._create_note_generation_prompt(transcript, context, settings)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating structured notes from audio transcripts. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.get("temperature", 0.7) if settings else 0.7,
                max_tokens=settings.get("max_tokens", 2000) if settings else 2000
            )
            
            response_text = response.choices[0].message.content
            return self._parse_openrouter_response(response_text)
            
        except Exception as e:
            logger.error(f"Error in OpenRouter service: {e}")
            return self._create_fallback_response(transcript)
    
    async def generate_chat_response(self, message: str, context: str = "", conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate chat response using OpenRouter"""
        try:
            messages = [
                {"role": "system", "content": f"You are a helpful AI assistant for working with notes and knowledge. Respond in Russian.\n\nContext: {context}"}
            ]
            
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages
                    role = "user" if msg.get("is_from_user") else "assistant"
                    messages.append({"role": role, "content": msg.get("content", "")})
            
            messages.append({"role": "user", "content": message})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "message": response.choices[0].message.content,
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"Error in OpenRouter chat service: {e}")
            return {"message": "Sorry, I encountered an error.", "confidence": 0.1}
    
    def _create_note_generation_prompt(self, transcript: str, context: str, settings: Dict[str, Any] = None) -> str:
        """Create prompt for note generation"""
        return f"""
Analyze the following transcript and create a structured note in Russian:

Transcript: {transcript}

Context: {context}

Create a JSON response with the following structure:
{{
    "title": "Brief and descriptive title in Russian",
    "summary": "2-3 sentence summary in Russian",
    "content": "Main content in Markdown format in Russian",
    "tags": ["tag1", "tag2", "tag3"],
    "key_points": ["point1", "point2", "point3"],
    "reading_time": 5
}}

Focus on:
- Main topics and concepts
- Key insights and conclusions
- Important details and examples
- Practical applications if mentioned
- Connections to other topics

Respond only with valid JSON, no additional text.
"""
    
    def _parse_openrouter_response(self, response_text: str) -> Dict[str, Any]:
        """Parse OpenRouter response into structured data"""
        try:
            import json
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        return self._create_fallback_response(response_text)
    
    def _create_fallback_response(self, text: str) -> Dict[str, Any]:
        """Create fallback response when parsing fails"""
        return {
            "title": "Заметка из аудио",
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "content": text,
            "tags": ["аудио", "заметка"],
            "key_points": [],
            "reading_time": max(1, len(text.split()) // 200)
        }

def get_llm_service(mode: str, **kwargs) -> LLMService:
    """Factory function to get appropriate LLM service"""
    if mode.lower() == "ollama":
        return OllamaService(**kwargs)
    elif mode.lower() == "openrouter":
        return OpenRouterService(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM mode: {mode}")

# Global service instances
ollama_service = OllamaService()
openrouter_service = OpenRouterService() if settings.OPENROUTER_API_KEY else None
