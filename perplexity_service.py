"""
Perplexity API integration for research and information gathering
"""

import httpx
from typing import Dict, Any, List, Optional
from config import settings
import logging

logger = logging.getLogger(__name__)

class PerplexityService:
    """Perplexity API service for research and information gathering"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.PERPLEXITY_API_KEY
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_and_research(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for information using Perplexity API"""
        if not self.api_key:
            logger.warning("Perplexity API key not configured")
            return self._create_fallback_response(query)
        
        try:
            response = await self.client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a research assistant. Provide accurate, up-to-date information with sources. Respond in Russian when appropriate."
                        },
                        {
                            "role": "user",
                            "content": f"Research the following topic and provide detailed information with sources: {query}"
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "return_citations": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_perplexity_response(result, query)
            else:
                logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
                return self._create_fallback_response(query)
                
        except Exception as e:
            logger.error(f"Error in Perplexity service: {e}")
            return self._create_fallback_response(query)
    
    async def research_note_topic(self, note_title: str, note_content: str, key_phrases: List[str] = None) -> Dict[str, Any]:
        """Research additional information for a note topic"""
        if not self.api_key:
            return {"sources": [], "additional_info": "", "research_available": False}
        
        # Create research query
        query_parts = [note_title]
        if key_phrases:
            query_parts.extend(key_phrases[:3])  # Top 3 key phrases
        
        research_query = f"Additional information about: {' '.join(query_parts)}"
        
        try:
            response = await self.client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a research assistant. Find additional relevant information about the given topic. Provide sources and key insights. Respond in Russian."
                        },
                        {
                            "role": "user",
                            "content": f"Find additional information about: {research_query}\n\nOriginal content: {note_content[:500]}..."
                        }
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.3,
                    "return_citations": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_research_response(result, note_title)
            else:
                logger.error(f"Perplexity research error: {response.status_code} - {response.text}")
                return {"sources": [], "additional_info": "", "research_available": False}
                
        except Exception as e:
            logger.error(f"Error in Perplexity research: {e}")
            return {"sources": [], "additional_info": "", "research_available": False}
    
    def _parse_perplexity_response(self, response: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Parse Perplexity API response"""
        try:
            content = response["choices"][0]["message"]["content"]
            citations = response.get("citations", [])
            
            # Extract sources from citations
            sources = []
            for citation in citations:
                if isinstance(citation, dict):
                    sources.append({
                        "title": citation.get("title", "Unknown"),
                        "url": citation.get("url", ""),
                        "type": "web",
                        "relevance": 0.8
                    })
            
            return {
                "content": content,
                "sources": sources,
                "query": query,
                "research_available": True,
                "total_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error parsing Perplexity response: {e}")
            return self._create_fallback_response(query)
    
    def _parse_research_response(self, response: Dict[str, Any], note_title: str) -> Dict[str, Any]:
        """Parse research response for note augmentation"""
        try:
            content = response["choices"][0]["message"]["content"]
            citations = response.get("citations", [])
            
            # Extract sources
            sources = []
            for citation in citations:
                if isinstance(citation, dict):
                    sources.append({
                        "title": citation.get("title", "Unknown"),
                        "url": citation.get("url", ""),
                        "type": "web",
                        "relevance": 0.7
                    })
            
            return {
                "additional_info": content,
                "sources": sources,
                "research_available": True,
                "note_title": note_title
            }
            
        except Exception as e:
            logger.error(f"Error parsing research response: {e}")
            return {"sources": [], "additional_info": "", "research_available": False}
    
    def _create_fallback_response(self, query: str) -> Dict[str, Any]:
        """Create fallback response when API is not available"""
        return {
            "content": f"Research for '{query}' is not available at the moment.",
            "sources": [],
            "query": query,
            "research_available": False,
            "total_sources": 0
        }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Global service instance
perplexity_service = PerplexityService() if settings.PERPLEXITY_API_KEY else None
