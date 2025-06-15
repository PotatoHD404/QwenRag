from openai import AsyncOpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
from typing import List, Dict, Any, Optional


class ExtendedOpenaiClient:
    """Unified async client for LM Studio compatible server with reranking support"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int = 300):
        self.base_url = base_url.rstrip('/') if base_url else None
        self.api_key = api_key
        self.timeout = timeout

        # Standard AsyncOpenAI client for chat, completions, embeddings
        self.openai_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout
        )

        # Headers for custom endpoints
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    # Delegate standard OpenAI methods
    @property
    def chat(self):
        """Access to chat completions"""
        return self.openai_client.chat

    @property
    def completions(self):
        """Access to completions"""
        return self.openai_client.completions

    @property
    def embeddings(self):
        """Access to embeddings"""
        return self.openai_client.embeddings

    @property
    def models(self):
        """Access to models"""
        return self.openai_client.models

    # Custom async reranking method
    async def rerank(self,
                     model: str,
                     query: str,
                     documents: List[str],
                     instruction: Optional[str] = None,
                     top_k: Optional[int] = None,
                     return_documents: bool = True) -> Dict[str, Any]:
        """
        Rerank documents by relevance to query using judge model

        Args:
            model: Name of the reranker model
            query: Search query
            documents: List of documents to rank
            instruction: Custom instruction (default: "Given a web search query, retrieve relevant passages that answer the query")
            top_k: Number of top results to return (None for all)
            return_documents: Whether to include document text in response

        Returns:
            Reranking response with sorted results
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/rerank",
                headers=self.headers,
                json={
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "instruction": instruction,
                    "top_k": top_k,
                    "return_documents": return_documents
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Reranking failed: {response.status} - {error_text}")

                return await response.json()

    async def get_health(self) -> Dict[str, Any]:
        """Get server health status"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()

    async def close(self):
        """Close the underlying OpenAI client"""
        await self.openai_client.close()


# Create a default client instance for backward compatibility
def create_client(base_url: str = "http://localhost:8001/v1", api_key: str = "sk-bebrus", timeout: int = 300) -> ExtendedOpenaiClient:
    """Create an ExtendedOpenaiClient instance"""
    return ExtendedOpenaiClient(base_url=base_url, api_key=api_key, timeout=timeout) 