"""Embedding service using OpenAI-compatible API."""

import asyncio
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI

from .config import CodeRAGConfig


class EmbeddingService:
    """Service for generating embeddings using OpenAI-compatible API."""
    
    def __init__(self, config: CodeRAGConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.api.base_url,
            api_key=config.api.api_key,
            timeout=config.api.timeout
        )
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
            
        try:
            # Process in batches to avoid hitting rate limits
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to be respectful to the API
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            return all_embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        # Prepare texts with instruction for retrieval
        formatted_texts = []
        for text in texts:
            # Add instruction as recommended by Qwen3-Embedding documentation
            task_description = "Retrieve relevant code snippets for the given query"
            formatted_text = f"Instruct: {task_description}\nQuery: {text}"
            formatted_texts.append(formatted_text)
        
        response = await self.client.embeddings.create(
            model=self.config.api.embedding_model,
            input=formatted_texts,
            encoding_format="float"
        )
        
        return [data.embedding for data in response.data]


class RerankingService:
    """Service for reranking search results using OpenAI-compatible API."""
    
    def __init__(self, config: CodeRAGConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.api.base_url,
            api_key=config.api.api_key,
            timeout=config.api.timeout
        )
    
    async def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[int]:
        """
        Rerank documents based on relevance to query.
        Returns indices of documents sorted by relevance score.
        """
        if not documents:
            return []
            
        try:
            # Create reranking prompts
            scores = []
            
            # Process documents in batches for efficiency
            batch_size = 5
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_indices = list(range(i, min(i + batch_size, len(documents))))
                
                batch_scores = await self._rerank_batch(query, batch_docs, batch_indices)
                scores.extend(batch_scores)
                
                # Small delay between batches
                if i + batch_size < len(documents):
                    await asyncio.sleep(0.1)
            
            # Sort by score (descending) and return top_k indices
            scores.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, score in scores[:top_k]]
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            # Fallback to original order
            return list(range(min(top_k, len(documents))))
    
    async def _rerank_batch(self, query: str, documents: List[str], indices: List[int]) -> List[tuple]:
        """Rerank a batch of documents."""
        results = []
        
        for i, doc in enumerate(documents):
            try:
                # Format the reranking prompt according to Qwen3-Reranker format
                task = "Given a web search query, retrieve relevant passages that answer the query"
                prompt = self._format_reranking_prompt(task, query, doc)
                
                response = await self.client.chat.completions.create(
                    model=self.config.api.reranking_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=5
                )
                
                # Extract relevance score from logprobs
                score = self._extract_relevance_score(response)
                results.append((indices[i], score))
                
            except Exception as e:
                print(f"Error reranking document {indices[i]}: {e}")
                # Assign neutral score on error
                results.append((indices[i], 0.5))
        
        return results
    
    def _format_reranking_prompt(self, task: str, query: str, document: str) -> str:
        """Format the reranking prompt according to Qwen3-Reranker specifications."""
        return f"<Instruct>: {task}\n\n<Query>: {query}\n\n<Document>: {document[:32000]}"  # Truncate long docs
    
    def _extract_relevance_score(self, response) -> float:
        """Extract relevance score from model response logprobs."""
        try:
            if not response.choices or not response.choices[0].logprobs:
                return 0.5
                
            logprobs = response.choices[0].logprobs.content[0].top_logprobs
            
            # Look for "yes" and "no" tokens
            yes_score = 0.0
            no_score = 0.0
            
            for logprob in logprobs:
                token = logprob.token.lower().strip()
                if token in ["yes", "y"]:
                    yes_score = max(yes_score, logprob.logprob)
                elif token in ["no", "n"]:
                    no_score = max(no_score, logprob.logprob)
            
            # Convert logprobs to probabilities and normalize
            import math
            yes_prob = math.exp(yes_score) if yes_score > -100 else 0
            no_prob = math.exp(no_score) if no_score > -100 else 0
            
            total_prob = yes_prob + no_prob
            if total_prob > 0:
                return yes_prob / total_prob
            else:
                return 0.5
                
        except Exception as e:
            print(f"Error extracting relevance score: {e}")
            return 0.5 