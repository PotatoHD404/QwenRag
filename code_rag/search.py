"""Search service for querying indexed code."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import CodeRAGConfig
from .embeddings import EmbeddingService, RerankingService
from .database import DatabaseManager, CodeChunk, SearchResult


@dataclass
class QueryResult:
    """Result of a search query."""
    query: str
    results: List[SearchResult]
    total_chunks_searched: int
    reranked: bool
    execution_time_ms: float


class SearchService:
    """Service for searching indexed code repositories."""
    
    def __init__(self, config: CodeRAGConfig):
        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.reranking_service = RerankingService(config) if config.search.use_reranking else None
        self.db_manager = DatabaseManager(config)
    
    async def search(self, query: str, top_k: Optional[int] = None, 
                    use_reranking: Optional[bool] = None, 
                    repository_filter: Optional[str] = None) -> QueryResult:
        """
        Search for code chunks relevant to the query.
        
        Args:
            query: The search query
            top_k: Number of final results to return
            use_reranking: Whether to use reranking (overrides config)
            repository_filter: Filter results to specific repository
            
        Returns:
            QueryResult with search results and metadata
        """
        import time
        start_time = time.time()
        
        # Use provided parameters or fall back to config
        if top_k is None:
            top_k = self.config.search.top_k_final
        if use_reranking is None:
            use_reranking = self.config.search.use_reranking and self.reranking_service is not None
        
        try:
            # Initialize database
            await self.db_manager.initialize()
            
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Perform vector search
            initial_k = self.config.search.top_k_initial if use_reranking else top_k
            candidate_chunks = await self.db_manager.search_similar(query_embedding, initial_k)
            
            # Apply repository filter if specified
            if repository_filter:
                candidate_chunks = [
                    chunk for chunk in candidate_chunks 
                    if chunk.repository_path == repository_filter
                ]
            
            # If no reranking, return vector search results
            if not use_reranking or not candidate_chunks:
                results = [
                    SearchResult(chunk=chunk, score=1.0 - (i * 0.01), rank=i)
                    for i, chunk in enumerate(candidate_chunks[:top_k])
                ]
                
                execution_time = (time.time() - start_time) * 1000
                return QueryResult(
                    query=query,
                    results=results,
                    total_chunks_searched=len(candidate_chunks),
                    reranked=False,
                    execution_time_ms=execution_time
                )
            
            # Perform reranking
            documents = [chunk.content for chunk in candidate_chunks]
            reranked_indices = await self.reranking_service.rerank(query, documents, top_k)
            
            # Create final results with reranking scores
            results = []
            for rank, idx in enumerate(reranked_indices):
                if idx < len(candidate_chunks):
                    chunk = candidate_chunks[idx]
                    # Score decreases with rank (higher rank = lower score)
                    score = 1.0 - (rank * 0.1)
                    results.append(SearchResult(chunk=chunk, score=score, rank=rank))
            
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                query=query,
                results=results,
                total_chunks_searched=len(candidate_chunks),
                reranked=True,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            print(f"Error during search: {e}")
            execution_time = (time.time() - start_time) * 1000
            return QueryResult(
                query=query,
                results=[],
                total_chunks_searched=0,
                reranked=False,
                execution_time_ms=execution_time
            )
    
    async def search_by_file_type(self, query: str, file_extension: str, 
                                 top_k: Optional[int] = None) -> QueryResult:
        """Search for code chunks in specific file types."""
        # This would require extending the database search capabilities
        # For now, we'll do post-filtering
        result = await self.search(query, top_k=top_k * 2 if top_k else 20)
        
        # Filter by file extension
        filtered_results = [
            res for res in result.results 
            if res.chunk.file_extension == file_extension
        ]
        
        # Take only top_k results
        if top_k:
            filtered_results = filtered_results[:top_k]
        
        # Update ranks
        for i, res in enumerate(filtered_results):
            res = SearchResult(chunk=res.chunk, score=res.score, rank=i)
        
        return QueryResult(
            query=result.query,
            results=filtered_results,
            total_chunks_searched=len([r for r in result.results if r.chunk.file_extension == file_extension]),
            reranked=result.reranked,
            execution_time_ms=result.execution_time_ms
        )
    
    async def search_by_chunk_type(self, query: str, chunk_type: str, 
                                  top_k: Optional[int] = None) -> QueryResult:
        """Search for specific types of code chunks (functions, classes, etc.)."""
        # Similar to file type search - post-filtering approach
        result = await self.search(query, top_k=top_k * 2 if top_k else 20)
        
        # Filter by chunk type
        filtered_results = [
            res for res in result.results 
            if res.chunk.chunk_type == chunk_type
        ]
        
        # Take only top_k results
        if top_k:
            filtered_results = filtered_results[:top_k]
        
        # Update ranks
        for i, res in enumerate(filtered_results):
            res = SearchResult(chunk=res.chunk, score=res.score, rank=i)
        
        return QueryResult(
            query=result.query,
            results=filtered_results,
            total_chunks_searched=len([r for r in result.results if r.chunk.chunk_type == chunk_type]),
            reranked=result.reranked,
            execution_time_ms=result.execution_time_ms
        )
    
    async def get_similar_to_chunk(self, chunk_id: str, top_k: int = 5) -> List[SearchResult]:
        """Find chunks similar to a given chunk."""
        try:
            await self.db_manager.initialize()
            
            # Get the reference chunk
            all_chunks = await self.db_manager.search_similar([0.0] * 2560, 10000)  # Get all chunks
            reference_chunk = None
            for chunk in all_chunks:
                if chunk.id == chunk_id:
                    reference_chunk = chunk
                    break
            
            if not reference_chunk:
                return []
            
            # Use the chunk's embedding to find similar chunks
            similar_chunks = await self.db_manager.search_similar(reference_chunk.embedding, top_k + 1)
            
            # Remove the reference chunk itself
            similar_chunks = [chunk for chunk in similar_chunks if chunk.id != chunk_id][:top_k]
            
            results = [
                SearchResult(chunk=chunk, score=1.0 - (i * 0.1), rank=i)
                for i, chunk in enumerate(similar_chunks)
            ]
            
            return results
            
        except Exception as e:
            print(f"Error finding similar chunks: {e}")
            return []
    
    def format_search_results(self, query_result: QueryResult, 
                            show_content: bool = True, 
                            max_content_length: int = 500) -> str:
        """Format search results for display."""
        if not query_result.results:
            return f"No results found for query: '{query_result.query}'"
        
        lines = []
        lines.append(f"Search Results for: '{query_result.query}'")
        lines.append(f"Found {len(query_result.results)} results in {query_result.execution_time_ms:.1f}ms")
        lines.append(f"Searched {query_result.total_chunks_searched} chunks")
        lines.append(f"Reranked: {'Yes' if query_result.reranked else 'No'}")
        lines.append("-" * 80)
        
        for i, result in enumerate(query_result.results, 1):
            chunk = result.chunk
            lines.append(f"\n{i}. {chunk.get_location_string()} (Score: {result.score:.3f})")
            lines.append(f"   Type: {chunk.chunk_type} | File: {chunk.file_extension}")
            lines.append(f"   Repository: {chunk.repository_path}")
            
            if show_content:
                content = chunk.content
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                
                lines.append(f"   Content:")
                for line in content.split('\n'):
                    lines.append(f"   | {line}")
                lines.append("")
        
        return '\n'.join(lines)
    
    async def close(self):
        """Close any open connections."""
        await self.db_manager.close() 