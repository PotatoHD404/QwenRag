"""Database service for managing LanceDB operations."""

import os
import asyncio
from typing import List, Dict, Any, Optional, NamedTuple
from pathlib import Path
import json

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

from .config import CodeRAGConfig
from .tree_sitter_utils import ChunkWithLocation


class CodeChunk(LanceModel):
    """LanceDB model for storing code chunks with metadata."""
    
    id: str = Field(description="Unique identifier for the chunk")
    content: str = Field(description="The actual code content")
    file_path: str = Field(description="Path to the source file")
    start_line: int = Field(description="Starting line number in the file")
    end_line: int = Field(description="Ending line number in the file")
    start_char: int = Field(description="Starting character position in the file")
    end_char: int = Field(description="Ending character position in the file")
    file_extension: str = Field(description="File extension")
    repository_path: str = Field(description="Path to the repository root")
    chunk_type: str = Field(description="Type of code chunk (function, class, etc.)")
    embedding: Vector(2560) = Field(description="Vector embedding of the content")  # Qwen3-Embedding-4B has 2560 dimensions
    
    def get_location_string(self) -> str:
        """Get a human-readable location string."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"
    
    def get_context_lines(self, file_content: str, context_lines: int = 3) -> str:
        """Get the chunk content with additional context lines."""
        lines = file_content.split('\n')
        start_idx = max(0, self.start_line - context_lines)
        end_idx = min(len(lines), self.end_line + context_lines + 1)
        
        context_content = '\n'.join(lines[start_idx:end_idx])
        return f"```{self.file_extension[1:] if self.file_extension.startswith('.') else self.file_extension}\n{context_content}\n```"


class SearchResult(NamedTuple):
    """Search result with relevance information."""
    chunk: CodeChunk
    score: float
    rank: int


class DatabaseManager:
    """Manager for LanceDB operations."""
    
    def __init__(self, config: CodeRAGConfig):
        self.config = config
        self.db_path = config.database.path
        self.table_name = config.database.table_name
        self.db = None
        self.table = None
        
    async def initialize(self):
        """Initialize the database connection."""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Connect to LanceDB
            self.db = lancedb.connect(self.db_path)
            
            # Check if table exists, create if not
            if self.table_name in self.db.table_names():
                self.table = self.db.open_table(self.table_name)
                print(f"Opened existing table: {self.table_name}")
            else:
                # Create empty table with schema
                self.table = self.db.create_table(self.table_name, schema=CodeChunk)
                print(f"Created new table: {self.table_name}")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    async def add_chunks(self, chunks: List[ChunkWithLocation], embeddings: List[List[float]], repository_path: str):
        """Add code chunks with their embeddings to the database."""
        if not chunks or not embeddings:
            return
            
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Convert chunks to CodeChunk models
            code_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                file_ext = Path(chunk.file_path).suffix
                
                # Determine chunk type based on content
                chunk_type = self._determine_chunk_type(chunk.content)
                
                # Create unique ID
                chunk_id = f"{repository_path}:{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{i}"
                
                code_chunk = CodeChunk(
                    id=chunk_id,
                    content=chunk.content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    file_extension=file_ext,
                    repository_path=repository_path,
                    chunk_type=chunk_type,
                    embedding=embedding
                )
                code_chunks.append(code_chunk)
            
            # Add to database
            self.table.add(code_chunks)
            print(f"Added {len(code_chunks)} chunks to database")
            
        except Exception as e:
            print(f"Error adding chunks to database: {e}")
            raise
    
    def _determine_chunk_type(self, content: str) -> str:
        """Determine the type of code chunk based on its content."""
        content_lower = content.lower().strip()
        
        # Function patterns
        if any(pattern in content_lower for pattern in ['def ', 'function ', 'func ', 'fn ']):
            return "function"
        
        # Class patterns
        if any(pattern in content_lower for pattern in ['class ', 'struct ', 'interface ']):
            return "class"
        
        # Method patterns (inside classes)
        if any(pattern in content_lower for pattern in ['    def ', '  def ', '\tdef ']):
            return "method"
        
        # Import patterns
        if any(pattern in content_lower for pattern in ['import ', 'from ', '#include', 'using ']):
            return "import"
        
        # Comment patterns
        if content_lower.startswith('#') or content_lower.startswith('//') or content_lower.startswith('/*'):
            return "comment"
        
        return "code"
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 20) -> List[CodeChunk]:
        """Search for similar code chunks using vector similarity."""
        try:
            if not self.table:
                await self.initialize()
            
            # Perform vector search
            results = (
                self.table.search(query_embedding)
                .limit(top_k)
                .to_pydantic(CodeChunk)
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching database: {e}")
            return []
    
    async def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks for a specific file."""
        try:
            if not self.table:
                await self.initialize()
            
            results = (
                self.table.search()
                .where(f"file_path = '{file_path}'")
                .to_pydantic(CodeChunk)
            )
            
            return results
            
        except Exception as e:
            print(f"Error retrieving chunks for file {file_path}: {e}")
            return []
    
    async def get_chunks_by_repository(self, repository_path: str) -> List[CodeChunk]:
        """Get all chunks for a specific repository."""
        try:
            if not self.table:
                await self.initialize()
            
            results = (
                self.table.search()
                .where(f"repository_path = '{repository_path}'")
                .to_pydantic(CodeChunk)
            )
            
            return results
            
        except Exception as e:
            print(f"Error retrieving chunks for repository {repository_path}: {e}")
            return []
    
    async def delete_repository(self, repository_path: str):
        """Delete all chunks for a specific repository."""
        try:
            if not self.table:
                await self.initialize()
            
            # Delete rows matching the repository path
            self.table.delete(f"repository_path = '{repository_path}'")
            print(f"Deleted chunks for repository: {repository_path}")
            
        except Exception as e:
            print(f"Error deleting repository {repository_path}: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            if not self.table:
                await self.initialize()
            
            # Get basic stats
            total_chunks = self.table.count_rows()
            
            # Get repository counts
            all_chunks = self.table.search().to_pydantic(CodeChunk)
            repo_counts = {}
            file_type_counts = {}
            chunk_type_counts = {}
            
            for chunk in all_chunks:
                # Repository counts
                repo_counts[chunk.repository_path] = repo_counts.get(chunk.repository_path, 0) + 1
                
                # File type counts
                file_type_counts[chunk.file_extension] = file_type_counts.get(chunk.file_extension, 0) + 1
                
                # Chunk type counts
                chunk_type_counts[chunk.chunk_type] = chunk_type_counts.get(chunk.chunk_type, 0) + 1
            
            return {
                "total_chunks": total_chunks,
                "repositories": len(repo_counts),
                "repository_counts": repo_counts,
                "file_type_counts": file_type_counts,
                "chunk_type_counts": chunk_type_counts
            }
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close database connections."""
        # LanceDB connections don't need explicit closing
        pass 