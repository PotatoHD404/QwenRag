"""Repository indexer for processing and indexing code files."""

import os
import asyncio
from typing import List, Set, Optional
from pathlib import Path
import fnmatch
from gitignore_parser import parse_gitignore

from tqdm import tqdm

from .config import CodeRAGConfig
from .tree_sitter_utils import CodeChunker, ChunkWithLocation
from .embeddings import EmbeddingService
from .database import DatabaseManager


class RepositoryIndexer:
    """Indexes repository files using tree-sitter parsing and embeddings."""
    
    def __init__(self, config: CodeRAGConfig):
        self.config = config
        self.chunker = CodeChunker(max_chunk_tokens=config.chunking.max_tokens)
        self.embedding_service = EmbeddingService(config)
        self.db_manager = DatabaseManager(config)
        
        # Default ignore patterns
        self.default_ignore_patterns = [
            "*.pyc", "*.pyo", "*.pyd", "__pycache__", ".git", ".svn", ".hg",
            "node_modules", "*.min.js", "*.bundle.js", "dist", "build",
            ".DS_Store", "Thumbs.db", "*.log", "*.tmp", "*.temp",
            ".vscode", ".idea", "*.orig", "*.rej", "*.swp", "*.swo",
            ".pytest_cache", ".coverage", "htmlcov", ".tox", ".mypy_cache",
            "*.egg-info", ".eggs", "site-packages"
        ]
    
    async def index_repository(self, repo_path: str, force_reindex: bool = False) -> dict:
        """Index an entire repository."""
        repo_path = os.path.abspath(repo_path)
        
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        print(f"Indexing repository: {repo_path}")
        
        # Initialize database
        await self.db_manager.initialize()
        
        # Check if repository is already indexed
        if not force_reindex:
            existing_chunks = await self.db_manager.get_chunks_by_repository(repo_path)
            if existing_chunks:
                print(f"Repository already indexed with {len(existing_chunks)} chunks. Use --force to reindex.")
                return {"status": "already_indexed", "chunks": len(existing_chunks)}
        else:
            # Delete existing chunks for this repository
            await self.db_manager.delete_repository(repo_path)
            print("Deleted existing index for repository")
        
        # Find all supported files
        files_to_process = self._find_files_to_process(repo_path)
        print(f"Found {len(files_to_process)} files to process")
        
        if not files_to_process:
            print("No supported files found in repository")
            return {"status": "no_files", "chunks": 0}
        
        # Process files in batches
        total_chunks = 0
        batch_size = 10  # Process files in batches to manage memory
        
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(files_to_process) + batch_size - 1)//batch_size}")
            
            batch_chunks = await self._process_file_batch(batch_files, repo_path)
            total_chunks += len(batch_chunks)
            
            if batch_chunks:
                # Generate embeddings for batch
                print(f"Generating embeddings for {len(batch_chunks)} chunks...")
                contents = [chunk.content for chunk in batch_chunks]
                embeddings = await self.embedding_service.embed_texts(contents)
                
                # Store in database
                await self.db_manager.add_chunks(batch_chunks, embeddings, repo_path)
        
        print(f"Successfully indexed repository with {total_chunks} chunks")
        return {"status": "success", "chunks": total_chunks}
    
    async def _process_file_batch(self, file_paths: List[str], repo_path: str) -> List[ChunkWithLocation]:
        """Process a batch of files and extract chunks."""
        all_chunks = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                chunks = await self._process_single_file(file_path, repo_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return all_chunks
    
    async def _process_single_file(self, file_path: str, repo_path: str) -> List[ChunkWithLocation]:
        """Process a single file and extract chunks."""
        try:
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.config.chunking.max_file_size_mb:
                print(f"Skipping {file_path}: file too large ({file_size_mb:.1f} MB)")
                return []
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception:
                    print(f"Could not read {file_path}: encoding issues")
                    return []
            
            # Skip empty files
            if not content.strip():
                return []
            
            # Convert absolute path to relative path from repo root
            relative_path = os.path.relpath(file_path, repo_path)
            
            # Use tree-sitter to chunk the file
            chunks = await self.chunker.chunk_file(relative_path, content)
            
            return chunks
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def _find_files_to_process(self, repo_path: str) -> List[str]:
        """Find all files in the repository that should be processed."""
        files_to_process = []
        
        # Parse .gitignore if it exists
        gitignore_path = os.path.join(repo_path, '.gitignore')
        gitignore_filter = None
        if os.path.exists(gitignore_path):
            gitignore_filter = parse_gitignore(gitignore_path)
        
        # Walk through directory
        for root, dirs, files in os.walk(repo_path):
            # Filter directories
            dirs[:] = [d for d in dirs if not self._should_ignore_dir(d, root)]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Check if file should be ignored
                if self._should_ignore_file(file, file_path, relative_path, gitignore_filter):
                    continue
                
                # Check if file extension is supported
                file_ext = Path(file).suffix.lower()
                if file_ext in self.config.chunking.supported_extensions:
                    files_to_process.append(file_path)
        
        return files_to_process
    
    def _should_ignore_dir(self, dir_name: str, dir_path: str) -> bool:
        """Check if a directory should be ignored."""
        ignore_dirs = {
            '.git', '.svn', '.hg', '.bzr',
            'node_modules', '__pycache__', '.pytest_cache',
            '.mypy_cache', '.tox', '.venv', 'venv',
            'env', '.env', 'site-packages', '.eggs',
            'dist', 'build', 'target', 'out', 'bin',
            '.idea', '.vscode', '.vs'
        }
        
        return dir_name in ignore_dirs
    
    def _should_ignore_file(self, filename: str, file_path: str, relative_path: str, 
                           gitignore_filter) -> bool:
        """Check if a file should be ignored."""
        # Check gitignore
        if gitignore_filter and gitignore_filter(relative_path):
            return True
        
        # Check default ignore patterns
        for pattern in self.default_ignore_patterns:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(relative_path, pattern):
                return True
        
        # Ignore hidden files (starting with .)
        if filename.startswith('.') and filename not in ['.env', '.gitignore', '.dockerignore']:
            return True
        
        # Ignore backup files
        if filename.endswith(('~', '.bak', '.backup', '.orig', '.rej')):
            return True
        
        # Ignore compiled files
        if filename.endswith(('.pyc', '.pyo', '.pyd', '.class', '.jar', '.war')):
            return True
        
        # Ignore binary files (basic check)
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:  # Contains null bytes, likely binary
                    return True
        except Exception:
            return True  # Can't read file, ignore it
            
        return False
    
    async def index_single_file(self, file_path: str, repo_path: Optional[str] = None) -> dict:
        """Index a single file."""
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
            
        if repo_path is None:
            repo_path = os.path.dirname(file_path)
        else:
            repo_path = os.path.abspath(repo_path)
        
        print(f"Indexing file: {file_path}")
        
        # Initialize database
        await self.db_service.initialize()
        
        # Process the file
        chunks = await self._process_single_file(file_path, repo_path)
        
        if not chunks:
            print("No chunks extracted from file")
            return {"status": "no_chunks", "chunks": 0}
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        contents = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_service.embed_texts(contents)
        
        # Store in database
        await self.db_service.add_chunks(chunks, embeddings, repo_path)
        
        print(f"Successfully indexed file with {len(chunks)} chunks")
        return {"status": "success", "chunks": len(chunks)} 