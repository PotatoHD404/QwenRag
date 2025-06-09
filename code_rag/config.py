"""Configuration management for Code RAG system."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """API configuration for OpenAI-compatible endpoints."""
    base_url: str = Field(default="http://localhost:1234/v1", description="API base URL")
    api_key: str = Field(default="dummy", description="API key")
    
    # Model configurations with larger context windows
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-4B", description="Embedding model name")
    embedding_max_tokens: int = Field(default=8192, description="Max tokens for embedding model")  # 8k context
    
    reranking_model: str = Field(default="Qwen/Qwen3-Reranker-4B", description="Reranking model name")
    reranking_max_tokens: int = Field(default=32768, description="Max tokens for reranking model")  # 32k context
    
    # Timeouts and retries
    timeout: int = Field(default=300, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class DatabaseConfig(BaseModel):
    """Database configuration for LanceDB."""
    path: str = Field(default="./rag_db", description="Database path")
    table_name: str = Field(default="code_chunks", description="Table name for code chunks")
    
    # Vector search configuration
    search_limit: int = Field(default=20, description="Initial search limit")
    nprobes: int = Field(default=1, description="Number of probes for vector search")


class ChunkingConfig(BaseModel):
    """Configuration for code chunking."""
    max_tokens: int = Field(default=1000, description="Maximum tokens per chunk")
    overlap_tokens: int = Field(default=100, description="Overlap between chunks in tokens")
    
    # Tree-sitter specific settings
    prefer_functions: bool = Field(default=True, description="Prefer function-level chunking")
    include_comments: bool = Field(default=True, description="Include comments in chunks")
    collapse_large_functions: bool = Field(default=True, description="Collapse large functions to signatures")
    
    # File processing settings
    max_file_size_mb: int = Field(default=10, description="Maximum file size to process in MB")
    supported_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".h", ".hpp",
            ".rs", ".go", ".cs", ".php", ".rb", ".swift", ".kt", ".scala", ".sh",
            ".sql", ".md", ".txt", ".yaml", ".yml", ".json", ".xml", ".html", ".css"
        ],
        description="Supported file extensions"
    )


class SearchConfig(BaseModel):
    """Configuration for search functionality."""
    use_reranking: bool = Field(default=True, description="Enable reranking")
    top_k_initial: int = Field(default=20, description="Initial number of results to retrieve")
    top_k_final: int = Field(default=5, description="Final number of results after reranking")
    similarity_threshold: float = Field(default=0.1, description="Minimum similarity threshold")


class CodeRAGConfig(BaseModel):
    """Main configuration for Code RAG system."""
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    
    @classmethod
    def from_env(cls) -> "CodeRAGConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # API configuration from environment
        if os.getenv("RAG_API_BASE"):
            config.api.base_url = os.getenv("RAG_API_BASE")
        if os.getenv("RAG_API_KEY"):
            config.api.api_key = os.getenv("RAG_API_KEY")
        if os.getenv("RAG_EMBEDDING_MODEL"):
            config.api.embedding_model = os.getenv("RAG_EMBEDDING_MODEL")
        if os.getenv("RAG_RERANKING_MODEL"):
            config.api.reranking_model = os.getenv("RAG_RERANKING_MODEL")
        
        # Context window sizes from environment
        if os.getenv("RAG_EMBEDDING_MAX_TOKENS"):
            config.api.embedding_max_tokens = int(os.getenv("RAG_EMBEDDING_MAX_TOKENS"))
        if os.getenv("RAG_RERANKING_MAX_TOKENS"):
            config.api.reranking_max_tokens = int(os.getenv("RAG_RERANKING_MAX_TOKENS"))
        
        # Database configuration
        if os.getenv("RAG_DB_PATH"):
            config.database.path = os.getenv("RAG_DB_PATH")
        
        # Chunking configuration
        if os.getenv("RAG_CHUNK_SIZE"):
            config.chunking.max_tokens = int(os.getenv("RAG_CHUNK_SIZE"))
        
        # Search configuration
        if os.getenv("RAG_DISABLE_RERANKING"):
            config.search.use_reranking = os.getenv("RAG_DISABLE_RERANKING").lower() not in ("true", "1", "yes")
        if os.getenv("RAG_TOP_K"):
            config.search.top_k_final = int(os.getenv("RAG_TOP_K"))
        
        return config
    
    @classmethod
    def from_file(cls, config_path: Path) -> "CodeRAGConfig":
        """Load configuration from YAML file."""
        import yaml
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_file(self, config_path: Path):
        """Save configuration to YAML file."""
        import yaml
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


def load_config(config_file: Optional[Path] = None) -> CodeRAGConfig:
    """Load configuration from file or environment variables."""
    if config_file and config_file.exists():
        return CodeRAGConfig.from_file(config_file)
    else:
        return CodeRAGConfig.from_env() 