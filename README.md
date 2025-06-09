# Qwen RAG - Repository Retrieval Augmented Generation

[![PyPI version](https://badge.fury.io/py/qwen-rag.svg)](https://badge.fury.io/py/qwen-rag)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful RAG system for querying code repositories using tree-sitter parsing, LanceDB vector storage, and Qwen models for embedding and reranking.

## 🚀 Features

- **🔍 Semantic Code Search**: Find code by meaning, not just keywords
- **🌐 Multi-Language Support**: Python, JavaScript, TypeScript, Java, C/C++, Rust, Go, C#, and more
- **🌳 Tree-sitter Parsing**: Intelligent code chunking preserving semantic structure  
- **⚡ Function-Level Indexing**: Automatically extracts and indexes functions, classes, and methods
- **🤖 Qwen Model Integration**: Uses Qwen3-Embedding-4B and Qwen3-Reranker-4B models
- **📍 Precise Location Tracking**: File paths, line numbers, and character positions
- **💾 Vector Database**: Powered by LanceDB for fast similarity search
- **🖥️ CLI Interface**: Easy-to-use command-line tool
- **⚙️ Configurable**: Flexible configuration via environment variables or files
- **📦 Multi-Repository Support**: Index and search across multiple code repositories

## 📋 Requirements

- **Python 3.9+**
- **4GB+ RAM** recommended
- **Qwen embedding and reranking models** accessible via OpenAI-compatible API
- **Tree-sitter language parsers** (installed automatically)

## 🛠️ Installation

### From PyPI (Recommended)

```bash
pip install qwen-rag
```

### From Source

```bash
git clone https://github.com/yourusername/QwenRag.git
cd QwenRag
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
qwen-rag --help
# or
python -m code_rag.cli --help
```

## 🤖 Model Setup

Qwen RAG works with any OpenAI-compatible API serving Qwen models. Here are the most popular options:

### Option 1: LM Studio (Recommended for Beginners)

1. **Download LM Studio**: [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Download Models**:
   - Search and download: `text-embedding-qwen3-embedding-4b`
   - Search and download: `qwen.qwen3-reranker-4b`
3. **Start Local Server**:
   - Load the embedding model
   - Go to "Local Server" tab
   - Start server on `http://localhost:1234`
4. **Configure Qwen RAG**: Use default settings (already configured for `localhost:1234`)

### Option 2: Ollama

```bash
# Install Ollama: https://ollama.ai/
ollama pull qwen:embedding    # For embeddings
ollama pull qwen:reranker     # For reranking

# Start Ollama server
ollama serve
```

### Option 3: vLLM or Other OpenAI-Compatible Servers

```bash
# Example with vLLM
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Embedding-4B \
  --port 1234
```

### Option 4: Remote API Services

Configure your API endpoint in the configuration file or environment variables.

## ⚙️ Configuration

### Quick Start (Using Defaults)

The system works out-of-the-box with LM Studio running on `localhost:1234`:

```bash
# Index your code repository
qwen-rag index /path/to/your/code

# Search your code
qwen-rag search "authentication function"
```

### Environment Variables Configuration

```bash
# API Configuration
export RAG_API_BASE="http://localhost:1234/v1"
export RAG_API_KEY="dummy"
export RAG_EMBEDDING_MODEL="text-embedding-qwen3-embedding-4b"
export RAG_RERANKING_MODEL="qwen.qwen3-reranker-4b"

# Optional: Context Window Sizes
export RAG_EMBEDDING_MAX_TOKENS="8192"
export RAG_RERANKING_MAX_TOKENS="32768"

# Optional: Database and Processing
export RAG_DB_PATH="./rag_db"
export RAG_CHUNK_SIZE="1000"
export RAG_DISABLE_RERANKING="false"
```

### Configuration File

Create `config.yaml`:

```yaml
# API Configuration
api:
  base_url: "http://localhost:1234/v1"
  api_key: "dummy"
  
  # Qwen Model Configuration
  embedding_model: "text-embedding-qwen3-embedding-4b"
  embedding_max_tokens: 8192  # 8k context window
  
  reranking_model: "qwen.qwen3-reranker-4b"
  reranking_max_tokens: 32768  # 32k context window
  
  # Request settings
  timeout: 300  # seconds
  max_retries: 3

# Database Configuration
database:
  path: "./rag_db"
  table_name: "code_chunks"

# Chunking Configuration
chunking:
  max_tokens: 1000  # Maximum tokens per chunk
  prefer_functions: true  # Prefer function-level chunking
  include_comments: true  # Include comments in chunks

# Search Configuration
search:
  use_reranking: true  # Enable reranking for better results
  top_k_initial: 20  # Initial number of results to retrieve
  top_k_final: 5  # Final number of results after reranking
```

## 🎯 Quick Start

### 1. Index a Repository

```bash
# Index current directory
qwen-rag index .

# Index specific repository
qwen-rag index /path/to/repo

# Index with custom chunk size
qwen-rag index . --chunk-size 500

# Force reindex existing repository
qwen-rag index . --force
```

### 2. Search Code

```bash
# Basic search with reranking
qwen-rag search "function that handles authentication"

# Fast search without reranking  
qwen-rag search "database connection" --no-reranking

# Limit results
qwen-rag search "error handling" --top-k 3

# Search only Python files
qwen-rag search "async function" --file-type .py

# Search only functions (when filtering is available)
qwen-rag search "validation logic" --chunk-type function
```

### 3. Interactive Mode

```bash
qwen-rag interactive
```

## 📖 Usage Examples

### Repository Management

```bash
# Index multiple repositories
qwen-rag index /path/to/frontend
qwen-rag index /path/to/backend  
qwen-rag index /path/to/scripts

# View database statistics
qwen-rag stats

# Show current configuration
qwen-rag config-show

# Delete repository from index
qwen-rag delete /path/to/repo
```

### Advanced Search Examples

```bash
# Find authentication code
qwen-rag search "user authentication login password"

# Look for error handling patterns
qwen-rag search "try catch exception handling error"

# Find database operations
qwen-rag search "database query insert update delete"

# Search for API endpoints
qwen-rag search "REST API endpoint route handler"

# Find specific algorithms
qwen-rag search "sorting algorithm implementation"

# Look for configuration management
qwen-rag search "config settings environment variables"
```

### Using Configuration Files

```bash
# Use custom config file
qwen-rag --config-file my-config.yaml index /path/to/repo

# Override settings via CLI
qwen-rag --api-base "http://localhost:8000" search "query"
```

## 🏗️ Architecture

### Components

1. **Tree-sitter Manager**: Handles parsing of 13+ programming languages
2. **Code Chunker**: Intelligently splits code into semantic chunks (functions, classes)
3. **Embedding Service**: Generates embeddings using Qwen3-Embedding-4B (2560 dimensions)
4. **Reranking Service**: Reranks results using Qwen3-Reranker-4B for better precision
5. **Database Manager**: Manages LanceDB operations and multi-repository support
6. **Search Service**: Orchestrates search and ranking across all repositories

### Data Flow

```
Repository → Tree-sitter → Semantic Chunks → Embeddings → LanceDB
     ↓
Query → Embedding → Vector Search → Reranking → Results
```

### Supported Languages

Tree-sitter parsing for: **Python**, **JavaScript**, **TypeScript**, **Java**, **C/C++**, **Rust**, **Go**, **C#**, **PHP**, **Ruby**, **Swift**, **Kotlin**, **Scala**

Fallback text processing for: **Shell**, **SQL**, **Markdown**, **YAML**, **JSON**, **HTML**, **CSS**, and more

## 🎨 Semantic Chunking

The system uses tree-sitter to create intelligent, semantically meaningful chunks:

### Function-Level Chunking
```python
def authenticate_user(username, password):
    """Authenticate user credentials."""
    # ... function body ...
```

### Class Overview
```python
class UserService:
    def __init__(self, database_url): ...
    def authenticate_user(self, username, password): ...
    def get_user_profile(self, user_id): ...
```

### Smart Collapsing
Large functions show signature + collapsed body for better overview.

## 🔧 Programmatic Usage

```python
import asyncio
from code_rag.config import load_config
from code_rag.indexer import RepositoryIndexer
from code_rag.search import SearchService

async def main():
    # Load configuration
    config = load_config()
    
    # Index repository
    indexer = RepositoryIndexer(config)
    await indexer.index_repository("./my_repo")
    
    # Search
    search_service = SearchService(config)
    results = await search_service.search("authentication function")
    
    for result in results.results:
        print(f"{result.chunk.file_path}:{result.chunk.start_line}")
        print(f"Score: {result.score}")
        print(result.chunk.content[:200])
        print("-" * 50)
    
    await search_service.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📊 Performance

### Typical Performance Metrics
- **Indexing**: ~1000 chunks/minute (depends on file complexity)
- **Embedding Search**: 100-500ms (without reranking)  
- **With Reranking**: 1-3 seconds (includes embedding + reranking)
- **Memory Usage**: ~100-500MB (scales with repository size)
- **Context Windows**: 8k tokens (embedding), 32k tokens (reranking)

### Optimization Tips

- Use `--no-reranking` for faster searches during development
- Reduce `--chunk-size` for memory efficiency
- Use file type filters (`--file-type .py`) to narrow search scope
- Index frequently used repositories locally

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tree-sitter](https://tree-sitter.github.io/) for excellent code parsing capabilities
- [LanceDB](https://lancedb.com/) for high-performance vector storage
- [Qwen Team](https://github.com/QwenLM) for powerful embedding and reranking models
- [OpenAI](https://openai.com/) for the API interface standard

## 🐛 Troubleshooting

### Common Issues

**Tree-sitter parsing errors**: Some language parsers may not initialize. The system automatically falls back to text chunking.

**API connection issues**: 
- Ensure your model server is running on the correct port
- Check that the model names match your server configuration
- Verify the API endpoint is accessible

**Memory issues**: 
- Reduce chunk size: `qwen-rag index . --chunk-size 500`
- Process smaller repositories or use file type filters
- Ensure you have sufficient RAM (4GB+ recommended)

**Slow performance**:
- Use `--no-reranking` for faster searches
- Check your model server performance
- Consider using GPU acceleration for your models

### Getting Help

- Check `qwen-rag --help` for all available commands
- Run `python test_setup.py` to verify installation
- Use `qwen-rag stats` to check database status
- Visit our [GitHub Issues](https://github.com/yourusername/QwenRag/issues) for support

## 🔗 Related Projects

- **LM Studio**: [https://lmstudio.ai/](https://lmstudio.ai/) - Easy local model hosting
- **Ollama**: [https://ollama.ai/](https://ollama.ai/) - Run LLMs locally
- **Qwen Models**: [https://github.com/QwenLM](https://github.com/QwenLM) - State-of-the-art language models

---

**Made with ❤️ for developers who love intelligent code search** 