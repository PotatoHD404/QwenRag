#!/usr/bin/env python3
"""Setup verification script for Qwen RAG system."""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test importing all required dependencies."""
    print("🚀 Qwen RAG Setup Test")
    print("=" * 50)
    print("🔍 Testing imports...")
    print("-" * 40)
    
    modules_to_test = [
        "lancedb",
        "tree_sitter", 
        "tree_sitter_python",
        "tree_sitter_javascript", 
        "tree_sitter_typescript",
        "tree_sitter_java",
        "tree_sitter_cpp",
        "tree_sitter_c",
        "tree_sitter_c_sharp",
        "tree_sitter_rust",
        "tree_sitter_go",
        "openai",
        "click",
        "pydantic",
        "tqdm",
        "gitignore_parser"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    print(f"\n📊 Results: {len(modules_to_test) - len(failed_imports)}/{len(modules_to_test)} modules imported successfully")
    
    if not failed_imports:
        print("🎉 All dependencies are installed correctly!")
        return True
    else:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False


def test_code_rag_package():
    """Test importing the Qwen RAG package."""
    print("\n🔍 Testing Qwen RAG package...")
    print("-" * 40)
    
    try:
        # Test individual modules
        from code_rag.config import CodeRAGConfig, load_config
        from code_rag.tree_sitter_utils import TreeSitterManager, CodeChunker
        from code_rag.embeddings import EmbeddingService, RerankingService
        from code_rag.database import DatabaseManager, CodeChunk
        from code_rag.indexer import RepositoryIndexer
        from code_rag.search import SearchService
        from code_rag.cli import cli
        
        print("✅ All Qwen RAG modules imported successfully!")
        
        # Test configuration loading
        config = load_config()
        print(f"✅ Configuration loaded: {config.api.base_url}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error importing Qwen RAG package: {e}")
        traceback.print_exc()
        return False


def test_tree_sitter():
    """Test tree-sitter functionality."""
    print("\n🔍 Testing Tree-sitter...")
    print("-" * 40)
    
    try:
        from code_rag.tree_sitter_utils import TreeSitterManager, CodeChunker
        
        # Initialize tree-sitter manager
        ts_manager = TreeSitterManager()
        
        # Test a simple Python code parsing
        test_code = '''
def hello_world():
    """A simple test function."""
    print("Hello, World!")
    return True

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
        
        # Test parsing
        root_node = ts_manager.parse_file("test.py", test_code)
        if root_node:
            print("✅ Tree-sitter parsing working")
            
            # Test chunking
            chunker = CodeChunker(max_chunk_tokens=500)
            import asyncio
            chunks = asyncio.run(chunker.chunk_file("test.py", test_code))
            
            print(f"✅ Code chunking working: {len(chunks)} chunks generated")
            
            # Show chunk types
            chunk_types = [chunk.content[:50] + "..." if len(chunk.content) > 50 else chunk.content for chunk in chunks]
            for i, chunk_preview in enumerate(chunk_types):
                print(f"   Chunk {i+1}: {chunk_preview}")
            
            return True
        else:
            print("⚠️  Tree-sitter parsing not working (using fallback text chunking)")
            return True  # Still OK, just using fallback
            
    except Exception as e:
        print(f"❌ Tree-sitter test failed: {e}")
        traceback.print_exc()
        return False


def test_api_connection():
    """Test API connection (optional)."""
    print("\n🔍 Testing API connection...")
    print("-" * 40)
    
    try:
        from code_rag.config import load_config
        from code_rag.embeddings import EmbeddingService
        
        config = load_config()
        print(f"📡 Testing connection to: {config.api.base_url}")
        print(f"🤖 Embedding model: {config.api.embedding_model}")
        print(f"🔄 Reranking model: {config.api.reranking_model}")
        print(f"📏 Embedding max tokens: {config.api.embedding_max_tokens}")
        print(f"📏 Reranking max tokens: {config.api.reranking_max_tokens}")
        
        # Note: We don't actually test the connection here to avoid API calls
        # during setup verification
        print("⚠️  API connection test skipped (would require live API endpoint)")
        
        return True
        
    except Exception as e:
        print(f"❌ API configuration test failed: {e}")
        return False


def main():
    """Run all setup tests."""
    tests = [
        ("Import Test", test_imports),
        ("Code RAG Package", test_code_rag_package), 
        ("Tree-sitter", test_tree_sitter),
        ("API Configuration", test_api_connection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Final Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Setup test completed successfully!")
        print("\nYou can now use the Qwen RAG system:")
        print("  python main.py --help")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 