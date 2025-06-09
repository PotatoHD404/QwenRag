#!/usr/bin/env python3
"""
Example usage of the Code RAG system.
This script demonstrates how to use the system programmatically.
"""

import asyncio
import os
from code_rag.config import Config
from code_rag.indexer import RepositoryIndexer
from code_rag.search import SearchService


async def main():
    """Example usage of Code RAG system."""
    
    # Load configuration
    config = Config.load_from_env()
    
    print("🤖 Code RAG Example")
    print("=" * 50)
    print(f"API Base URL: {config.api.base_url}")
    print(f"Embedding Model: {config.api.embedding_model}")
    print(f"Reranking Model: {config.api.reranking_model}")
    print(f"Database Path: {config.db_path}")
    print()
    
    # Example 1: Index current directory
    print("📂 Example 1: Indexing current directory")
    print("-" * 40)
    
    indexer = RepositoryIndexer(config)
    try:
        current_dir = os.getcwd()
        result = await indexer.index_repository(current_dir)
        print(f"✅ Indexing result: {result}")
    except Exception as e:
        print(f"❌ Indexing error: {e}")
    
    print()
    
    # Example 2: Search for code
    print("🔍 Example 2: Searching for code")
    print("-" * 40)
    
    search_service = SearchService(config)
    try:
        # Search queries
        queries = [
            "function definition",
            "class implementation",
            "import statement",
            "error handling",
            "configuration setup"
        ]
        
        for query in queries:
            print(f"\nSearching for: '{query}'")
            result = await search_service.search(query, top_k=3)
            
            if result.results:
                print(f"Found {len(result.results)} results in {result.execution_time_ms:.1f}ms")
                for i, search_result in enumerate(result.results, 1):
                    chunk = search_result.chunk
                    print(f"  {i}. {chunk.get_location_string()} (Type: {chunk.chunk_type})")
            else:
                print("  No results found")
    
    except Exception as e:
        print(f"❌ Search error: {e}")
    finally:
        await search_service.close()
    
    print()
    print("✨ Example completed!")


if __name__ == "__main__":
    asyncio.run(main()) 