#!/usr/bin/env python3
"""Debug script to test search functionality step by step."""

import asyncio
import sys
from code_rag.config import load_config
from code_rag.search import SearchService

async def debug_search():
    """Debug the search functionality step by step."""
    
    print("🔍 Debug Search Test")
    print("=" * 30)
    
    try:
        # Step 1: Load configuration
        print("1. Loading configuration...")
        config = load_config()
        print(f"   ✅ Config loaded: {config.api.base_url}")
        
        # Step 2: Create search service
        print("2. Creating search service...")
        search_service = SearchService(config)
        print("   ✅ Search service created")
        
        # Step 3: Test embedding generation
        print("3. Testing embedding generation...")
        query = "tree-sitter parsing"
        embedding = await search_service.embedding_service.embed_text(query)
        print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
        
        # Step 4: Test database initialization
        print("4. Testing database initialization...")
        await search_service.db_manager.initialize()
        print("   ✅ Database initialized")
        
        # Step 5: Test vector search
        print("5. Testing vector search...")
        candidates = await search_service.db_manager.search_similar(embedding, 5)
        print(f"   ✅ Vector search completed: {len(candidates)} results")
        
        # Step 6: Test full search (without reranking)
        print("6. Testing full search (no reranking)...")
        config.search.use_reranking = False
        result = await search_service.search(query, top_k=3)
        print(f"   ✅ Search completed: {len(result.results)} results")
        
        # Display results
        print("\n📊 Search Results:")
        for i, res in enumerate(result.results, 1):
            print(f"{i}. {res.chunk.file_path}:{res.chunk.start_line} (Score: {res.score:.3f})")
            print(f"   Type: {res.chunk.chunk_type}")
            print(f"   Content: {res.chunk.content[:100]}...")
            print()
        
        await search_service.close()
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(debug_search())
    if success:
        print("\n🎉 Search functionality is working!")
    else:
        print("\n❌ Search functionality has issues.")
        sys.exit(1) 