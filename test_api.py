#!/usr/bin/env python3
"""Test script to verify API connection with Qwen models."""

import asyncio
import os
from openai import AsyncOpenAI

async def test_api():
    """Test the API connection and models."""
    
    # Configuration
    api_base = os.getenv("RAG_API_BASE", "http://localhost:1234/v1")
    api_key = os.getenv("RAG_API_KEY", "dummy")
    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b")
    reranking_model = os.getenv("RAG_RERANKING_MODEL", "qwen.qwen3-reranker-4b")
    
    print("🔍 Testing Qwen RAG API Connection")
    print("=" * 50)
    print(f"API Base: {api_base}")
    print(f"Embedding Model: {embedding_model}")
    print(f"Reranking Model: {reranking_model}")
    print()
    
    client = AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=30
    )
    
    # Test 1: Embedding
    print("🧮 Testing Embedding Model...")
    try:
        test_text = "def hello_world(): print('Hello, World!')"
        
        response = await client.embeddings.create(
            model=embedding_model,
            input=[test_text],
            encoding_format="float"
        )
        
        embedding = response.data[0].embedding
        print(f"✅ Embedding successful! Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return False
    
    # Test 2: Reranking
    print("\n🔄 Testing Reranking Model...")
    try:
        query = "function that prints hello world"
        document = "def hello_world(): print('Hello, World!')"
        
        prompt = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n<Query>: {query}\n\n<Document>: {document}"
        
        response = await client.chat.completions.create(
            model=reranking_model,
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
        
        print(f"✅ Reranking successful!")
        print(f"   Response: {response.choices[0].message.content}")
        
        if response.choices[0].logprobs:
            print(f"   Logprobs available: {len(response.choices[0].logprobs.content)} tokens")
        
    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        return False
    
    print("\n🎉 All API tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_api())
    if success:
        print("\n✅ Your Qwen models are ready to use!")
        print("You can now run: python main.py search 'your query'")
    else:
        print("\n❌ API tests failed. Please check your endpoint and models.") 