#!/usr/bin/env python3
"""Test script to verify API connection with Qwen models."""

import asyncio
import os
from openai import AsyncOpenAI
from code_rag.client import ExtendedOpenaiClient

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
    
    client = ExtendedOpenaiClient(
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
    
    # Test 2: Reranking with New API
    print("\n🔄 Testing New Reranking API...")
    try:
        query = "function that prints hello world"
        documents = [
            "def hello_world(): print('Hello, World!')",
            "def calculate_sum(a, b): return a + b",
            "print('This is a test')",
            "def greet(): return 'Hello there!'"
        ]
        
        result = await client.rerank(
            model=reranking_model,
            query=query,
            documents=documents,
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            top_k=2,
            return_documents=True
        )
        
        print(f"✅ Reranking API successful!")
        print(f"   Results: {result}")
        
    except Exception as e:
        print(f"❌ Reranking API failed: {e}")
        # Fallback to original reranking test
        print("   Trying fallback reranking method...")
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
            
            print(f"✅ Fallback reranking successful!")
            print(f"   Response: {response.choices[0].message.content}")
            
            if response.choices[0].logprobs:
                print(f"   Logprobs available: {len(response.choices[0].logprobs.content)} tokens")
                
        except Exception as fallback_e:
            print(f"❌ Fallback reranking also failed: {fallback_e}")
            return False
    
    print("\n🎉 All API tests passed!")
    
    # Clean up
    await client.close()
    return True

if __name__ == "__main__":
    success = asyncio.run(test_api())
    if success:
        print("\n✅ Your Qwen models are ready to use!")
        print("You can now run: python main.py search 'your query'")
    else:
        print("\n❌ API tests failed. Please check your endpoint and models.") 