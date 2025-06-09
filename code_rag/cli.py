"""Command Line Interface for Qwen RAG system."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

from .config import load_config, CodeRAGConfig
from .indexer import RepositoryIndexer
from .search import SearchService
from .database import DatabaseManager


def get_config(ctx) -> CodeRAGConfig:
    """Get configuration from context or create new one."""
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    # Override with CLI parameters
    if ctx.obj:
        for key, value in ctx.obj.items():
            if key == "config_file":
                continue
            if value is not None:
                if key in ["api_base", "api_key", "embedding_model", "reranking_model"]:
                    setattr(config.api, key.replace("api_", ""), value)
                elif key == "db_path":
                    config.database.path = value
                elif key == "chunk_size":
                    config.chunking.max_tokens = value
                elif key == "disable_reranking":
                    config.search.use_reranking = not value
                elif key == "top_k":
                    config.search.top_k_final = value
    
    return config


@click.group()
@click.option('--config-file', type=click.Path(exists=True, path_type=Path), 
              help='Path to configuration file')
@click.option('--api-base', help='API base URL')
@click.option('--api-key', help='API key')
@click.option('--embedding-model', help='Embedding model name')
@click.option('--reranking-model', help='Reranking model name')
@click.option('--db-path', help='Database path')
@click.option('--disable-reranking', is_flag=True, help='Disable reranking')
@click.pass_context
def cli(ctx, **kwargs):
    """Qwen RAG - Repository Retrieval Augmented Generation System"""
    ctx.ensure_object(dict)
    ctx.obj.update({k: v for k, v in kwargs.items() if v is not None})


@cli.command()
@click.argument('repository_path', type=click.Path(exists=True, path_type=Path))
@click.option('--chunk-size', type=int, help='Maximum tokens per chunk')
@click.option('--batch-size', default=10, help='Number of files to process in each batch')
@click.option('--force', is_flag=True, help='Force reindexing even if repository is already indexed')
@click.pass_context
def index(ctx, repository_path: Path, chunk_size: Optional[int], batch_size: int, force: bool):
    """Index a repository for search."""
    config = get_config(ctx)
    
    if chunk_size:
        config.chunking.max_tokens = chunk_size
    
    async def _index():
        indexer = RepositoryIndexer(config)
        await indexer.index_repository(repository_path, force_reindex=force)
    
    asyncio.run(_index())


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--chunk-size', type=int, help='Maximum tokens per chunk')
@click.pass_context
def index_file(ctx, file_path: Path, chunk_size: Optional[int]):
    """Index a single file."""
    config = get_config(ctx)
    
    if chunk_size:
        config.chunking.max_tokens = chunk_size
    
    async def _index_file():
        indexer = RepositoryIndexer(config)
        await indexer.index_single_file(file_path)
    
    asyncio.run(_index_file())


@cli.command()
@click.argument('query')
@click.option('--top-k', type=int, help='Number of results to return')
@click.option('--no-reranking', 'disable_reranking', is_flag=True, help='Disable reranking')
@click.option('--chunk-type', help='Filter by chunk type (function, class, comment, etc.)')
@click.option('--file-type', help='Filter by file extension (e.g., .py, .js)')
@click.option('--no-content', is_flag=True, help='Hide content in results')
@click.option('--max-content', type=int, default=500, help='Maximum content length to display')
@click.pass_context
def search(ctx, query: str, top_k: Optional[int], disable_reranking: bool, 
          chunk_type: Optional[str], file_type: Optional[str], 
          no_content: bool, max_content: int):
    """Search for code chunks."""
    config = get_config(ctx)
    
    if top_k:
        config.search.top_k_final = top_k
    if disable_reranking:
        config.search.use_reranking = False
    
    async def _search():
        search_service = SearchService(config)
        
        start_time = asyncio.get_event_loop().time()
        
        # Choose the appropriate search method based on filters
        if chunk_type:
            query_result = await search_service.search_by_chunk_type(
                query=query,
                chunk_type=chunk_type,
                top_k=config.search.top_k_final
            )
        elif file_type:
            query_result = await search_service.search_by_file_type(
                query=query,
                file_extension=file_type,
                top_k=config.search.top_k_final
            )
        else:
            query_result = await search_service.search(
                query=query,
                top_k=config.search.top_k_final,
                use_reranking=config.search.use_reranking
            )
        
        results = query_result.results
        end_time = asyncio.get_event_loop().time()
        
        print(f"Search Results for: '{query}'")
        print(f"Found {len(results)} results in {(end_time - start_time) * 1000:.1f}ms")
        print(f"Searched {config.search.top_k_initial} chunks")
        print(f"Reranked: {'Yes' if config.search.use_reranking else 'No'}")
        print("-" * 80)
        
        if not results:
            print("No results found for query:", query)
            return
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            score = result.score
            
            print(f"\n{i}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (Score: {score:.3f})")
            print(f"   Type: {chunk.chunk_type} | File: {Path(chunk.file_path).suffix}")
            print(f"   Repository: {chunk.repository_path}")
            
            if not no_content:
                content = chunk.content
                if len(content) > max_content:
                    content = content[:max_content] + "..."
                
                print("   Content:")
                for line in content.split('\n'):
                    print(f"   | {line}")
        
        await search_service.close()
    
    try:
        asyncio.run(_search())
    except Exception as e:
        print(f"❌ Error: {e}")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive search mode."""
    config = get_config(ctx)
    
    async def _interactive():
        search_service = SearchService(config)
        
        print("🔍 Interactive Qwen RAG Search")
        print("Enter your search queries (type 'quit' to exit, 'help' for commands)")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n🔍 Search: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    print("\nCommands:")
                    print("  - Type any search query to find code")
                    print("  - 'stats' - Show database statistics")  
                    print("  - 'config' - Show current configuration")
                    print("  - 'quit' - Exit interactive mode")
                    continue
                elif query.lower() == 'stats':
                    db_manager = DatabaseManager(config)
                    stats = await db_manager.get_stats()
                    print(f"\n📊 Database Statistics")
                    print(f"Total chunks: {stats['total_chunks']}")
                    print(f"Repositories: {stats['repositories']}")
                    continue
                elif query.lower() == 'config':
                    print(f"\n⚙️ Current Configuration")
                    print(f"API Base: {config.api.base_url}")
                    print(f"Embedding Model: {config.api.embedding_model}")
                    print(f"Embedding Max Tokens: {config.api.embedding_max_tokens}")
                    print(f"Reranking Model: {config.api.reranking_model}")
                    print(f"Reranking Max Tokens: {config.api.reranking_max_tokens}")
                    print(f"Database Path: {config.database.path}")
                    print(f"Chunk Size: {config.chunking.max_tokens}")
                    print(f"Use Reranking: {config.search.use_reranking}")
                    print(f"Top-K Final: {config.search.top_k_final}")
                    continue
                elif not query:
                    continue
                
                start_time = asyncio.get_event_loop().time()
                query_result = await search_service.search(query, top_k=config.search.top_k_final)
                end_time = asyncio.get_event_loop().time()
                
                if not query_result.results:
                    print(f"❌ No results found for: '{query}'")
                    continue
                
                print(f"\n✅ Found {len(query_result.results)} results ({(end_time - start_time) * 1000:.1f}ms)")
                
                for i, result in enumerate(query_result.results, 1):
                    chunk = result.chunk
                    score = result.score
                    
                    print(f"\n{i}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (Score: {score:.3f})")
                    content = chunk.content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    
                    for line in content.split('\n')[:5]:  # Show first 5 lines
                        print(f"   | {line}")
                    
                    if len(chunk.content.split('\n')) > 5:
                        print("   | ...")
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        await search_service.close()
    
    asyncio.run(_interactive())


@cli.command()
@click.pass_context
def stats(ctx):
    """Show database statistics."""
    config = get_config(ctx)
    
    async def _stats():
        db_manager = DatabaseManager(config)
        stats = await db_manager.get_stats()
        
        print("📊 Database Statistics")
        print("-" * 30)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Repositories: {stats['repositories']}")
        
        if stats.get('repository_counts'):
            print(f"\n📁 Repository Breakdown:")
            for repo, count in stats['repository_counts'].items():
                print(f"  {repo}: {count} chunks")
        
        if stats.get('file_type_counts'):
            print(f"\n📄 File Type Breakdown:")
            for file_type, count in stats['file_type_counts'].items():
                print(f"  {file_type}: {count} chunks")
        
        if stats.get('chunk_type_counts'):
            print(f"\n🧩 Chunk Type Breakdown:")
            for chunk_type, count in stats['chunk_type_counts'].items():
                print(f"  {chunk_type}: {count} chunks")
    
    asyncio.run(_stats())


@cli.command()
@click.argument('repository_path', type=click.Path(path_type=Path))
@click.pass_context
def delete(ctx, repository_path: Path):
    """Delete indexed data for a repository."""
    config = get_config(ctx)
    
    async def _delete():
        db_manager = DatabaseManager(config)
        deleted_count = await db_manager.delete_repository(str(repository_path.absolute()))
        print(f"✅ Deleted {deleted_count} chunks for repository: {repository_path}")
    
    asyncio.run(_delete())


@cli.command()
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    config = get_config(ctx)
    
    print("\n⚙️  Current Configuration")
    print("-" * 30)
    print(f"API Base URL: {config.api.base_url}")
    print(f"Embedding Model: {config.api.embedding_model}")
    print(f"Embedding Max Tokens: {config.api.embedding_max_tokens}")
    print(f"Reranking Model: {config.api.reranking_model}")
    print(f"Reranking Max Tokens: {config.api.reranking_max_tokens}")
    print(f"Database Path: {config.database.path}")
    print(f"Chunk Size: {config.chunking.max_tokens}")
    print(f"Use Reranking: {config.search.use_reranking}")
    print(f"Top-K Final: {config.search.top_k_final}")


if __name__ == '__main__':
    cli() 