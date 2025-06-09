"""Tree-sitter utilities for intelligent code parsing and chunking."""

import os
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, NamedTuple
from pathlib import Path

try:
    import tree_sitter_python as tspython
except ImportError:
    tspython = None

try:
    import tree_sitter_javascript as tsjavascript
except ImportError:
    tsjavascript = None

try:
    import tree_sitter_typescript as tstypescript
except ImportError:
    tstypescript = None

try:
    import tree_sitter_java as tsjava
except ImportError:
    tsjava = None

try:
    import tree_sitter_cpp as tscpp
except ImportError:
    tscpp = None

try:
    import tree_sitter_c as tsc
except ImportError:
    tsc = None

try:
    import tree_sitter_c_sharp as tscsharp
except ImportError:
    tscsharp = None

try:
    import tree_sitter_rust as tsrust
except ImportError:
    tsrust = None

try:
    import tree_sitter_go as tsgo
except ImportError:
    tsgo = None

from tree_sitter import Language, Parser, Node


class ChunkWithLocation(NamedTuple):
    """Code chunk with location information."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int


def _try_get_language(module, module_name: str):
    """Try different ways to get the language from a tree-sitter module."""
    if module is None:
        return None
    
    # Try different attribute names that different versions use
    for attr_name in ['language', 'LANGUAGE', 'Language']:
        if hasattr(module, attr_name):
            try:
                lang_func = getattr(module, attr_name)
                if callable(lang_func):
                    return Language(lang_func())
                else:
                    return Language(lang_func)
            except Exception as e:
                print(f"Failed to load {module_name} with {attr_name}: {e}")
                continue
    
    print(f"Could not find language function in {module_name}")
    return None


class TreeSitterManager:
    """Manages tree-sitter parsers for different languages."""
    
    def __init__(self):
        self._parsers: Dict[str, Parser] = {}
        self._languages = {}
        
        # Initialize languages with fallback handling
        language_modules = [
            ('.py', tspython, 'tree_sitter_python'),
            ('.js', tsjavascript, 'tree_sitter_javascript'),  
            ('.jsx', tsjavascript, 'tree_sitter_javascript'),
            ('.ts', tstypescript, 'tree_sitter_typescript'),
            ('.tsx', tstypescript, 'tree_sitter_typescript'),
            ('.java', tsjava, 'tree_sitter_java'),
            ('.cpp', tscpp, 'tree_sitter_cpp'),
            ('.cc', tscpp, 'tree_sitter_cpp'),
            ('.cxx', tscpp, 'tree_sitter_cpp'),
            ('.hpp', tscpp, 'tree_sitter_cpp'),
            ('.h', tsc, 'tree_sitter_c'),
            ('.c', tsc, 'tree_sitter_c'),
            ('.cs', tscsharp, 'tree_sitter_c_sharp'),
            ('.rs', tsrust, 'tree_sitter_rust'),
            ('.go', tsgo, 'tree_sitter_go'),
        ]
        
        for ext, module, name in language_modules:
            lang = _try_get_language(module, name)
            if lang:
                self._languages[ext] = lang
        
        print(f"✅ Initialized tree-sitter for {len(self._languages)} language(s): {', '.join(self._languages.keys())}")
    
    def get_parser(self, file_extension: str) -> Optional[Parser]:
        """Get or create a parser for the given file extension."""
        if file_extension not in self._languages:
            return None
            
        if file_extension not in self._parsers:
            parser = Parser()
            try:
                # Try both old and new API methods
                if hasattr(parser, 'set_language'):
                    parser.set_language(self._languages[file_extension])
                elif hasattr(parser, 'language'):
                    parser.language = self._languages[file_extension]
                else:
                    print(f"Could not set language for {file_extension}: No suitable method found")
                    return None
                    
                self._parsers[file_extension] = parser
            except Exception as e:
                print(f"Error setting language for {file_extension}: {e}")
                return None
            
        return self._parsers[file_extension]
    
    def parse_file(self, file_path: str, content: str) -> Optional[Node]:
        """Parse a file and return the root node."""
        file_extension = Path(file_path).suffix.lower()
        parser = self.get_parser(file_extension)
        
        if not parser:
            return None
            
        try:
            tree = parser.parse(content.encode('utf-8'))
            return tree.root_node
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters for code)."""
    return len(text) // 4


def collapse_node_content(node: Node, original_content: str) -> str:
    """Create a collapsed version of a node's content."""
    if node.type in ["block", "statement_block", "compound_statement"]:
        return "{ ... }"
    elif node.type in ["function_body", "method_body"]:
        return "{ ... }"
    else:
        return "..."


class CodeChunker:
    """Intelligent code chunker using tree-sitter for semantic chunking."""
    
    def __init__(self, max_chunk_tokens: int = 1000):
        self.max_chunk_tokens = max_chunk_tokens
        self.ts_manager = TreeSitterManager()
        
        # Node types that represent complete functions/methods
        self.function_types = {
            "function_definition", "function_declaration", "method_definition", 
            "method_declaration", "function_item", "arrow_function", "function",
            "method", "constructor_definition", "function_expression"
        }
        
        # Node types that represent classes/interfaces
        self.class_types = {
            "class_definition", "class_declaration", "interface_declaration",
            "struct_item", "impl_item", "trait_item", "class", "interface",
            "struct", "enum", "trait"
        }
        
        # Node types that represent code blocks
        self.block_types = {
            "block", "statement_block", "compound_statement", "function_body",
            "class_body", "declaration_list"
        }
    
    async def chunk_file(self, file_path: str, content: str) -> List[ChunkWithLocation]:
        """Chunk a file into semantically meaningful pieces based on functions and classes."""
        if not content.strip():
            return []
            
        root_node = self.ts_manager.parse_file(file_path, content)
        if not root_node:
            print(f"⚠️  No parser available for {file_path}, using simple text chunking")
            return self._simple_text_chunk(file_path, content)
        
        print(f"🌳 Using tree-sitter parsing for {file_path}")
        chunks = []
        
        # Extract function and class-level chunks
        await self._extract_semantic_chunks(root_node, content, file_path, chunks)
            
        # If no meaningful chunks found, fall back to simple chunking
        if not chunks:
            print(f"⚠️  No semantic chunks found in {file_path}, using simple text chunking")
            chunks = self._simple_text_chunk(file_path, content)
            
        return chunks
    
    async def _extract_semantic_chunks(self, node: Node, content: str, file_path: str, chunks: List[ChunkWithLocation]):
        """Extract semantic chunks (functions, classes, etc.) from the AST."""
        
        # Check if this node is a function or class we want to extract
        if node.type in self.function_types:
            await self._add_function_chunk(node, content, file_path, chunks)
            return  # Don't recurse into children of functions
            
        if node.type in self.class_types:
            await self._add_class_chunk(node, content, file_path, chunks)
            # Also recurse into class body to extract individual methods
            for child in node.children:
                if child.type in self.block_types:
                    for method in child.children:
                        if method.type in self.function_types:
                            await self._add_function_chunk(method, content, file_path, chunks)
            return
        
        # For other nodes, recurse into children
        for child in node.children:
            await self._extract_semantic_chunks(child, content, file_path, chunks)
    
    async def _add_function_chunk(self, node: Node, content: str, file_path: str, chunks: List[ChunkWithLocation]):
        """Add a function as a chunk."""
        function_content = content[node.start_byte:node.end_byte]
        
        # If function is too large, create a collapsed version
        if estimate_token_count(function_content) > self.max_chunk_tokens:
            collapsed_content = await self._create_collapsed_function(node, content)
            function_content = collapsed_content
        
        chunks.append(ChunkWithLocation(
            content=function_content,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_char=node.start_byte,
            end_char=node.end_byte
        ))
    
    async def _add_class_chunk(self, node: Node, content: str, file_path: str, chunks: List[ChunkWithLocation]):
        """Add a class as a chunk (collapsed overview)."""
        class_content = content[node.start_byte:node.end_byte]
        
        # Create collapsed version showing class structure
        if estimate_token_count(class_content) > self.max_chunk_tokens:
            collapsed_content = await self._create_collapsed_class(node, content)
            class_content = collapsed_content
        
        chunks.append(ChunkWithLocation(
            content=class_content,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_char=node.start_byte,
            end_char=node.end_byte
        ))
    
    async def _create_collapsed_function(self, node: Node, content: str) -> str:
        """Create a collapsed version of a function showing signature and key structure."""
        full_content = content[node.start_byte:node.end_byte]
        lines = full_content.split('\n')
        
        # Keep function signature and first few lines, collapse body
        if len(lines) <= 5:
            return full_content
        
        # Find where the function body starts (after the signature)
        signature_lines = []
        body_start_idx = 0
        
        for i, line in enumerate(lines):
            signature_lines.append(line)
            if '{' in line or ':' in line:  # Function body indicator
                body_start_idx = i + 1
                break
        
        # Add collapsed body indicator
        if body_start_idx < len(lines):
            signature_lines.append("    # ... function body ...")
            # Add closing brace/return if present
            if lines[-1].strip():
                signature_lines.append(lines[-1])
        
        return '\n'.join(signature_lines)
    
    async def _create_collapsed_class(self, node: Node, content: str) -> str:
        """Create a collapsed version of a class showing structure."""
        full_content = content[node.start_byte:node.end_byte]
        lines = full_content.split('\n')
        
        # Keep class definition and method signatures
        result_lines = []
        indent_level = 0
        
        for line in lines[:10]:  # First 10 lines for class overview
            stripped = line.strip()
            if stripped.startswith('class ') or stripped.startswith('def ') or stripped.startswith('async def '):
                result_lines.append(line)
            elif not stripped and len(result_lines) > 0:
                result_lines.append(line)
        
        if len(lines) > 10:
            result_lines.append("    # ... class body ...")
        
        return '\n'.join(result_lines)
    
    def _simple_text_chunk(self, file_path: str, content: str) -> List[ChunkWithLocation]:
        """Fallback simple text chunking when tree-sitter parsing fails."""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_tokens = 0
        start_line = 0
        start_char = 0
        
        for i, line in enumerate(lines):
            line_tokens = estimate_token_count(line)
            
            if current_tokens + line_tokens > self.max_chunk_tokens and current_chunk:
                # Create chunk from current content
                chunk_content = '\n'.join(current_chunk)
                end_char = start_char + len(chunk_content)
                
                chunks.append(ChunkWithLocation(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=i - 1,
                    start_char=start_char,
                    end_char=end_char
                ))
                
                # Start new chunk
                current_chunk = [line]
                current_tokens = line_tokens
                start_line = i
                start_char = end_char + 1
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            end_char = start_char + len(chunk_content)
            
            chunks.append(ChunkWithLocation(
                content=chunk_content,
                file_path=file_path,
                start_line=start_line,
                end_line=len(lines) - 1,
                start_char=start_char,
                end_char=end_char
            ))
        
        return chunks 