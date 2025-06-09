"""Setup script for Qwen RAG package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="qwen-rag",
    version="0.1.0",
    author="Qwen RAG Contributors",
    author_email="",
    description="A powerful RAG system for querying code repositories using tree-sitter parsing, LanceDB vector storage, and Qwen models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PotatoHD404/QwenRag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'code_rag': ['config.example.yaml'],
    },
    entry_points={
        'console_scripts': [
            'qwen-rag=code_rag.cli:cli',
        ],
    },
    keywords="rag, code search, embeddings, vector database, qwen, tree-sitter, semantic search",
    project_urls={
        "Bug Reports": "https://github.com/PotatoHD404/QwenRag/issues",
        "Source": "https://github.com/PotatoHD404/QwenRag",
        "Documentation": "https://github.com/PotatoHD404/QwenRag#readme",
    },
) 