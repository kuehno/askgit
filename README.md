# askgit

A full-stack vector-based knowledge indexing system for GitHub repositories that scrapes code, generates embeddings, and enables semantic search.

Based upon and extends the great work done by Karpathy: https://github.com/karpathy/rendergit

## Overview

Askgit clones GitHub repositories, chunks code files intelligently, generates embeddings using LiteLLM, and stores them in PostgreSQL with pgvector for semantic search. It supports parallel batch processing for high-throughput ingestion.

### Code Parsing Strategy

The system employs three different chunking strategies based on file type and language:

1. **AST-Based Chunking** (Python, TypeScript, Java, C#)
   - Uses ASTChunkBuilder for syntax-aware chunking based on [ASTChunk research](https://arxiv.org/abs/2506.15655)
   - Preserves complete code blocks (functions, classes, methods) within chunks
   - Maintains structural integrity of the code

2. **Language-Specific Code Splitting** (JavaScript, Go, Rust, C/C++, etc.)
   - Uses tree-sitter based CodeSplitter from llama_index for languages not supported by AST chunker
   - Respects code structure with configurable line-based chunking and overlap
   - Supports 15+ programming languages

3. **Semantic Chunking** (Markdown, text, unsupported file types)
   - Uses GPT-4 tokenizer-based semantic chunking via semchunk
   - Token-aware chunking for non-code content with configurable overlap
   - Ensures chunks maintain semantic coherence


## Setup

### Prerequisites

- Python 3.13+
- Docker and Docker Compose
- UV package manager

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

Copy the template and configure your settings:

```bash
cp .env.template .env
```

Edit `.env` to set:
- Database credentials (`POSTGRES_*`)
- Embedding model configuration (`EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`)
- Optional: API credentials for external embedding providers

### 3. Start PostgreSQL Database

```bash
docker compose up -d
```

This starts a PostgreSQL instance with the pgvector extension enabled.

### 4. Run Database Migrations

```bash
uv run alembic upgrade head
```

This creates the `knowledge` schema and `github` table with vector support.

## Usage

### Scrape and Index a Repository

```bash
uv run scrape-github <repo-url>
```

### Use an agent framework (e.g. agno or whatever you prefer to connect to the local mcp server)

```bash
uv run agno_agent_demo.py "<query>"
```

### Run the MCP Server

Start the local MCP server to expose search functionality to AI assistants:

```bash
uv run fastmcp run server.py
```

This will start the server with two available tools:
- `search_similar`: Search for similar code/documentation using semantic search
- `check_repo_exists`: Check if a repository has been indexed

#### Integrating with Claude Code

Add to your Claude Code MCP settings (`~/.config/claude-code/mcp_settings.json`):

```json
{
  "mcpServers": {
    "askgit": {
      "command": "uv",
      "args": ["run", "fastmcp", "run", "server.py"],
      "cwd": "/path/to/askgit",
      "env": {
        "POSTGRES_SERVER": "localhost",
        "POSTGRES_DB": "index",
        "POSTGRES_USER": "root",
        "POSTGRES_PASSWORD": "your-password",
        "POSTGRES_PORT": "5432",
        "POSTGRES_SCHEMA": "knowledge",
        "EMBEDDING_MODEL": "bedrock/eu.cohere.embed-v4:0",
        "EMBEDDING_DIMENSIONS": "1536",
        "API_KEY": "your-api-key-if-needed",
        "API_BASE": "your-api-base-if-needed"
      }
    }
  }
}
```

**Note:** Make sure to replace `/path/to/askgit` with the actual path to this repository and configure your environment variables (especially database credentials and API keys) according to your setup.

### Core Components

- **GithubScraper**: Clones repositories and extracts code files with intelligent chunking
- **EmbeddingService**: Generates embeddings via LiteLLM with batch processing
- **IngestionService**: Stores documents with parallel batch ingestion (10x concurrent batches)
- **Vector Search**: Semantic search using pgvector's cosine similarity

## Architecture

1. **Scraping**: Clones repo, filters files, chunks code using AST-aware splitting
2. **Embedding**: Generates embeddings in batches with configurable concurrency
3. **Storage**: PostgreSQL with pgvector extension and HNSW indexing
4. **Search**: Vector similarity search with configurable thresholds

## Development

- **Code Style**: Ruff (88 char line limit)
- **Type Checking**: Full type hints required
- **Package Management**: UV only (never pip)
- **Migrations**: Alembic for schema changes
