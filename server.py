"""MCP Server for Knowledge Indexer search functionality.

This server exposes the search_similar tool via the Model Context Protocol,
allowing AI assistants like Claude Code, Cursor, and VS Code agents to search
the indexed GitHub repository knowledge base.
"""

from dotenv import load_dotenv
from fastmcp import FastMCP

from askgit.services.ingestion import IngestionService

load_dotenv()

mcp = FastMCP("askgit")
ingestion_service = IngestionService()


@mcp.tool()
async def search_similar(query: str) -> str:
    """Search for similar code and documentation in the indexed repositories.

    This tool performs semantic search across all indexed GitHub repositories,
    returning the most relevant code snippets and documentation based on the
    query using vector similarity search.

    Args:
        query: The search query describing what you're looking for.
               Can be natural language or code-related terms.

    Returns:
        A formatted string containing the most relevant search results,
        including file paths, content, and metadata.

    Example:
        search_similar("How to implement authentication in FastAPI")
    """
    # Perform the search
    results = await ingestion_service.search_similar(query)

    if not results:
        return "No results found for the query."

    # Format results for better readability
    formatted_results = []
    for idx, doc in enumerate(results, 1):
        result_lines = [
            f"### Result {idx}",
            f"**Repository:** {doc.repo_url or 'Unknown'}",
        ]

        # Add metadata if available
        if doc.meta:
            if "file_path" in doc.meta:
                result_lines.append(f"**File:** {doc.meta['file_path']}")
            if "language" in doc.meta:
                result_lines.append(f"**Language:** {doc.meta['language']}")
            if "chunk_index" in doc.meta:
                result_lines.append(f"**Chunk:** {doc.meta['chunk_index']}")

        # Add content
        result_lines.append("\n**Content:**")
        result_lines.append("```")
        result_lines.append(doc.content or "")
        result_lines.append("```")
        result_lines.append("")

        formatted_results.append("\n".join(result_lines))

    return "\n".join(formatted_results)


# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
