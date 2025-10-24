"""GitHub repository scraper for extracting and indexing code files."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import nbformat
import semchunk
from astchunk import ASTChunkBuilder
from llama_index.core.node_parser.text.code import CodeSplitter
from loguru import logger

# Maximum file size to include (50KB default)
MAX_DEFAULT_BYTES = 50 * 1024

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wav",
    ".ogg",
    ".flac",
    ".ttf",
    ".otf",
    ".eot",
    ".woff",
    ".woff2",
    ".so",
    ".dll",
    ".dylib",
    ".class",
    ".jar",
    ".exe",
    ".bin",
}


@dataclass
class RenderDecision:
    """Decision about whether to include a file."""

    include: bool
    reason: str  # "ok" | "binary" | "too_large" | "ignored"


@dataclass
class FileInfo:
    """Information about a file in the repository."""

    path: pathlib.Path  # absolute path on disk
    rel: str  # relative path from repo root
    size: int
    decision: RenderDecision


def git_clone(repo_url: str, dest_dir: str) -> None:
    """Clone a git repository to a destination directory.

    Args:
        repo_url: URL of the git repository to clone.
        dest_dir: Destination directory path.

    Raises:
        RuntimeError: If git clone fails.
    """
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, dest_dir],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git clone failed: {e.stderr}") from e


def git_head_commit(repo_dir: str) -> str:
    """Get the HEAD commit hash of a git repository.

    Args:
        repo_dir: Path to the git repository.

    Returns:
        The HEAD commit hash.

    Raises:
        RuntimeError: If git command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get HEAD commit: {e.stderr}") from e


def should_ignore(path: pathlib.Path, repo_dir: pathlib.Path) -> bool:
    """Check if a file should be ignored.

    Args:
        path: Path to check.
        repo_dir: Root directory of the repository.

    Returns:
        True if the file should be ignored.
    """
    # Get relative path
    try:
        rel = path.relative_to(repo_dir)
    except ValueError:
        return True

    parts = rel.parts

    # Ignore .git directory
    if ".git" in parts:
        return True

    # Ignore common build/dependency directories
    ignore_dirs = {
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".egg-info",
        "target",
    }

    if any(part in ignore_dirs for part in parts):
        return True

    return False


def decide_render(path: pathlib.Path, max_bytes: int) -> RenderDecision:
    """Decide whether to include a file based on its properties.

    Args:
        path: Path to the file.
        max_bytes: Maximum file size to include.

    Returns:
        RenderDecision indicating whether to include the file.
    """
    ext = path.suffix.lower()

    # Check if binary
    if ext in BINARY_EXTENSIONS:
        return RenderDecision(include=False, reason="binary")

    # Check file size
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return RenderDecision(include=False, reason="too_large")
    except OSError:
        return RenderDecision(include=False, reason="ignored")

    return RenderDecision(include=True, reason="ok")


def collect_files(repo_dir: pathlib.Path, max_bytes: int) -> list[FileInfo]:
    """Collect all files from a repository with metadata.

    Args:
        repo_dir: Root directory of the repository.
        max_bytes: Maximum file size to include.

    Returns:
        List of FileInfo objects for all files.
    """
    infos: list[FileInfo] = []

    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue

        if should_ignore(path, repo_dir):
            continue

        try:
            rel = str(path.relative_to(repo_dir))
            size = path.stat().st_size
            decision = decide_render(path, max_bytes)

            infos.append(FileInfo(path=path, rel=rel, size=size, decision=decision))
        except (OSError, ValueError) as e:
            logger.warning(f"Skipping file {path}: {e}")
            continue

    return infos


def read_text(path: pathlib.Path) -> str:
    """Read text content from a file.

    Args:
        path: Path to the file.

    Returns:
        The file content as a string.

    Raises:
        OSError: If file cannot be read.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def is_jupyter_notebook(path: pathlib.Path) -> bool:
    """Check if a file is a Jupyter notebook.

    Args:
        path: Path to the file.

    Returns:
        True if the file is a Jupyter notebook.
    """
    return path.suffix.lower() == ".ipynb"


def extract_notebook_content(path: pathlib.Path) -> str:
    """Extract content from a Jupyter notebook.

    Extracts both code and markdown cells from the notebook,
    formatting them with cell type indicators.

    Args:
        path: Path to the Jupyter notebook file.

    Returns:
        Extracted content as a formatted string.

    Raises:
        OSError: If file cannot be read.
        nbformat.reader.NotJSONError: If file is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    content_parts = []

    for idx, cell in enumerate(notebook["cells"]):
        cell_type = cell["cell_type"]
        source = cell.get("source", "")

        # Skip empty cells
        if not source or not source.strip():
            continue

        # Add cell type header
        if cell_type == "code":
            content_parts.append(f"\n## Code Cell {idx + 1}\n```python\n{source}\n```")
        elif cell_type == "markdown":
            content_parts.append(f"\n## Markdown Cell {idx + 1}\n{source}")
        else:
            # Handle other cell types (raw, etc.)
            content_parts.append(f"\n## {cell_type.title()} Cell {idx + 1}\n{source}")

    return "\n".join(content_parts) if content_parts else ""


def bytes_human(num_bytes: int) -> str:
    """Convert bytes to human-readable format.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Human-readable string (e.g., "1.5 KB").
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def detect_language_from_extension(file_path: str) -> str | None:
    """Detect programming language from file extension.

    Only returns languages supported by CodeSplitter. For unsupported
    file types, returns None to trigger semantic chunking fallback.

    Args:
        file_path: Path to the file.

    Returns:
        Programming language name if supported by CodeSplitter, None otherwise.
    """
    # Languages supported by CodeSplitter (tree-sitter based)
    ext_to_lang = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "bash",
        ".ipynb": "python",  # Jupyter notebooks contain Python code
    }

    ext = pathlib.Path(file_path).suffix.lower()
    return ext_to_lang.get(ext)


class GithubScraper:
    """Service for scraping GitHub repositories and extracting code files."""

    def __init__(
        self,
        max_file_size: int = MAX_DEFAULT_BYTES,
        chunk_lines: int = 40,
        chunk_lines_overlap: int = 15,
        chunk_size: int = 1024,
        chunk_overlap: int = 64,
    ) -> None:
        """Initialize the GitHub scraper.

        Args:
            max_file_size: Maximum file size in bytes to include.
            chunk_lines: Maximum number of lines per chunk for code (default: 80).
            chunk_lines_overlap: Lines to overlap between code chunks (default: 15).
            chunk_size: Maximum characters per code chunk (default: 1500).
            chunk_overlap: Token overlap for semantic chunking (default: 128).
        """
        self.max_file_size = max_file_size
        self.chunk_lines = chunk_lines
        self.chunk_lines_overlap = chunk_lines_overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ast_chunker_supported_languages = [
            "python",
            "typescript",
            "java",
            "csharp",
        ]

        # Cache code splitters per language for efficiency
        self._code_splitter_cache: dict[str, CodeSplitter] = {}
        self._ast_chunker_cache: dict[str, ASTChunkBuilder] = {}

        # Semantic chunker for non-code files (markdown, text, etc.)
        # Using GPT-4 tokenizer for token-aware semantic chunking
        self._semantic_chunker = semchunk.chunkerify("gpt-4", chunk_size)

        logger.info(
            f"Initialized GithubScraper with max file size: "
            f"{bytes_human(max_file_size)}, "
            f"code: {chunk_lines} lines/{chunk_size} chars, "
            f"semantic: {chunk_size} tokens"
        )

    def _get_code_splitter(self, language: str) -> CodeSplitter:
        """Get or create a CodeSplitter for the given language.

        Args:
            language: Programming language.

        Returns:
            CodeSplitter instance for the language.
        """
        if language not in self._code_splitter_cache:
            self._code_splitter_cache[language] = CodeSplitter(
                language=language,
                chunk_lines=self.chunk_lines,
                chunk_lines_overlap=self.chunk_lines_overlap,
                max_chars=self.chunk_size,
            )
            logger.debug(f"Created CodeSplitter for language: {language}")
        return self._code_splitter_cache[language]

    def _get_ast_chunker(self, language: str) -> None:
        if (
            language not in self._code_splitter_cache
            and language in self.ast_chunker_supported_languages
        ):
            self._ast_chunker_cache[language] = ASTChunkBuilder(
                language=language,
                metadata_template="default",
                chunk_overlap=3,
                chunk_expansion=True,
                max_chunk_size=self.chunk_size,
            )
            logger.debug(f"Created ASTChunkBuilder for language: {language}")
            return self._ast_chunker_cache[language]
        else:
            return None

    def scrape_repository(
        self, repo_url: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Scrape a GitHub repository and extract all code files.

        Args:
            repo_url: URL of the GitHub repository to scrape.

        Returns:
            Tuple of (documents, metadata) where documents is a list of
            document dictionaries suitable for ingestion, and metadata
            contains repository information.

        Raises:
            RuntimeError: If scraping fails.
        """
        # Create temporary directory for cloning
        tmpdir = tempfile.mkdtemp(prefix="github_scraper_")
        repo_dir = pathlib.Path(tmpdir, "repo")

        try:
            logger.info(f"Cloning repository: {repo_url}")
            git_clone(repo_url, str(repo_dir))

            head_commit = git_head_commit(str(repo_dir))
            logger.info(f"Cloned successfully (HEAD: {head_commit[:8]})")

            logger.info("Scanning repository files...")
            file_infos = collect_files(repo_dir, self.max_file_size)

            # Filter to only included files
            included_files = [f for f in file_infos if f.decision.include]

            # Count statistics
            total_files = len(file_infos)
            skipped_binary = sum(1 for f in file_infos if f.decision.reason == "binary")
            skipped_large = sum(
                1 for f in file_infos if f.decision.reason == "too_large"
            )

            logger.info(
                f"Found {total_files} files: {len(included_files)} to include, "
                f"{skipped_binary} binary, {skipped_large} too large"
            )

            # Step 1: Extract all file contents and build file data list
            logger.info("Extracting file contents...")
            file_data_list: list[dict[str, Any]] = []
            skipped_empty = 0

            for file_info in included_files:
                try:
                    # Extract content based on file type
                    if is_jupyter_notebook(file_info.path):
                        # Use nbformat to extract notebook cells
                        content = extract_notebook_content(file_info.path)
                        logger.debug(f"Extracted Jupyter notebook: {file_info.rel}")
                    else:
                        # Read regular text file
                        content = read_text(file_info.path)

                    # Skip empty files
                    if not content or not content.strip():
                        skipped_empty += 1
                        logger.debug(f"Skipping empty file: {file_info.rel}")
                        continue

                    file_data_list.append(
                        {
                            "file_info": file_info,
                            "content": content,
                        }
                    )

                except (OSError, Exception) as e:
                    # Catch file read errors and notebook parsing errors
                    logger.warning(
                        f"Failed to process {file_info.rel}: {type(e).__name__}: {e}"
                    )
                    continue

            # Step 2: Group files by language/splitter type
            logger.info(f"Grouping {len(file_data_list)} files by language/type...")
            groups: dict[str | None, list[dict[str, Any]]] = {}

            for file_data in file_data_list:
                file_info = file_data["file_info"]
                language = detect_language_from_extension(file_info.rel)

                if language not in groups:
                    groups[language] = []
                groups[language].append(file_data)

            logger.info(
                f"Found {len(groups)} language/type groups: "
                f"{', '.join(str(k) if k else 'semantic' for k in groups.keys())}"
            )

            # Step 3: Concatenate files within each group and split
            all_chunks: list[str] = []
            files_in_chunks: list[list[str]] = []

            for language, group_files in groups.items():
                logger.info(
                    f"Processing {len(group_files)} files for "
                    f"{'semantic' if language is None else language}"
                )

                # Concatenate all files in this group with separators
                concatenated_parts = []
                file_paths = []

                for file_data in group_files:
                    file_info = file_data["file_info"]
                    content = file_data["content"]

                    # Add file header as separator
                    concatenated_parts.append(content)
                    file_paths.append(file_info.rel)

                concatenated_content: str = "\n".join(concatenated_parts)

                if (
                    language is not None
                    and language in self.ast_chunker_supported_languages
                ):
                    # Use ast chunker based on: https://arxiv.org/abs/2506.15655 / https://github.com/yilinjz/astchunk
                    splitter: ASTChunkBuilder = self._get_ast_chunker(language)
                    chunks = splitter.chunkify(concatenated_content)
                    chunks = [c.get("content") for c in chunks]
                elif (
                    language is not None
                    and language not in self.ast_chunker_supported_languages
                ):
                    # Use language-specific code splitter
                    splitter: CodeSplitter = self._get_code_splitter(language)
                    chunks = splitter.split_text(concatenated_content)
                else:
                    # Use semantic chunker for non-code files
                    chunks = self._semantic_chunker(
                        concatenated_content, overlap=self.chunk_overlap
                    )

                logger.info(
                    f"Generated {len(chunks)} chunks from "
                    f"{'semantic' if language is None else language} group"
                )

                # Track which files contributed to these chunks
                for chunk in chunks:
                    all_chunks.append(chunk)
                    files_in_chunks.append(file_paths)

            # Step 4: Remove duplicate chunks while preserving order
            logger.info(f"Removing duplicates from {len(all_chunks)} total chunks...")
            seen = set()
            unique_chunks = []
            unique_files = []

            for chunk, files in zip(all_chunks, files_in_chunks, strict=False):
                if chunk not in seen:
                    seen.add(chunk)
                    unique_chunks.append(chunk)
                    unique_files.append(files)

            logger.info(
                f"Removed {len(all_chunks) - len(unique_chunks)} duplicate chunks, "
                f"{len(unique_chunks)} unique chunks remain"
            )

            # Step 5: Create documents with metadata
            documents: list[dict[str, Any]] = []

            for chunk_idx, (chunk, file_list) in enumerate(
                zip(unique_chunks, unique_files, strict=False)
            ):
                documents.append(
                    {
                        "content": chunk,
                        "meta": {
                            "source": "github",
                            "repo_url": repo_url,
                            "file_paths": file_list,  # List of contributing files
                            "commit": head_commit,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(unique_chunks),
                        },
                    }
                )

            # Create metadata summary
            metadata = {
                "repo_url": repo_url,
                "commit": head_commit,
                "total_files": total_files,
                "processed_files": len(included_files) - skipped_empty,
                "total_chunks": len(documents),
                "skipped_binary": skipped_binary,
                "skipped_large": skipped_large,
                "skipped_empty": skipped_empty,
            }

            logger.info(
                f"Successfully extracted {len(documents)} chunks from "
                f"{len(included_files) - skipped_empty} files "
                f"(skipped {skipped_empty} empty files)"
            )

            return documents, metadata

        finally:
            # Clean up temporary directory
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary directory: {tmpdir}")
