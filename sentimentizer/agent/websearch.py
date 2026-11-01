"""Web search tool using Ollama's web search API.

Provides a typed, secure Python interface for searching the web via
Ollama's API. Can be used standalone or as a Pydantic AI agent tool.

Security safeguards:
- API key is read from the environment, never passed as a parameter
- Error messages are sanitized to never include the API key value
- Query length is validated to prevent data leakage via long queries
- Content is truncated to prevent context window blowup
- Potential prompt-injection patterns are stripped from content
- Results are capped per call and per agent run (rate limiting)
- Requests timeout to prevent hanging

Usage::

    from sentimentizer.agent.websearch import web_search, WebSearchResult

    results = web_search("best learning rate for RNN")
    for r in results:
        print(r.title, r.url)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL
from sentimentizer.safety import INJECTION_PATTERNS

logger = new_logger(DEFAULT_LOG_LEVEL)

# ---------------------------------------------------------------------------
# Constants & security defaults
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://ollama.com/api/web_search"
DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_QUERY_LENGTH = 200
DEFAULT_MAX_CONTENT_LENGTH = 2000
MAX_TITLE_LENGTH = 300
MAX_URL_LENGTH = 2048
MAX_RESULTS_LIMIT = 10
DEFAULT_TIMEOUT_SECONDS = 15
MAX_CALLS_PER_RUN = 3

# Patterns that look like secrets/API keys in queries
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|credential)\s*[=:]\s*\S+"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style keys
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub PATs
    re.compile(r"glpat-[a-zA-Z0-9\-]{20,}"),  # GitLab PATs
]

# Call counter for rate limiting per agent run
_call_counter: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class WebSearchResult:
    """A single web search result.

    Attributes:
        title: Page title.
        url: Page URL (returned as text only — never fetched or followed).
        content: Snippet of page content (truncated and sanitized).
    """

    title: str
    url: str
    content: str


# ---------------------------------------------------------------------------
# Input validation & sanitization
# ---------------------------------------------------------------------------


def validate_query(query: str, max_length: int = DEFAULT_MAX_QUERY_LENGTH) -> str:
    """Validate and sanitize a search query.

    Args:
        query: The search query string.
        max_length: Maximum allowed query length.

    Returns:
        The stripped query string.

    Raises:
        ValueError: If the query is empty, too long, or contains secrets.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty")

    query = query.strip()

    if len(query) > max_length:
        raise ValueError(
            f"Query length ({len(query)}) exceeds maximum ({max_length}). "
            f"Shorten the query to prevent data leakage."
        )

    # Check for secret-like patterns
    for pattern in _SECRET_PATTERNS:
        match = pattern.search(query)
        if match:
            raise ValueError(
                "Query contains what appears to be a secret or credential. "
                "Remove sensitive values before searching."
            )

    return query


def sanitize_content(content: str, max_length: int = DEFAULT_MAX_CONTENT_LENGTH) -> str:
    """Truncate and sanitize web content to prevent prompt injection.

    Strips common prompt-injection patterns and truncates to max_length.

    Args:
        content: Raw web content string.
        max_length: Maximum content length in characters.

    Returns:
        Sanitized and truncated content string.
    """
    if not content:
        return ""

    # Truncate
    if len(content) > max_length:
        content = content[:max_length] + "...[truncated]"

    # Strip potential prompt-injection patterns
    for pattern in INJECTION_PATTERNS:
        content = pattern.sub("[filtered]", content)

    return content


def _sanitize_error(message: str) -> str:
    """Remove any accidental API key leakage from error messages.

    Args:
        message: Original error message.

    Returns:
        Sanitized error message with key values removed.
    """
    # Remove anything that looks like a Bearer token value
    sanitized = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", message)

    # Remove anything that looks like an API key (long alphanumeric strings
    # after common key-related keywords)
    sanitized = re.sub(
        r"(api[_-]?key|token|secret|password|credential)\s*[=:]\s*\S+",
        r"\1=[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def _check_rate_limit(run_id: str = "default", max_calls: int = MAX_CALLS_PER_RUN) -> None:
    """Enforce per-run rate limit on web search calls.

    Args:
        run_id: Identifier for the current agent run.
        max_calls: Maximum allowed calls per run.

    Raises:
        RuntimeError: If the rate limit has been exceeded.
    """
    current = _call_counter.get(run_id, 0)
    if current >= max_calls:
        raise RuntimeError(
            f"Web search rate limit exceeded ({max_calls} calls per run). "
            f"Use the results you already have."
        )
    _call_counter[run_id] = current + 1


def reset_rate_limit(run_id: str = "default") -> None:
    """Reset the call counter for a given run.

    Call this at the start of each agent run to reset the rate limiter.

    Args:
        run_id: Identifier for the agent run to reset.
    """
    _call_counter[run_id] = 0


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------


def web_search(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_query_length: int = DEFAULT_MAX_QUERY_LENGTH,
    max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    api_url: str = DEFAULT_API_URL,
    run_id: str = "default",
) -> list[WebSearchResult]:
    """Search the web using Ollama's web search API.

    Args:
        query: Search query string (validated for length and secrets).
        max_results: Maximum number of results to return (1-10).
        max_query_length: Maximum query length to prevent data leakage.
        max_content_length: Maximum content length per result.
        timeout: Request timeout in seconds.
        api_url: Ollama web search API endpoint.
        run_id: Identifier for rate limiting per agent run.

    Returns:
        List of ``WebSearchResult`` objects with title, url, and content.

    Raises:
        ValueError: If the query is invalid or contains secrets, or the API
            response does not have the expected shape.
        RuntimeError: If the rate limit is exceeded.
        OSError: If ``OLLAMA_API_KEY`` is not set.
        urllib.error.URLError: If the API request fails.
    """
    if not 1 <= max_results <= MAX_RESULTS_LIMIT:
        raise ValueError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}")

    # Validate query
    validated_query = validate_query(query, max_length=max_query_length)

    # Check rate limit
    _check_rate_limit(run_id=run_id)

    # Read API key from environment
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not api_key:
        raise OSError(
            "OLLAMA_API_KEY is not set. Set it via: export OLLAMA_API_KEY=your_key "
            "or add it to your .env file."
        )

    # Build request
    payload = json.dumps({"query": validated_query}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    logger.info("web_search_request", query=validated_query, max_results=max_results)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        sanitized = _sanitize_error(str(e))
        logger.warning("web_search_http_error", error=sanitized)
        raise urllib.error.HTTPError(
            url=api_url,
            code=e.code,
            msg=f"Web search request failed: {sanitized}",
            hdrs=e.hdrs,
            fp=e.fp,
        ) from e
    except urllib.error.URLError as e:
        sanitized = _sanitize_error(str(e))
        logger.warning("web_search_url_error", error=sanitized)
        raise urllib.error.URLError(f"Web search request failed: {sanitized}") from e
    except TimeoutError as e:
        logger.warning("web_search_timeout", timeout=timeout)
        raise TimeoutError(f"Web search request timed out after {timeout}s") from e

    # Parse response
    try:
        data = json.loads(response_data)
    except json.JSONDecodeError as e:
        logger.warning("web_search_parse_error", error=str(e))
        raise ValueError(f"Failed to parse web search response: {e}") from e

    # Extract results
    raw_results = data.get("results", []) if isinstance(data, dict) else None
    if not isinstance(raw_results, list) or not all(isinstance(r, dict) for r in raw_results):
        logger.warning("web_search_unexpected_response")
        raise ValueError("Web search response did not contain a list of results")
    if not raw_results:
        logger.info("web_search_no_results", query=validated_query)
        return []

    # Every field reaches the LLM, so all of them are untrusted.
    results: list[WebSearchResult] = []
    for item in raw_results[:max_results]:
        results.append(
            WebSearchResult(
                title=sanitize_content(str(item.get("title") or ""), max_length=MAX_TITLE_LENGTH),
                url=sanitize_content(str(item.get("url") or ""), max_length=MAX_URL_LENGTH),
                content=sanitize_content(
                    str(item.get("content") or ""), max_length=max_content_length
                ),
            )
        )

    logger.info(
        "web_search_success",
        query=validated_query,
        result_count=len(results),
    )

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

FormatOption = Literal["text", "json", "markdown"]


def format_results(results: list[WebSearchResult], fmt: FormatOption = "text") -> str:
    """Format search results for display.

    Args:
        results: List of search results.
        fmt: Output format — ``text`` (default), ``json``, or ``markdown``.

    Returns:
        Formatted string ready to print.
    """
    if fmt == "json":
        return _format_json(results)
    elif fmt == "markdown":
        return _format_markdown(results)
    else:
        return _format_text(results)


def _format_text(results: list[WebSearchResult]) -> str:
    """Format results as plain text."""
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.url}")
        lines.append(f"   {r.content}")
        if i < len(results):
            lines.append("")
    return "\n".join(lines)


def _format_json(results: list[WebSearchResult]) -> str:
    """Format results as a JSON array."""
    return json.dumps(
        [{"title": r.title, "url": r.url, "content": r.content} for r in results],
        indent=2,
        ensure_ascii=False,
    )


def _format_markdown(results: list[WebSearchResult]) -> str:
    """Format results as Markdown."""
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append("")
        lines.append(r.content)
        if i < len(results):
            lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DOTENV_PATH = pathlib.Path.cwd() / ".env"


def _load_dotenv(path: pathlib.Path = _DOTENV_PATH) -> None:
    """Load KEY=VALUE pairs from a .env file into ``os.environ``.

    Only sets variables that are not already set. Skips blank lines
    and comments (lines starting with ``#``). Does not require
    ``python-dotenv`` — uses stdlib only.

    Args:
        path: Path to the .env file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Remove surrounding quotes if present
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _suppress_structlog() -> None:
    """Raise structlog's log level to CRITICAL to suppress JSON output.

    This is only called in CLI mode so that structured log lines don't
    mix with formatted results. Agent runs are in separate processes
    and are not affected.
    """
    import structlog

    structlog.configure(
        cache_logger_on_first_use=True,
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="websearch",
        description="Search the web using Ollama's web search API.",
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        dest="format",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Maximum results to return, 1-10 (default: {DEFAULT_MAX_RESULTS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--snippet-length",
        type=int,
        default=DEFAULT_MAX_CONTENT_LENGTH,
        help=f"Max content length per result (default: {DEFAULT_MAX_CONTENT_LENGTH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable structured JSON logging (suppressed by default in CLI)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for web search.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 = success, 1 = user error, 2 = rate limit, 3 = network error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Auto-load .env if present (before checking API key)
    _load_dotenv()

    # Suppress structured JSON logging unless --verbose
    if not args.verbose:
        _suppress_structlog()

    # Auto-reset rate limit for standalone CLI invocations
    reset_rate_limit()

    # Validate max_results range
    if not 1 <= args.max_results <= MAX_RESULTS_LIMIT:
        print(f"Error: --max-results must be between 1 and {MAX_RESULTS_LIMIT}.", file=sys.stderr)
        return 1

    try:
        results = web_search(
            args.query,
            max_results=args.max_results,
            max_content_length=args.snippet_length,
            timeout=args.timeout,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        # Must be before OSError — URLError is a subclass of OSError
        print(f"Error: {_sanitize_error(str(e))}", file=sys.stderr)
        return 3
    except TimeoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    output = format_results(results, fmt=args.format)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
