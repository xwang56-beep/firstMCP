import sys
from pathlib import Path

try:
    # Prefer the MCP CLI-compatible FastMCP when running under `mcp dev`.
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - fallback when CLI extras aren't installed
    from fastmcp import FastMCP  # type: ignore[no-redef]


server = FastMCP(
    name="example-server",
    instructions="A minimal FastMCP demo server with a couple of tools.",
)

NDC_RAG_ROOT = Path(__file__).resolve().parent.parent / "NDC-RAG"
if str(NDC_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(NDC_RAG_ROOT))

try:  # pragma: no cover - runtime dependency
    from src.config import load_config
    from src.main import query_knowledge_base
except ModuleNotFoundError as exc:  # pragma: no cover - helpful error
    raise ModuleNotFoundError(
        "Unable to import NDC-RAG modules. Ensure `NDC-RAG/src` is importable."
    ) from exc


@server.tool(description="Return a friendly greeting for the provided name.")
def greet(name: str) -> str:
    """Generate a greeting."""
    return f"Hello, {name}!"


@server.tool(description="Add two numbers together and return the sum.")
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@server.tool(
    description=(
        "Answer knowledge-base questions by running the NDC-RAG pipeline "
        "and returning the top matching document snippets."
    )
)
def query_rag(query: str, top_k: int = 3) -> str:
    """
    Execute the NDC-RAG query workflow and return its output.

    Args:
        query: Natural language question to ask the retriever.
        top_k: Number of results to return (defaults to 3).
    """

    config = load_config()
    response = query_knowledge_base(query, config=config, top_k=top_k)

    results = response.get("results") or []
    evaluation = response.get("evaluation")

    lines = []
    for entry in results:
        metadata = getattr(entry, "metadata", None)
        score = getattr(entry, "score", "NA")
        lines.append(f"Score: {score:0.4f}")
        if metadata is not None:
            lines.append(f"Source: {metadata}")
            if isinstance(metadata, dict):
                content = (metadata.get("content") or "").strip()
                if content:
                    lines.append("Content:")
                    lines.append(content)
        lines.append("-" * 40)

    if evaluation is not None:
        precision = getattr(evaluation, "precision_at_k", None)
        if precision is not None:
            lines.append(f"Precision@k: {precision:.2f}")

    return "\n".join(lines) if results else "No results returned."


if __name__ == "__main__":
    # Run over stdio by default so that MCP-compatible clients can connect.
    # To expose an HTTP endpoint instead, pass transport parameters, for example:
    # server.run(transport="http", host="127.0.0.1", port=8000)
    server.run()

