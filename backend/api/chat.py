import logging
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.utils import validate_repo_name

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    repo: str
    question: str = Field(..., max_length=5000)
    history: list[ChatMessage] = []
    use_rag: bool = True


class ChatResponse(BaseModel):
    answer: str
    model: str
    usage: dict
    context_source: str = "flat_context"
    context_nodes: int = 0


# ─── System prompts ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a senior software engineer answering questions about a codebase.\n"
    "You have been given the COMPLETE context document for the repository.\n\n"
    "RULES:\n"
    "1. Answer ONLY from the provided context document. Do not use prior knowledge "
    "about libraries, frameworks, or general software patterns unless the context "
    "explicitly mentions them.\n"
    "2. If the context does not contain enough information to answer the question, say: "
    '"I don\'t have enough context to answer this. The context document does not '
    'cover [specific gap]."\n'
    "3. Always cite specific file paths, function names, class names, or business "
    "rules from the context.\n"
    "4. Be precise about business logic — quote the relevant section of the context "
    "when answering business rule questions.\n"
    "5. If the question is ambiguous, state your interpretation before answering.\n"
    "6. Use markdown formatting for readability."
)

_RAG_SYSTEM_PROMPT = (
    "You are a senior software engineer answering questions about a codebase.\n"
    "You have been given a TARGETED subset of the repository context — the most "
    "relevant files, functions, classes, and business rules for your question.\n\n"
    "RULES:\n"
    "1. Answer from the provided context. The context is selectively assembled, so it "
    "focuses on what's most relevant to the question.\n"
    "2. If the provided context doesn't cover something, say: "
    '"The retrieved context doesn\'t include [specific gap]. '
    'You may need to analyze more of the codebase."\n'
    "3. Always cite specific file paths, function names, class names, or business "
    "rules from the context.\n"
    "4. Pay special attention to decision points, business rules, and call relationships.\n"
    "5. If the question is ambiguous, state your interpretation before answering.\n"
    "6. Use markdown formatting for readability."
)


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    """Ask a question about a repo using Graph RAG context + Claude."""
    import anthropic

    repo = validate_repo_name(req.repo)
    data_dir = Path(os.environ.get("DATA_DIR", "/tmp/context_builder"))

    # Try Graph RAG first
    context = ""
    context_source = "flat_context"
    context_node_count = 0

    if req.use_rag:
        context, context_source, context_node_count = _try_rag(repo, data_dir, req.question)

    # Fallback to flat context.md
    if not context:
        context_path = data_dir / repo / "context.md"
        if not context_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Context not yet generated for '{repo}'. Run analysis first.",
            )
        try:
            context = context_path.read_text()
        except (IOError, OSError) as e:
            logger.error(f"Failed to read context for {repo}: {e}")
            raise HTTPException(status_code=500, detail="Failed to load repository context.")

        MAX_CONTEXT_CHARS = 100_000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated due to size limit]"
        context_source = "flat_context"

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    # Select system prompt based on context source
    system = _RAG_SYSTEM_PROMPT if context_source == "graph_rag" else _SYSTEM_PROMPT

    # Build messages
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "<repository_context>\n"
                f"{context}\n"
                "</repository_context>\n\n"
                "I will now ask you questions about this repository. "
                "Use ONLY the context above to answer."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "I've read the repository context. "
                "I'm ready to answer your questions using the information provided. "
                "Go ahead!"
            ),
        },
    ]

    # Append conversation history (capped at last 20 messages)
    history = req.history[-20:] if len(req.history) > 20 else req.history
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": req.question})

    # Call Claude
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=messages,
        )
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    except anthropic.APIError as e:
        logger.exception("Anthropic API error during chat")
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")

    text_blocks = [b.text for b in response.content if hasattr(b, 'text')]
    if not text_blocks:
        raise HTTPException(status_code=502, detail="Unexpected response format from LLM")
    answer = text_blocks[0]
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return ChatResponse(
        answer=answer,
        model=model,
        usage=usage,
        context_source=context_source,
        context_nodes=context_node_count,
    )


def _try_rag(repo: str, data_dir: Path, question: str) -> tuple[str, str, int]:
    """
    Attempt Graph RAG retrieval. Returns (context, source, node_count).
    Returns ("", "", 0) if RAG is unavailable.
    """
    try:
        from rag.retriever import GraphRAGRetriever
        from rag.context_assembler import ContextAssembler

        retriever = GraphRAGRetriever(repo, data_dir)
        result = retriever.retrieve(question, max_nodes=30)

        if not result.all_node_ids:
            return ("", "", 0)

        assembler = ContextAssembler(repo, data_dir)
        context = assembler.assemble(
            primary_ids=result.primary_nodes,
            expanded_ids=result.expanded_nodes,
            edges=result.edges,
            scores=result.scores,
            token_budget=15000,
        )

        if not context or len(context) < 100:
            return ("", "", 0)

        logger.info(
            "Graph RAG assembled context: %d nodes, ~%d tokens",
            len(result.all_node_ids),
            len(context) // 4,
        )
        return (context, "graph_rag", len(result.all_node_ids))

    except Exception as e:
        logger.debug("Graph RAG unavailable, falling back to flat context: %s", e)
        return ("", "", 0)
