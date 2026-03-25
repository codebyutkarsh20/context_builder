import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from project root (one level up from backend/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()  # Also check backend/.env

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
for _lib in ("httpcore", "httpx", "chromadb", "urllib3", "openai._base_client"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

from fastapi.middleware.cors import CORSMiddleware

from api.repos import router as repos_router
from api.graph import router as graph_router
from api.context import router as context_router
from api.search import router as search_router
from api.chat import router as chat_router
from api.agent import router as agent_router
from api.knowledge import router as knowledge_router
from api.metrics import router as metrics_router
from api.flags import router as flags_router
from api.eval import router as eval_router
from api.tools import router as tools_router
from graph.neo4j_client import neo4j_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        neo4j_client.connect()
        neo4j_client.ensure_constraints()
    except Exception as e:
        import logging
        logging.getLogger("main").warning("Neo4j unavailable at startup (running without graph DB): %s", e)
    yield
    neo4j_client.close()


app = FastAPI(
    title="Context Builder API",
    description="Build and query knowledge graphs for code repositories",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(repos_router, prefix="/api")
app.include_router(graph_router, prefix="/api")
app.include_router(context_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(knowledge_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(flags_router, prefix="/api")
app.include_router(eval_router, prefix="/api")
app.include_router(tools_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "neo4j": neo4j_client.is_connected()}
