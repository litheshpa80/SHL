import os

from fastapi import FastAPI, HTTPException

from app.agent import run_turn
from app.retrieval import Catalog
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title="SHL Assessment Recommender")

CATALOG_PATH = os.environ.get("CATALOG_PATH", "data/catalog.json")
_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = Catalog.load(CATALOG_PATH)
    return _catalog


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
    try:
        catalog = get_catalog()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"catalog unavailable: {e}") from e
    try:
        return run_turn(req.messages, catalog)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {e}"
        ) from e


