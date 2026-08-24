from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.chat import router as chat_router


app = FastAPI(
    title="Contract Intelligence Assistant",
    description=(
        "Local LLM-powered document processing "
        "and RAG assistant."
    ),
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message": "Contract Intelligence Assistant",
        "status": "running"
    }


app.include_router(
    documents_router
)

app.include_router(
    chat_router
)









