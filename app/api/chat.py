import uuid

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest
from app.services.rag import answer_with_rag


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


conversation_store = {}


@router.post("")
def chat(request: ChatRequest):

    try:
        # First message: create new conversation
        if request.conversation_id is None:

            conversation_id = str(uuid.uuid4())

            conversation_store[conversation_id] = []

        # Existing conversation
        else:
            conversation_id = request.conversation_id

            if conversation_id not in conversation_store:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found."
                )

        history = conversation_store[
            conversation_id
        ]

        answer, sources = answer_with_rag(
            document_id=request.document_id,
            question=request.question,
            conversation_history=history,
        )

        # Store user message
        history.append({
            "role": "user",
            "content": request.question,
        })

        # Store assistant response
        history.append({
            "role": "assistant",
            "content": answer,
        })

        return {
            "conversation_id": conversation_id,
            "document_id": request.document_id,
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(e)}"
        )