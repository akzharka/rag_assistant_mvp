import uuid

from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from app.services.pdf import extract_pdf
from app.services.llm import extract_contract_data
from app.services.rag import (
    create_langchain_documents,
    chunk_documents,
    index_documents,
)
from app.services.registry import (
    document_registry,
    save_document_registry,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/process")
async def process_document(
    file: UploadFile = File(...)
):
    pages = await extract_pdf(file)

    full_text = "\n\n".join(
        page["text"]
        for page in pages
    )

    if not full_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from PDF."
        )

    document_id = str(uuid.uuid4())

    try:
        # Extract structured contract information
        extracted_data = extract_contract_data(
            full_text
        )

        # Prepare LangChain documents
        documents = create_langchain_documents(
            pages=pages,
            filename=file.filename,
            document_id=document_id,
        )

        # Split into chunks
        chunks = chunk_documents(
            documents
        )

        # Create embeddings and store in Chroma
        index_documents(
            chunks
        )

        # Save document metadata
        document_registry[document_id] = {
            "document_id": document_id,
            "filename": file.filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "contract_type": extracted_data.get(
                "contract_type",
                "Unknown",
            ),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        save_document_registry(
            document_registry
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "extracted_data": extracted_data,
    }


@router.get("")
def get_documents():

    return {
        "count": len(document_registry),
        "documents": list(
            document_registry.values()
        ),
    }


@router.get("/{document_id}")
def get_document(
    document_id: str
):

    document = document_registry.get(
        document_id
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return document