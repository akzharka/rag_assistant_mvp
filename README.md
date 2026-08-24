# RAG Assistant

Simple MVP for uploading PDF contracts, extracting structured information, and asking questions using a local RAG pipeline.

## Stack

* Python
* FastAPI
* Hugging Face Transformers
* Qwen2.5-1.5B-Instruct
* LangChain
* Sentence Transformers
* ChromaDB
* PyPDF
* PyTorch

## Features

* Upload and process PDF contracts
* Extract structured contract information with a local LLM
* Split documents into chunks
* Generate embeddings locally
* Store embeddings in ChromaDB
* Ask questions about uploaded documents
* Filter retrieval by `document_id`
* Return answers with source filename and page
* Maintain basic conversation history

## Project Structure

```text
app/
├── api/
│   ├── documents.py
│   └── chat.py
├── models/
│   └── schemas.py
├── services/
│   ├── pdf.py
│   ├── llm.py
│   ├── rag.py
│   └── registry.py
└── main.py

sample_documents/
requirements.txt
README.md
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8001
```

Open Swagger:

```text
http://127.0.0.1:8001/docs
```

## Usage

### Process a document

Use:

```text
POST /documents/process
```

Upload a PDF.

The response returns a `document_id` and extracted contract information.

### List processed documents

```text
GET /documents
```

### Ask a question

```text
POST /chat
```

Example:

```json
{
  "document_id": "your-document-id",
  "question": "What is the termination notice period?"
}
```

Example response:

```json
{
  "question": "What is the termination notice period?",
  "answer": "Either party may terminate the agreement with 60 calendar days' written notice.",
  "sources": [
    {
      "filename": "service_agreement.pdf",
      "page": 2
    }
  ]
}
```

## Notes

The LLM and embedding models run locally.

The current MVP supports text-based PDFs only. Conversation history is stored in memory, and document metadata/vector storage is local.

Sample synthetic contracts are included for testing.
