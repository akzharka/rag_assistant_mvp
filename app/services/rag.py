from pathlib import Path
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
from app.services.llm import tokenizer, model


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    length_function=len
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

def chunk_documents(documents):

    chunks = text_splitter.split_documents(documents)

    return chunks


def index_documents(chunks):

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return len(chunks)

def create_langchain_documents(pages, filename, document_id):
    documents = []

    for page in pages:

        if not page["text"].strip():
            continue

        document = Document(
            page_content=page["text"],
            metadata={
                "document_id": document_id,
                "filename": filename,
                "page": page["page"]
            }
        )

        documents.append(document)

    return documents


vector_store = Chroma(
    collection_name="contracts",
    embedding_function=embeddings,
    persist_directory="./data/chroma"
)


def answer_with_rag(
    document_id: str,
    question: str,
    conversation_history: list
):

    results = vector_store.similarity_search(
        question,
        k=3,
        filter={
            "document_id": document_id
        }
    )

    context_parts = []

    for doc in results:

        filename = doc.metadata.get(
            "filename",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE: {filename}
PAGE: {page}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    history_text = ""

    for message in conversation_history[-6:]:

        role = message["role"]
        content = message["content"]

        history_text += (
            f"{role.upper()}: {content}\n"
        )

    prompt = f"""
        You are a contract analysis assistant.

        Answer using ONLY the retrieved contract context.

        Use the previous conversation only to understand
        references such as:

        "what about that?"
        "and the Provider?"
        "when does it expire?"

        Do not use conversation history as factual evidence.
        Contract facts must come from the retrieved context.

        If the answer cannot be found in the retrieved
        context, say:

        "The provided document does not contain enough
        information to answer this question."

        PREVIOUS CONVERSATION:

        {history_text}

        RETRIEVED CONTRACT CONTEXT:

        {context}

        CURRENT QUESTION:

        {question}

        ANSWER:
        """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a contract analysis assistant. "
                "Use retrieved contract evidence for factual answers."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=250,
            do_sample=False
        )

    generated_ids = generated_ids[
        :,
        model_inputs.input_ids.shape[1]:
    ]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    sources = []

    for doc in results:

        source = {
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page")
        }

        if source not in sources:
            sources.append(source)

    return answer.strip(), sources