"""FastAPI implementation for document analysis and question answering."""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import List
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.document_service import DocumentService
from services.qa_service import QAService
from config.constants import DEFAULT_TOP_K


vector_stores = {}
retrievers = {}
qa_chains = {}


class QueryRequest(BaseModel):
    """Request model for querying documents."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Question to ask about the documents")


class QueryResponse(BaseModel):
    """Response model for query results."""

    answer: str = Field(..., description="Answer to the question")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class ProcessResponse(BaseModel):
    """Response model for document processing status."""

    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    temp_dir = tempfile.mkdtemp()
    app.state.temp_dir = temp_dir
    yield
    for file in os.listdir(temp_dir):
        try:
            os.remove(os.path.join(temp_dir, file))
        except Exception:
            pass
    os.rmdir(temp_dir)


app = FastAPI(
    title="Document Analysis API",
    description="API for document analysis and question answering",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_document_service():
    """Dependency for document service."""
    return DocumentService()


def get_qa_service():
    """Dependency for QA service."""
    return QAService()


async def process_documents_task(
    session_id: str,
    temp_files: List[str],
    top_k: int,
    document_service: DocumentService,
    qa_service: QAService,
):
    """Background task for processing documents.

    Args:
        session_id: Session identifier
        temp_files: List of temporary file paths
        top_k: Number of documents to retrieve
        document_service: Document service
        qa_service: QA service
    """
    try:
        vector_store = document_service.process_uploaded_paths(temp_files)

        if not vector_store:
            return

        retriever = document_service.create_retriever(vector_store, top_k)
        qa_chain = qa_service.build_qa_chain(retriever)

        vector_stores[session_id] = vector_store
        retrievers[session_id] = retriever
        qa_chains[session_id] = qa_chain
    except Exception as e:
        print(f"Error processing documents: {str(e)}")


@app.post("/documents/upload", response_model=ProcessResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    top_k: int = DEFAULT_TOP_K,
    document_service: DocumentService = Depends(get_document_service),
    qa_service: QAService = Depends(get_qa_service),
):
    """Upload and process documents.

    Args:
        background_tasks: Background tasks manager
        files: List of files to process
        top_k: Number of documents to retrieve (default: 3)
        document_service: Document service
        qa_service: QA service

    Returns:
        ProcessResponse: Processing status
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    session_id = str(uuid.uuid4())

    temp_files = []
    for file in files:
        try:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ["pdf", "csv", "txt"]:
                continue

            temp_file = os.path.join(
                app.state.temp_dir, f"{uuid.uuid4()}.{file_extension}"
            )
            with open(temp_file, "wb") as f:
                f.write(await file.read())
            temp_files.append(temp_file)
        except Exception as e:
            print(f"Error saving file {file.filename}: {str(e)}")

    if not temp_files:
        raise HTTPException(status_code=400, detail="No valid files provided")

    background_tasks.add_task(
        process_documents_task,
        session_id,
        temp_files,
        top_k,
        document_service,
        qa_service,
    )

    return ProcessResponse(
        session_id=session_id,
        status="processing",
        message="Documents are being processed",
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    """Query processed documents.

    Args:
        query_request: Query request

    Returns:
        QueryResponse: Query response
    """
    session_id = query_request.session_id

    if session_id not in qa_chains:
        raise HTTPException(
            status_code=404,
            detail="Session not found or documents not processed yet",
        )

    qa_chain = qa_chains[session_id]

    start_time = datetime.now()
    try:
        response = qa_chain.invoke({"input": query_request.query})
        answer = response["answer"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    return QueryResponse(
        answer=answer,
        processing_time_ms=int(processing_time),
    )


@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get session status.

    Args:
        session_id: Session identifier

    Returns:
        dict: Session status
    """
    if session_id in qa_chains:
        return {
            "status": "ready",
            "message": "Documents processed and ready for queries",
        }
    return {"status": "processing", "message": "Documents are still being processed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
