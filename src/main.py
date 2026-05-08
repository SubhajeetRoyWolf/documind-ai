import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our pipeline modules
from ingestion import ingest_pdf
from retrieval  import ask

load_dotenv()

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind AI",
    description="Agentic RAG document intelligence API",
    version="1.0.0"
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k:    int = 3           # how many sources to return

class SourceModel(BaseModel):
    text:   str
    source: str
    page:   int
    score:  float

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[SourceModel]

class IngestResponse(BaseModel):
    filename:     str
    status:       str
    message:      str

class HealthResponse(BaseModel):
    status:  str
    service: str
    version: str


# ── Routes ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Standard production health check endpoint.
    Load balancers and monitoring tools ping this.
    """
    return {
        "status":  "healthy",
        "service": "DocuMind AI",
        "version": "1.0.0"
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF → triggers full ingestion pipeline:
    extract → chunk → embed → upsert to Pinecone
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run the full ingestion pipeline
        ingest_pdf(tmp_path)

        return {
            "filename": file.filename,
            "status":   "success",
            "message":  f"'{file.filename}' ingested successfully."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

    finally:
        # Always clean up temp file
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Ask a question → returns grounded answer + cited sources.
    GPT-4o answers ONLY from retrieved document context.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = ask(request.question, verbose=False)

        return {
            "question": result["query"],
            "answer":   result["answer"],
            "sources":  result["sources"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


# ── Dev server entry point ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True          # auto-restart on code changes
    )