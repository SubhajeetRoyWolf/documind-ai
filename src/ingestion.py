import os
import hashlib
import pymupdf                          
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm                   # progress bar

load_dotenv()

# ── Clients ───────────────────────────────────────────────────
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)

pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


# ── Stage 1: Extract text from PDF ───────────────────────────
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Returns a list of {page_num, text} dicts — one per page.
    """
    doc   = pymupdf.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if text:                         # skip blank pages
            pages.append({
                "page_num": page_num + 1,
                "text":     text
            })

    print(f"✅ Extracted {len(pages)} pages from '{os.path.basename(pdf_path)}'")
    return pages


# ── Stage 2: Chunk the text ───────────────────────────────────
def chunk_pages(pages: list[dict], chunk_size=500, chunk_overlap=50) -> list[dict]:
    """
    Splits each page's text into overlapping chunks.
    Preserves page number in metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # tries paragraph → sentence → word
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])

        for i, split in enumerate(splits):
            chunks.append({
                "page_num":   page["page_num"],
                "chunk_index": i,
                "text":        split
            })

    print(f"✅ Created {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── Stage 3: Embed chunks ─────────────────────────────────────
def embed_chunks(chunks: list[dict], batch_size=50) -> list[dict]:
    """
    Adds an 'embedding' key to each chunk dict.
    Sends in batches to avoid rate limits.
    """
    texts = [c["text"] for c in chunks]
    all_embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        response = openai_client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    # Attach embeddings back to chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    print(f"✅ Embedded {len(chunks)} chunks")
    return chunks


# ── Stage 4: Upsert to Pinecone ───────────────────────────────
def upsert_to_pinecone(chunks: list[dict], source_filename: str, batch_size=100):
    """
    Uploads all chunk vectors to Pinecone with metadata.
    Uses a hash of text as a stable unique ID.
    """
    vectors = []

    for chunk in chunks:
        # Stable unique ID: hash of source + page + chunk index
        chunk_id = hashlib.md5(
            f"{source_filename}_p{chunk['page_num']}_c{chunk['chunk_index']}".encode()
        ).hexdigest()

        vectors.append({
            "id":     chunk_id,
            "values": chunk["embedding"],
            "metadata": {
                "text":     chunk["text"],
                "source":   source_filename,
                "page":     chunk["page_num"],
                "chunk_idx": chunk["chunk_index"]
            }
        })

    # Upsert in batches (Pinecone max = 100 per call)
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting"):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    print(f"✅ Upserted {len(vectors)} vectors to Pinecone")


# ── Main pipeline ─────────────────────────────────────────────
def ingest_pdf(pdf_path: str):
    """
    Full pipeline: PDF → extract → chunk → embed → upsert
    """
    filename = os.path.basename(pdf_path)
    print(f"\n{'='*50}")
    print(f"INGESTING: {filename}")
    print(f"{'='*50}")

    pages  = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)
    chunks = embed_chunks(chunks)
    upsert_to_pinecone(chunks, filename)

    # Confirm in Pinecone
    stats = index.describe_index_stats()
    print(f"\n✅ INGESTION COMPLETE!")
    print(f"✅ Total vectors in Pinecone: {stats.total_vector_count}")
    print(f"{'='*50}\n")


# ── Run it ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/ingestion.py data/your_file.pdf")
        print("\nRunning with sample test instead...\n")

        # Quick self-test without a PDF
        test_chunks = [
            {"page_num": 1, "chunk_index": 0,
             "text": "RAG stands for Retrieval Augmented Generation."},
            {"page_num": 1, "chunk_index": 1,
             "text": "Pinecone is a managed vector database for production AI."},
            {"page_num": 2, "chunk_index": 0,
             "text": "Azure OpenAI provides GPT-4o for enterprise applications."},
        ]
        test_chunks = embed_chunks(test_chunks)
        upsert_to_pinecone(test_chunks, "self_test.txt")
        stats = index.describe_index_stats()
        print(f"✅ Self-test done. Vectors in index: {stats.total_vector_count}")

    else:
        ingest_pdf(sys.argv[1])
        

def ingest_text_directly(text: str, source: str = "manual"):
    """
    Ingest a raw text string directly into Pinecone.
    Useful for adding specific facts that chunking might miss.
    """
    from openai import AzureOpenAI
    import hashlib

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2025-01-01-preview"
    )

    embed_response = client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        input=text
    )
    vector = embed_response.data[0].embedding

    chunk_id = hashlib.md5(text.encode()).hexdigest()

    index.upsert(vectors=[{
        "id":     chunk_id,
        "values": vector,
        "metadata": {
            "text":      text,
            "source":    source,
            "page":      1,
            "chunk_idx": 0
        }
    }])
    print(f"✅ Directly ingested: '{text[:60]}...'")