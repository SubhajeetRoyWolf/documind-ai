import os
import cohere
from openai import AzureOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ── Clients ───────────────────────────────────────────────────
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)

pc        = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index     = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
co        = cohere.Client(os.getenv("COHERE_API_KEY"))


# ── Stage 1: Embed the query ──────────────────────────────────
def embed_query(query: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        input=query
    )
    return response.data[0].embedding


# ── Stage 2: Vector search in Pinecone ───────────────────────
def vector_search(query_vector: list[float], top_k: int = 10) -> list[dict]:
    """
    Returns top_k most similar chunks from Pinecone.
    We fetch more than we need (10) so the reranker has
    enough candidates to pick the best from.
    """
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    chunks = []
    for match in results.matches:
        chunks.append({
            "id":     match.id,
            "score":  match.score,
            "text":   match.metadata.get("text", ""),
            "source": match.metadata.get("source", ""),
            "page":   match.metadata.get("page", 0)
        })

    return chunks


# ── Stage 3: Rerank with Cohere ───────────────────────────────
def rerank_chunks(query: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
    """
    Cohere reranker scores each chunk specifically for
    relevance to the query — more precise than vector similarity.
    Returns top_n best chunks.
    """
    if not chunks:
        return []

    # Cohere needs plain text documents
    documents = [c["text"] for c in chunks]

    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-english-v3.0"
    )

    # Map reranked results back to our chunk dicts
    reranked = []
    for result in response.results:
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)

    return reranked


# ── Stage 4: Build grounded prompt ───────────────────────────
def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Constructs a prompt that forces GPT-4o to answer
    ONLY from the retrieved context — no hallucination.
    """
    # Format context with source citations
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1} | {chunk['source']} | Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are DocuMind AI — a precise document assistant.

RULES:
1. Answer using ONLY the context provided below
2. If the answer is not in the context, say "I don't have enough information in the document to answer this."
3. Always cite your sources using [Source X] notation
4. Never make up or infer information not present in the context
5. Be concise and direct"""

    user_prompt = f"""Context:
{context}

---

Question: {query}

Answer (with source citations):"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]


# ── Stage 5: Generate answer ──────────────────────────────────
def generate_answer(messages: list[dict]) -> str:
    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=messages,
        temperature=0.1,        # low temp = more factual, less creative
        max_tokens=500
    )
    return response.choices[0].message.content


# ── Full RAG pipeline ─────────────────────────────────────────
def ask(query: str, verbose: bool = True) -> dict:
    """
    Full pipeline: question → embed → search → rerank → generate
    Returns answer + sources.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"{'='*50}")

    # Step 1: Embed query
    query_vector = embed_query(query)
    if verbose:
        print(f"✅ Query embedded")

    # Step 2: Vector search
    chunks = vector_search(query_vector, top_k=10)
    if verbose:
        print(f"✅ Retrieved {len(chunks)} chunks from Pinecone")
        for c in chunks:
            print(f"   → [{c['score']:.3f}] {c['text'][:60]}...")

    # Step 3: Rerank
    reranked = rerank_chunks(query, chunks, top_n=3)
    if verbose:
        print(f"\n✅ Reranked → top 3 chunks:")
        for c in reranked:
            print(f"   → [{c['rerank_score']:.3f}] {c['text'][:60]}...")

    # Step 4 + 5: Build prompt and generate
    messages = build_prompt(query, reranked)
    answer   = generate_answer(messages)

    if verbose:
        print(f"\n{'='*50}")
        print(f"ANSWER:")
        print(f"{'='*50}")
        print(answer)
        print(f"{'='*50}\n")

    return {
        "query":   query,
        "answer":  answer,
        "sources": [
            {
                "text":   c["text"][:100],
                "source": c["source"],
                "page":   c["page"],
                "score":  c["rerank_score"]
            }
            for c in reranked
        ]
    }


# ── Test it ───────────────────────────────────────────────────
if __name__ == "__main__":

    test_questions = [
        "What are this candidate's AI and GenAI skills?",
        "What projects has the candidate built?",
        "How many years of experience does the candidate have?",
        "What is the candidate's educational background?",
    ]

    for question in test_questions:
        result = ask(question)
        input("\nPress Enter for next question...")