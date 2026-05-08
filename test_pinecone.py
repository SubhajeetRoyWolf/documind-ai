import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pinecone import Pinecone

load_dotenv()

# ── Connect to Pinecone ───────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

print("="*50)
print("TEST 1: Pinecone connection")
print("="*50)
stats = index.describe_index_stats()
print(f"✅ Connected to Pinecone!")
print(f"✅ Index: {os.getenv('PINECONE_INDEX_NAME')}")
print(f"✅ Total vectors stored: {stats.total_vector_count}")

# ── Connect to Azure OpenAI ───────────────────────────────────
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)

# ── Generate a test embedding ─────────────────────────────────
print("\n" + "="*50)
print("TEST 2: Embed and upsert a test vector")
print("="*50)

sample_text = "Azure OpenAI provides enterprise-grade AI capabilities."

embed_response = client.embeddings.create(
    model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    input=sample_text
)
vector = embed_response.data[0].embedding
print(f"✅ Embedding created: {len(vector)} dimensions")

# ── Upsert into Pinecone ──────────────────────────────────────
index.upsert(vectors=[{
    "id":       "test_vector_001",
    "values":   vector,
    "metadata": {
        "text":   sample_text,
        "source": "test",
        "page":   1
    }
}])
print(f"✅ Vector upserted to Pinecone!")

# ── Query it back ─────────────────────────────────────────────
print("\n" + "="*50)
print("TEST 3: Semantic search")
print("="*50)

query = "What does Azure OpenAI offer for businesses?"
query_embed = client.embeddings.create(
    model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    input=query
).data[0].embedding

results = index.query(
    vector=query_embed,
    top_k=1,
    include_metadata=True
)

print(f"Query: '{query}'")
print(f"Best match: '{results.matches[0].metadata['text']}'")
print(f"Similarity score: {results.matches[0].score:.4f}")
print("\n" + "="*50)
print("ALL TESTS PASSED — Pinecone is live! 🚀")
print("="*50)