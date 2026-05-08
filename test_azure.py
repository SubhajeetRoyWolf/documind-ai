import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)

# Quick sanity check
print("ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))

# ── Test 1: Chat ──────────────────────────────────────────────
print("\n" + "="*50)
print("TEST 1: GPT-4o Chat")
print("="*50)
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Say: I am GPT-4o running on Azure OpenAI!"}
    ],
    temperature=0.1
)
print(response.choices[0].message.content)

# ── Test 2: Embeddings ────────────────────────────────────────
print("\n" + "="*50)
print("TEST 2: text-embedding-3-small")
print("="*50)
embed_response = client.embeddings.create(
    model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    input="DocuMind AI is an agentic RAG system."
)
vector = embed_response.data[0].embedding
print(f"✅ Embedding dimensions: {len(vector)}")
print(f"✅ Sample values: {vector[:3]}")

print("\n" + "="*50)
print("ALL TESTS PASSED — Azure OpenAI is live! 🚀")
print("="*50)