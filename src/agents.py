import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from agent_state import AgentState
from retrieval import embed_query, vector_search, rerank_chunks

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)


# ── Helper: call GPT-4o ───────────────────────────────────────
def llm(system: str, user: str, temperature: float = 0.1) -> str:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


# ════════════════════════════════════════════════════════════════
# AGENT 1: PLANNER
# Decides what strategy to use based on the question
# ════════════════════════════════════════════════════════════════
def planner_agent(state: AgentState) -> AgentState:
    print(f"\n🧠 PLANNER thinking...")

    # Hardcoded rules — no LLM needed for simple routing
    question = state["question"].lower().strip()

    # Only clarify if question is too short or meaningless
    if len(question) < 5 or question in ["?", "help", "hi", "hello"]:
        plan = "clarify"
    elif any(word in question for word in ["summarise", "summarize", "summary", "overview"]):
        plan = "summarise"
    else:
        plan = "retrieve_and_answer"

    print(f"   → Plan: {plan}")
    return {**state, "plan": plan}


# ════════════════════════════════════════════════════════════════
# AGENT 2: RETRIEVER
# Fetches relevant context from Pinecone using the question
# ════════════════════════════════════════════════════════════════
def retriever_agent(state: AgentState) -> AgentState:
    """
    Embeds the question, searches Pinecone, reranks results.
    If critic rejected the previous answer, refines the query.
    """
    print(f"\n🔍 RETRIEVER fetching context...")

    # If critic gave feedback, use it to improve the query
    query = state["question"]
    if state["critic_feedback"]:
        query = f"{state['question']} {state['critic_feedback']}"
        print(f"   → Refined query with critic feedback")

    # Run retrieval pipeline from Week 1
    query_vector = embed_query(query)
    chunks       = vector_search(query_vector, top_k=10)
    reranked     = rerank_chunks(query, chunks, top_n=3)

    print(f"   → Retrieved {len(reranked)} chunks")
    for c in reranked:
        print(f"      [{c['rerank_score']:.3f}] {c['text'][:60]}...")

    return {**state, "context": reranked, "attempts": state["attempts"] + 1}


# ════════════════════════════════════════════════════════════════
# AGENT 3: GENERATOR
# Builds grounded answer from retrieved context
# ════════════════════════════════════════════════════════════════
def generator_agent(state: AgentState) -> AgentState:
    """
    Generates a grounded answer using ONLY the retrieved context.
    Includes source citations.
    """
    print(f"\n✍️  GENERATOR creating answer...")

    # Format context
    context_parts = []
    for i, chunk in enumerate(state["context"]):
        context_parts.append(
            f"[Source {i+1} | {chunk['source']} | Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Add chat history for multi-turn memory
    history = ""
    if state["chat_history"]:
        history = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in state["chat_history"][-4:]  # last 2 turns
        ])
        history = f"\nPrevious conversation:\n{history}\n"

    system = """You are DocuMind AI — a precise document assistant.
Answer using ONLY the provided context.
Always cite sources using [Source X] notation.
If the answer is not in the context say: "I don't have enough information."
Never hallucinate or make up facts."""

    user = f"""{history}
Context:
{context}

Question: {state['question']}

Answer:"""

    answer = llm(system, user, temperature=0.1)
    print(f"   → Answer generated ({len(answer)} chars)")

    # Build sources list
    sources = [
        {
            "text":   c["text"][:100],
            "source": c["source"],
            "page":   c["page"],
            "score":  c["rerank_score"]
        }
        for c in state["context"]
    ]

    return {**state, "answer": answer, "sources": sources}


# ════════════════════════════════════════════════════════════════
# AGENT 4: CRITIC
# Quality-checks the answer — approves or rejects
# ════════════════════════════════════════════════════════════════
def critic_agent(state: AgentState) -> AgentState:
    """
    Evaluates the generated answer against the context.
    Checks for: hallucination, completeness, relevance.
    Approves or rejects with specific feedback.
    """
    print(f"\n🔎 CRITIC evaluating answer...")

    context_text = "\n".join([c["text"] for c in state["context"]])

    # Inside planner_agent function, replace the system variable:

    system = """You are a quality checker. Be LENIENT.

APPROVE the answer if it:
- Answers the question using the context provided
- Says "I don't have enough information" when context is missing
- Contains source citations like [Source 1]

REJECT ONLY if the answer:
- Completely ignores the question
- Invents specific facts with no basis in context

Respond EXACTLY like this:
VERDICT: APPROVED
REASON: one sentence"""

    user = f"""Question: {state['question']}

Context:
{context_text}

Answer to evaluate:
{state['answer']}"""

    evaluation = llm(system, user, temperature=0.0)

    # Parse the verdict
    approved = "APPROVED" in evaluation.upper()
    feedback = ""

    if not approved:
        lines = evaluation.split("\n")
        for line in lines:
            if "REASON:" in line.upper():
                feedback = line.split(":", 1)[-1].strip()
                break

    print(f"   → Verdict: {'✅ APPROVED' if approved else '❌ REJECTED'}")
    if feedback:
        print(f"   → Feedback: {feedback}")

    return {
        **state,
        "approved":       approved,
        "critic_feedback": feedback
    }