from langgraph.graph import StateGraph, END
from agent_state import AgentState
from agents import (
    planner_agent,
    retriever_agent,
    generator_agent,
    critic_agent
)
from memory import ConversationMemory
from web_search_tool import web_search

# Global memory — persists across questions
memory = ConversationMemory(max_turns=5)


# ── Web search agent — used when document has no answer ───────
def web_search_agent(state: AgentState) -> AgentState:
    """
    Fallback agent — searches web when Pinecone has no answer.
    Uses LlamaIndex web_search_tool.
    """
    print(f"\n🌐 WEB SEARCH AGENT searching online...")

    results = web_search(state["question"])
    print(f"   → Found web results ({len(results)} chars)")

    # Add web results as a context chunk
    web_context = [{
        "text":         results[:1000],
        "source":       "web_search",
        "page":         0,
        "rerank_score": 0.5
    }]

    return {**state, "context": web_context}


# ── Smart routing after Planner ───────────────────────────────
def route_after_planner(state: AgentState) -> str:
    """
    Route based on plan:
    - retrieve_and_answer → retriever
    - summarise           → retriever (needs context)
    - clarify             → end immediately
    """
    if state["plan"] == "clarify":
        # Inject a clarification message
        clarify_state = {
            **state,
            "answer":   "Could you please be more specific? "
                        "I can answer questions about the document's "
                        "content, skills, experience, and projects.",
            "approved": True
        }
        return "end_clarify"

    return "retriever"


def route_after_critic(state: AgentState) -> str:
    # Always end after 2 attempts to prevent loops
    if state["attempts"] >= 2:
        return END

    if state["approved"]:
        return END

    # Low relevance scores → try web search once
    if state["context"]:
        avg_score = sum(
            c.get("rerank_score", 0) 
            for c in state["context"]
        ) / len(state["context"])

        if avg_score < 0.05 and state["attempts"] < 2:
            print("\n🌐 Low relevance → switching to web search")
            return "web_search"

    return END


# ── Build graph v2 ────────────────────────────────────────────
def build_graph_v2():
    graph = StateGraph(AgentState)

    graph.add_node("planner",    planner_agent)
    graph.add_node("retriever",  retriever_agent)
    graph.add_node("web_search", web_search_agent)
    graph.add_node("generator",  generator_agent)
    graph.add_node("critic",     critic_agent)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "end_clarify": END
        }
    )

    graph.add_edge("retriever",  "generator")
    graph.add_edge("web_search", "generator")
    graph.add_edge("generator",  "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retriever":  "retriever",
            "web_search": "web_search",
            END:           END
        }
    )

    return graph.compile()


# ── Multi-turn chat loop ──────────────────────────────────────
def chat():
    """
    Interactive multi-turn chat with memory.
    The agent remembers previous questions in the session.
    """
    app = build_graph_v2()

    print("\n" + "="*60)
    print("DocuMind AI — Agentic RAG Assistant")
    print("Multi-agent: Planner + Retriever + Generator + Critic")
    print("Type 'quit' to exit | 'clear' to reset memory")
    print("="*60)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if question.lower() == "clear":
            memory.clear()
            print("🧹 Memory cleared.")
            continue

        # Build initial state with conversation history
        initial_state: AgentState = {
            "question":        question,
            "plan":            "",
            "context":         [],
            "answer":          "",
            "sources":         [],
            "approved":        False,
            "critic_feedback": "",
            "attempts":        0,
            "chat_history":    memory.get()
        }

        # Run the agent graph
        final_state = app.invoke(initial_state)

        # Store this turn in memory
        memory.add("user",      question)
        memory.add("assistant", final_state["answer"])

        # Display answer
        print(f"\n{'─'*60}")
        print(f"DocuMind: {final_state['answer']}")
        print(f"{'─'*60}")
        print(f"[Plan: {final_state['plan']} | "
              f"Attempts: {final_state['attempts']} | "
              f"Approved: {final_state['approved']}]")


if __name__ == "__main__":
    chat()