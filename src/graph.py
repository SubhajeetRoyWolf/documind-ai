from langgraph.graph import StateGraph, END
from agent_state import AgentState
from agents import (
    planner_agent,
    retriever_agent,
    generator_agent,
    critic_agent
)


# ── Routing logic ─────────────────────────────────────────────

def route_after_planner(state: AgentState) -> str:
    """After planning, always retrieve context."""
    return "retriever"


def route_after_critic(state: AgentState) -> str:
    """
    After critic evaluates:
    - APPROVED → end, return answer to user
    - REJECTED + attempts < 2 → retry retrieval
    - REJECTED + attempts >= 2 → end anyway (avoid infinite loop)
    """
    if state["approved"]:
        print("\n✅ Answer approved — returning to user")
        return END

    if state["attempts"] >= 2:
        print("\n⚠️  Max attempts reached — returning best answer")
        return END

    print("\n🔄 Answer rejected — retrying with critic feedback")
    return "retriever"


# ── Build the graph ───────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add all agent nodes
    graph.add_node("planner",   planner_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("generator", generator_agent)
    graph.add_node("critic",    critic_agent)

    # Entry point
    graph.set_entry_point("planner")

    # Fixed edges
    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")

    # Conditional edge — critic decides what happens next
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retriever": "retriever",   # retry
            END:          END            # done
        }
    )

    return graph.compile()


# ── Test the full graph ───────────────────────────────────────
if __name__ == "__main__":

    app = build_graph()

    test_questions = [
        "What are this candidate's Python and ML skills?",
        "What is the candidate's work experience?",
        "Summarise this candidate's profile in 3 bullet points",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"USER: {question}")
        print(f"{'='*60}")

        # Initial state
        initial_state: AgentState = {
            "question":        question,
            "plan":            "",
            "context":         [],
            "answer":          "",
            "sources":         [],
            "approved":        False,
            "critic_feedback": "",
            "attempts":        0,
            "chat_history":    []
        }

        # Run the graph
        final_state = app.invoke(initial_state)

        print(f"\n{'='*60}")
        print(f"FINAL ANSWER:")
        print(f"{'='*60}")
        print(final_state["answer"])
        print(f"\nPlan used: {final_state['plan']}")
        print(f"Attempts:  {final_state['attempts']}")
        print(f"Approved:  {final_state['approved']}")

        input("\nPress Enter for next question...")