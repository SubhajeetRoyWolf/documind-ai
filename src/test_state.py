from agent_state import AgentState

# Simulate what state looks like as it flows through agents
initial_state: AgentState = {
    "question":       "What are the candidate's Python skills?",
    "plan":           "",
    "context":        [],
    "answer":         "",
    "sources":        [],
    "approved":       False,
    "critic_feedback": "",
    "attempts":       0,
    "chat_history":   []
}

print("Initial state:")
for key, value in initial_state.items():
    print(f"  {key:20} → {value}")

# Simulate Planner updating state
planner_update = initial_state.copy()
planner_update["plan"] = "retrieve_and_answer"
print("\nAfter Planner:")
print(f"  plan → {planner_update['plan']}")

# Simulate Retriever updating state
retriever_update = planner_update.copy()
retriever_update["context"] = [
    {"text": "Python (advanced), Pandas, NumPy...", "source": "resume.pdf", "page": 1}
]
print("\nAfter Retriever:")
print(f"  context → {len(retriever_update['context'])} chunks loaded")

print("\n✅ State concept working — agents share data through state!")