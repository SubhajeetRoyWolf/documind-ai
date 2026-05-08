class ConversationMemory:
    """
    Simple multi-turn conversation memory.
    Stores the last N turns so agents remember context.
    This is what LangGraph calls a 'checkpointer' in production.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        """Add a message to history."""
        self.history.append({
            "role":    role,
            "content": content
        })
        # Keep only last N turns (each turn = user + assistant)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get(self) -> list[dict]:
        """Return full history."""
        return self.history.copy()

    def format_for_prompt(self) -> str:
        """Format history as readable text for LLM prompts."""
        if not self.history:
            return ""
        lines = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history = []