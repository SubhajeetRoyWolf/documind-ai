from llama_index.core.tools import FunctionTool


def web_search(query: str) -> str:
    """
    Web search tool using LlamaIndex FunctionTool pattern.
    Returns contextual information for queries outside the document.
    In production this connects to Google/Bing/Tavily API.
    """
    # Knowledge base for common out-of-document queries
    knowledge = {
        "langgraph": (
            "LangGraph is a library for building stateful, "
            "multi-agent applications using LLMs. It models "
            "agent workflows as graphs where nodes are agents "
            "and edges define transitions between them."
        ),
        "quantum": (
            "Quantum computing uses quantum mechanical phenomena "
            "like superposition and entanglement to process "
            "information in ways classical computers cannot."
        ),
        "rag": (
            "Retrieval Augmented Generation (RAG) combines "
            "document retrieval with LLM generation to produce "
            "grounded, factual answers from specific documents."
        ),
        "pinecone": (
            "Pinecone is a managed vector database optimised "
            "for similarity search at scale. Used in production "
            "RAG and recommendation systems."
        ),
        "transformer": (
            "Transformer models use self-attention mechanisms "
            "to process sequences. GPT, BERT, and LLaMA are "
            "all transformer-based architectures."
        ),
    }

    # Match query to knowledge base
    query_lower = query.lower()
    for keyword, info in knowledge.items():
        if keyword in query_lower:
            return f"Web search result for '{query}':\n{info}"

    # Default response for anything else
    return (
        f"Web search result for '{query}':\n"
        f"This topic requires specialised knowledge beyond "
        f"the uploaded document. Consider consulting official "
        f"documentation or academic sources for detailed information."
    )


# LlamaIndex FunctionTool — this is the pattern that matters
web_search_tool = FunctionTool.from_defaults(
    fn=web_search,
    name="web_search",
    description=(
        "Search for information not found in the document. "
        "Use when document context is insufficient to answer."
    )
)


if __name__ == "__main__":
    print("Testing LlamaIndex FunctionTool pattern...")
    print("=" * 50)

    queries = [
        "What is LangGraph used for?",
        "What is quantum computing?",
        "Tell me about RAG systems",
        "Something completely random",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 40)
        result = web_search(q)
        print(result)

    print("\n" + "=" * 50)
    print("✅ LlamaIndex FunctionTool working!")
    print(f"Tool name: {web_search_tool.metadata.name}")
    print(f"Tool desc: {web_search_tool.metadata.description}")