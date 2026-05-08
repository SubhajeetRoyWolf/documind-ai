# Hand-crafted golden Q&A pairs based on resume.pdf
# These are questions we KNOW the answers to
# Used for regression testing across model versions

GOLDEN_DATASET = [
    {
        "question": "How many years of experience does the candidate have?",
        "ground_truth": "3.5 years",
        "keywords": ["3.5", "years"]
    },
    {
        "question": "What is the candidate's current employer?",
        "ground_truth": "Saks Global India",
        "keywords": ["saks", "global"]
    },
    {
        "question": "What agent frameworks does the candidate know?",
        "ground_truth": "LangChain and Agentic Workflows",
        "keywords": ["langchain", "agentic"]
    },
    {
        "question": "What vector database has the candidate used?",
        "ground_truth": "FAISS",
        "keywords": ["faiss"]
    },
    {
        "question": "What is the candidate's educational qualification?",
        "ground_truth": "B.Tech in Computer Science Engineering from Presidency University",
        "keywords": ["b.tech", "computer science", "presidency"]
    },
    {
        "question": "What MLOps tools does the candidate use?",
        "ground_truth": "MLflow, Docker, Jenkins CI/CD",
        "keywords": ["mlflow", "docker", "jenkins"]
    },
    {
        "question": "What programming language is the candidate most proficient in?",
        "ground_truth": "Python",
        "keywords": ["python"]
    },
    {
        "question": "What cloud platform has the candidate worked with?",
        "ground_truth": "Azure OpenAI",
        "keywords": ["azure"]
    },
]