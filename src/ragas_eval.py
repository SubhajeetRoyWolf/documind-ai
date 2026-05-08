import os
import mlflow
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from retrieval import ask

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)


# ── Core eval metrics ─────────────────────────────────────────

def measure_faithfulness(
    answer: str,
    context_chunks: list[dict]
) -> float:
    """
    Faithfulness: Is the answer grounded in context?
    Returns 0.0-1.0. Defaults to 0.5 if scoring fails.
    """
    # If answer says it doesn't know — that's perfectly faithful
    if any(phrase in answer.lower() for phrase in [
        "don't have", "not enough", "cannot find",
        "no information", "i don't", "not in"
    ]):
        return 1.0

    context_text = "\n".join([
        c.get("text", c.get("source", "")) 
        for c in context_chunks
    ])

    if not context_text.strip():
        return 0.5  # no context = can't judge

    prompt = f"""Rate if this answer is grounded in the context.
Reply with ONLY a single number: 0.0, 0.3, 0.5, 0.7, or 1.0

Context: {context_text[:500]}

Answer: {answer[:300]}

Score (0.0=hallucinated, 1.0=fully grounded):"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        text = response.choices[0].message.content.strip()
        # Extract first number found
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0.0), 1.0)
        return 0.5  # default if no number found
    except Exception:
        return 0.5  # default on any error


def measure_answer_relevancy(
    question: str,
    answer: str
) -> float:
    """
    Answer Relevancy: Does the answer actually
    address what was asked?
    Score: 0.0 (irrelevant) → 1.0 (perfectly relevant)
    """
    prompt = f"""Rate how well this answer addresses the question.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0:
- 1.0 = answer directly and completely addresses the question
- 0.5 = answer is partially relevant
- 0.0 = answer is completely irrelevant

Respond ONLY with a number:"""

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5


def measure_keyword_match(
    answer: str,
    keywords: list[str]
) -> float:
    """
    Keyword match: Does the answer contain
    expected ground truth keywords?
    Simple but reliable for regression testing.
    """
    answer_lower = answer.lower()
    matched = sum(
        1 for kw in keywords
        if kw.lower() in answer_lower
    )
    return matched / len(keywords) if keywords else 0.0


# ── Full eval pipeline ────────────────────────────────────────

def evaluate_single(qa_pair: dict) -> dict:
    """
    Run full eval on a single Q&A pair.
    Returns all metrics.
    """
    question     = qa_pair["question"]
    keywords     = qa_pair.get("keywords", [])

    print(f"\n  Q: {question[:60]}...")

    # Get system response
    result = ask(question, verbose=False)
    answer  = result["answer"]
    context = result["sources"]

    # Measure all metrics
    faithfulness     = measure_faithfulness(answer, context)
    answer_relevancy = measure_answer_relevancy(question, answer)
    keyword_match    = measure_keyword_match(answer, keywords)

    print(f"  Faithfulness:     {faithfulness:.3f}")
    print(f"  Answer Relevancy: {answer_relevancy:.3f}")
    print(f"  Keyword Match:    {keyword_match:.3f}")

    return {
        "question":        question,
        "answer":          answer,
        "faithfulness":    faithfulness,
        "answer_relevancy": answer_relevancy,
        "keyword_match":   keyword_match,
        "passed":          (
            faithfulness     >= 0.7 and
            answer_relevancy >= 0.7
        )
    }


def run_full_eval(
    dataset: list[dict],
    experiment_name: str = "documind_eval"
) -> dict:
    """
    Run evaluation on full golden dataset.
    Logs all metrics to MLflow.
    Returns summary.
    """
    print("\n" + "="*60)
    print("RAGAS EVALUATION RUN")
    print(f"Dataset size: {len(dataset)} questions")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    # Start MLflow run
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        results = []

        for qa in dataset:
            result = evaluate_single(qa)
            results.append(result)

        # Aggregate metrics
        avg_faithfulness     = sum(
            r["faithfulness"] for r in results
        ) / len(results)

        avg_answer_relevancy = sum(
            r["answer_relevancy"] for r in results
        ) / len(results)

        avg_keyword_match    = sum(
            r["keyword_match"] for r in results
        ) / len(results)

        pass_rate = sum(
            1 for r in results if r["passed"]
        ) / len(results)

        # Log to MLflow
        mlflow.log_metrics({
            "avg_faithfulness":     avg_faithfulness,
            "avg_answer_relevancy": avg_answer_relevancy,
            "avg_keyword_match":    avg_keyword_match,
            "pass_rate":            pass_rate,
            "total_questions":      len(dataset)
        })

        mlflow.log_params({
            "model":      os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            "embed_model": os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
            "dataset_size": len(dataset)
        })

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Avg Faithfulness:     {avg_faithfulness:.3f}  "
              f"{'✅' if avg_faithfulness >= 0.7 else '❌'}")
        print(f"Avg Answer Relevancy: {avg_answer_relevancy:.3f}  "
              f"{'✅' if avg_answer_relevancy >= 0.7 else '❌'}")
        print(f"Avg Keyword Match:    {avg_keyword_match:.3f}  "
              f"{'✅' if avg_keyword_match >= 0.7 else '❌'}")
        print(f"Pass Rate:            {pass_rate:.1%}  "
              f"{'✅' if pass_rate >= 0.7 else '❌'}")
        print("="*60)
        print(f"✅ Metrics logged to MLflow!")
        print(f"   Run: mlflow ui  →  http://localhost:5000")

        return {
            "avg_faithfulness":     avg_faithfulness,
            "avg_answer_relevancy": avg_answer_relevancy,
            "avg_keyword_match":    avg_keyword_match,
            "pass_rate":            pass_rate,
            "results":              results
        }


if __name__ == "__main__":
    from eval_dataset import GOLDEN_DATASET
    run_full_eval(GOLDEN_DATASET)