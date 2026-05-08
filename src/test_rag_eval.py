"""
Pytest regression test suite for DocuMind AI.
These tests run in CI/CD and FAIL the build if
quality drops below acceptable thresholds.

Run: pytest test_rag_eval.py -v
"""
import pytest
from retrieval import ask
from ragas_eval import (
    measure_faithfulness,
    measure_answer_relevancy,
    measure_keyword_match
)

# ── Thresholds — build fails if below these ───────────────────
FAITHFULNESS_THRESHOLD     = 0.30
ANSWER_RELEVANCY_THRESHOLD = 0.70
KEYWORD_MATCH_THRESHOLD    = 0.50


# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def experience_response():
    """Cache response so we don't call API multiple times."""
    return ask(
        "How many years of experience does the candidate have?",
        verbose=False
    )

@pytest.fixture(scope="module")
def skills_response():
    return ask(
        "What are the candidate's Python and ML skills?",
        verbose=False
    )

@pytest.fixture(scope="module")
def employer_response():
    return ask(
        "Who is the candidate's current employer?",
        verbose=False
    )


# ── Tests ─────────────────────────────────────────────────────

class TestFaithfulness:
    """Hallucination detection tests."""

    def test_experience_answer_is_grounded(self, experience_response):
        score = measure_faithfulness(
            experience_response["answer"],
            experience_response["sources"]
        )
        assert score >= FAITHFULNESS_THRESHOLD, (
            f"Faithfulness {score:.3f} below threshold "
            f"{FAITHFULNESS_THRESHOLD}. "
            f"Answer may contain hallucinations."
        )

    def test_skills_answer_is_grounded(self, skills_response):
        score = measure_faithfulness(
            skills_response["answer"],
            skills_response["sources"]
        )
        assert score >= FAITHFULNESS_THRESHOLD, (
            f"Faithfulness {score:.3f} below threshold."
        )


class TestAnswerRelevancy:
    """Answer quality tests."""

    def test_experience_answer_is_relevant(self, experience_response):
        score = measure_answer_relevancy(
            "How many years of experience does the candidate have?",
            experience_response["answer"]
        )
        assert score >= ANSWER_RELEVANCY_THRESHOLD, (
            f"Relevancy {score:.3f} below threshold."
        )

    def test_employer_answer_is_relevant(self, employer_response):
        score = measure_answer_relevancy(
            "Who is the candidate's current employer?",
            employer_response["answer"]
        )
        assert score >= ANSWER_RELEVANCY_THRESHOLD, (
            f"Relevancy {score:.3f} below threshold."
        )


class TestKeywordMatch:
    """Ground truth keyword tests."""

    def test_experience_contains_years(self, experience_response):
        score = measure_keyword_match(
            experience_response["answer"],
            ["3.5", "years"]
        )
        assert score >= KEYWORD_MATCH_THRESHOLD, (
            f"Answer missing expected keywords. "
            f"Got: {experience_response['answer'][:100]}"
        )

    def test_employer_contains_saks(self, employer_response):
        score = measure_keyword_match(
            employer_response["answer"],
            ["saks", "global"]
        )
        assert score >= KEYWORD_MATCH_THRESHOLD, (
            f"Answer missing 'Saks Global'. "
            f"Got: {employer_response['answer'][:100]}"
        )

    def test_skills_contains_python(self, skills_response):
        score = measure_keyword_match(
            skills_response["answer"],
            ["python"]
        )
        assert score >= KEYWORD_MATCH_THRESHOLD, (
            f"Skills answer missing Python."
        )


class TestHallucinationPrevention:
    """Tests that system refuses to answer out-of-scope questions."""

    def test_refuses_out_of_scope(self):
        """System should say it doesn't know for unrelated topics."""
        result = ask("What is the capital of France?", verbose=False)
        answer = result["answer"].lower()

        # Should NOT confidently answer with "paris"
        # Should say it doesn't have information
        refuses = any(phrase in answer for phrase in [
            "don't have",
            "not in",
            "cannot find",
            "no information",
            "i don't",
            "not enough"
        ])
        assert refuses, (
            f"System should refuse out-of-scope questions. "
            f"Got: {result['answer'][:100]}"
        )