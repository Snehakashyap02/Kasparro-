from src.models import Candidate, GitHubResult, ResumeAnalysis
from src.scoring import apply_llm_analysis, deterministic_score


def candidate():
    return Candidate(
        source_file="test.pdf",
        candidate_name="Test Candidate",
        full_text="",
        skills=["Python", "FastAPI", "LangChain"],
        projects=(
            "Built a RAG application using LangChain, embeddings, retrieval, "
            "FastAPI, Docker, pytest, caching and evaluation."
        ),
        experience="Python backend development",
    )


def github():
    return GitHubResult(
        points=8,
        recent_activity_points=4,
        maintained_relevant_points=4,
        summary="Active GitHub profile.",
    )


def test_deterministic_score_is_within_100():
    breakdown, _ = deterministic_score(candidate(), github())
    total = (
        breakdown.ai_project_depth
        + breakdown.python_backend
        + breakdown.cloud_fullstack
        + breakdown.github
        + breakdown.engineering_depth
    )
    assert 0 <= total <= 100


def test_llm_scores_are_clamped():
    analysis = ResumeAnalysis(
        ai_project_depth_score=40,
        python_backend_score=30,
        cloud_fullstack_score=15,
        engineering_depth_score=5,
        project_summary="Strong project",
        is_thin_wrapper=False,
    )
    breakdown, _ = apply_llm_analysis(analysis, github())
    assert breakdown.ai_project_depth <= 40
    assert breakdown.python_backend <= 30
    assert breakdown.cloud_fullstack <= 15
    assert breakdown.engineering_depth <= 5


def test_thin_wrapper_is_penalized():
    normal = ResumeAnalysis(
        ai_project_depth_score=35,
        python_backend_score=25,
        cloud_fullstack_score=10,
        engineering_depth_score=4,
        is_thin_wrapper=False,
    )
    thin = normal.model_copy(update={"is_thin_wrapper": True})

    normal_breakdown, _ = apply_llm_analysis(normal, github())
    thin_breakdown, _ = apply_llm_analysis(thin, github())

    normal_total = sum(normal_breakdown.model_dump().values())
    thin_total = sum(thin_breakdown.model_dump().values())

    assert thin_total < normal_total
