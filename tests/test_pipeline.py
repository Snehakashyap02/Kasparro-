from concurrent.futures import Future
from pathlib import Path

import src.pipeline as pipeline
from src.models import (
    Candidate,
    EligibilityResult,
    GitHubResult,
    ResumeAnalysis,
    ScoreBreakdown,
    ScreeningResult,
)


def test_screen_one_includes_contact_and_evidence(monkeypatch):
    candidate = Candidate(
        source_file="candidate.pdf",
        candidate_name="Test Candidate",
        email="candidate@example.com",
        github_url="https://github.com/candidate",
        github_username="candidate",
        full_text="",
        skills=["Python", "FastAPI", "LangGraph"],
        projects="Built a LangGraph RAG service.",
    )
    monkeypatch.setattr(
        pipeline,
        "extract_candidate_from_text",
        lambda *_args: candidate,
    )
    monkeypatch.setattr(
        pipeline,
        "check_eligibility",
        lambda _candidate: EligibilityResult(
            eligible=True,
            matched_skills=["Python", "FastAPI", "LangGraph"],
            evidence_snippets=["Built a LangGraph RAG service."],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "apply_llm_analysis",
        lambda *_args: (
            ScoreBreakdown(ai_project_depth=20),
            {"ai": "LangGraph RAG service"},
        ),
    )

    class FakeGitHubClient:
        def enrich(self, _username):
            return GitHubResult(
                points=0,
                recent_activity_points=0,
                maintained_relevant_points=0,
                summary="Profile checked; no qualifying activity found.",
                username="candidate",
            )

    class FakeLLMClient:
        def analyze_resume(self, _candidate):
            return ResumeAnalysis(
                ai_project_depth_score=20,
                python_backend_score=0,
                cloud_fullstack_score=0,
                engineering_depth_score=0,
                evidence={"ai": "LangGraph RAG service"},
            )

    result = pipeline._screen_one(
        Path("candidate.pdf"),
        "resume text",
        FakeLLMClient(),
        FakeGitHubClient(),
    )

    assert result.email == "candidate@example.com"
    assert result.github_url == "https://github.com/candidate"
    assert result.github_username == "candidate"
    assert result.evidence["ai"] == "LangGraph RAG service"
    assert result.evidence_snippets == ["Built a LangGraph RAG service."]


def test_keyboard_interrupt_returns_completed_results_and_cancels_pending(
    monkeypatch,
):
    class FakeExecutor:
        instance = None

        def __init__(self, max_workers):
            self.futures = []
            self.shutdown_args = None
            FakeExecutor.instance = self

        def submit(self, *_args, **_kwargs):
            future = Future()
            if not self.futures:
                future.set_result(
                    ScreeningResult(
                        candidate_name="Completed",
                        eligible=False,
                        source_file="completed.pdf",
                    )
                )
            self.futures.append(future)
            return future

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_args = (wait, cancel_futures)

    def interrupt_after_one(futures):
        yield futures[0]
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline, "as_completed", interrupt_after_one)
    monkeypatch.setattr(
        pipeline,
        "ingest_directory",
        lambda _input_dir: (
            [(Path("completed.pdf"), "resume"), (Path("pending.pdf"), "resume")],
            [],
        ),
    )
    monkeypatch.setattr(pipeline, "LLMClient", lambda: object())
    monkeypatch.setattr(pipeline, "GitHubClient", lambda: object())

    results = pipeline.run_pipeline(Path("resumes"))

    assert results["batch_summary"]["completed"] == 1
    assert results["batch_summary"]["interrupted"] is True
    assert [item["source_file"] for item in results["candidates"]] == [
        "completed.pdf"
    ]
    assert FakeExecutor.instance.futures[1].cancelled()
    assert FakeExecutor.instance.shutdown_args == (False, True)