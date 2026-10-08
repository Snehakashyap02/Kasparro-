import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from src.eligibility import check_eligibility
from src.extract import extract_candidate_from_text
from src.github import GitHubClient
from src.ingest import ingest_directory
from src.llm import LLMClient
from src.models import ScreeningResult
from src.scoring import apply_llm_analysis, deterministic_score


LOGGER = logging.getLogger(__name__)


def _screen_one(
    source_path: Path,
    text: str,
    llm_client: LLMClient,
    github_client: GitHubClient,
) -> ScreeningResult:
    candidate = extract_candidate_from_text(source_path.name, text)
    eligibility = check_eligibility(candidate)

    if not eligibility.eligible:
        return ScreeningResult(
            candidate_name=candidate.candidate_name,
            email=candidate.email,
            github_url=candidate.github_url,
            github_username=candidate.github_username,
            eligible=False,
            matched_skills=eligibility.matched_skills,
            evidence_snippets=eligibility.evidence_snippets,
            rejection_reasons=eligibility.rejection_reasons,
            source_file=source_path.name,
        )

    github = github_client.enrich(candidate.github_username)
    analysis = llm_client.analyze_resume(candidate)

    if analysis is not None:
        breakdown, evidence = apply_llm_analysis(analysis, github)
        project_summary = analysis.project_summary
        strengths = analysis.strengths
        concerns = analysis.concerns
    else:
        breakdown, evidence = deterministic_score(candidate, github)
        project_summary = _fallback_project_summary(candidate)
        strengths = _fallback_strengths(candidate)
        concerns = ["LLM scoring unavailable; deterministic scoring used."]

    total = (
        breakdown.ai_project_depth
        + breakdown.python_backend
        + breakdown.cloud_fullstack
        + breakdown.github
        + breakdown.engineering_depth
    )

    return ScreeningResult(
        candidate_name=candidate.candidate_name,
        email=candidate.email,
        github_url=candidate.github_url,
        github_username=candidate.github_username,
        eligible=True,
        total_score=max(0, min(100, total)),
        score_breakdown=breakdown,
        matched_skills=eligibility.matched_skills,
        evidence=evidence,
        evidence_snippets=eligibility.evidence_snippets,
        project_summary=project_summary,
        github_summary=github.summary,
        strengths=strengths,
        concerns=concerns,
        rejection_reasons=[],
        source_file=source_path.name,
    )


def _fallback_project_summary(candidate) -> str:
    project = " ".join(candidate.projects.split())
    return project[:300] if project else "No detailed project section found."


def _fallback_strengths(candidate) -> List[str]:
    strengths = []
    if "Python" in candidate.skills or any(s.lower() == "python" for s in candidate.skills):
        strengths.append("Python evidence present")
    if candidate.projects:
        strengths.append("Project experience provided")
    return strengths


def run_pipeline(input_dir: Path) -> Dict:
    parsed, failures = ingest_directory(input_dir)

    llm_client = LLMClient()
    github_client = GitHubClient()

    results: List[ScreeningResult] = []
    completed_futures = set()
    interrupted = False

    # Bounded concurrency is used for the network-heavy eligible-candidate path.
    executor = ThreadPoolExecutor(max_workers=5)
    futures = []
    try:
        futures = [
            executor.submit(
                _screen_one,
                source_path,
                text,
                llm_client,
                github_client,
            )
            for source_path, text in parsed
        ]

        for future in as_completed(futures):
            completed_futures.add(future)
            try:
                results.append(future.result())
            except Exception as exc:
                LOGGER.exception("Unexpected candidate processing failure: %s", exc)
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Interrupt received; cancelling unfinished candidates.")

        for future in futures:
            if future in completed_futures:
                continue

            if future.done() and not future.cancelled():
                completed_futures.add(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    LOGGER.exception(
                        "Unexpected candidate processing failure: %s",
                        exc,
                    )
            else:
                future.cancel()
    finally:
        executor.shutdown(
            wait=not interrupted,
            cancel_futures=interrupted,
        )

    eligible = [r for r in results if r.eligible]
    rejected = [r for r in results if not r.eligible]

    eligible.sort(key=lambda r: r.total_score, reverse=True)
    for index, result in enumerate(eligible, start=1):
        result.rank = index

    # Rejected candidates follow ranked eligible candidates.
    final_results = eligible + rejected

    return {
        "batch_summary": {
            "total": len(parsed) + len(failures),
            "parsed": len(parsed),
            "eligible": len(eligible),
            "rejected": len(rejected),
            "failed": len(failures),
            "completed": len(results),
            "interrupted": interrupted,
        },
        "candidates": [
            result.model_dump() for result in final_results
        ],
        "failed_files": failures,
    }
