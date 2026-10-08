import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import requests

from src.config import GITHUB_TOKEN, GITHUB_RECENT_DAYS, REQUEST_TIMEOUT_SECONDS
from src.models import GitHubResult


LOGGER = logging.getLogger(__name__)


class GitHubClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.cache: Dict[str, GitHubResult] = {}

    def _headers(self) -> dict:
        headers = {"Accept": "application/vnd.github+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        return headers

    def enrich(self, username: Optional[str]) -> GitHubResult:
        if not username:
            return GitHubResult(
                points=0,
                recent_activity_points=0,
                maintained_relevant_points=0,
                summary="No GitHub profile found in resume.",
            )

        if username in self.cache:
            return self.cache[username]

        try:
            repos_response = self.session.get(
                f"https://api.github.com/users/{username}/repos",
                params={"per_page": 100, "sort": "updated"},
                headers=self._headers(),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if repos_response.status_code == 404:
                result = GitHubResult(
                    points=0,
                    recent_activity_points=0,
                    maintained_relevant_points=0,
                    summary="GitHub profile not found (404).",
                    username=username,
                )
                self.cache[username] = result
                return result

            if repos_response.status_code == 403:
                raise RuntimeError("rate limited or forbidden")

            repos_response.raise_for_status()
            repos = repos_response.json()

            events_response = self.session.get(
                f"https://api.github.com/users/{username}/events/public",
                params={"per_page": 100},
                headers=self._headers(),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            recent_activity = 0
            cutoff = datetime.now(timezone.utc) - timedelta(days=GITHUB_RECENT_DAYS)

            if events_response.ok:
                for event in events_response.json():
                    created = event.get("created_at")
                    if created:
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        if dt >= cutoff:
                            recent_activity += 1

            recent_points = min(5, 1 + recent_activity // 3) if recent_activity else 0

            relevant = 0
            for repo in repos:
                if repo.get("fork"):
                    continue

                updated = repo.get("updated_at")
                recently_updated = False
                if updated:
                    dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    recently_updated = dt >= cutoff

                language = (repo.get("language") or "").lower()
                topics = " ".join(repo.get("topics") or []).lower()

                if recently_updated and (
                    language == "python"
                    or any(term in topics for term in ("ai", "llm", "rag", "machine-learning"))
                ):
                    relevant += 1

            maintained_points = min(5, relevant)

            result = GitHubResult(
                points=recent_points + maintained_points,
                recent_activity_points=recent_points,
                maintained_relevant_points=maintained_points,
                summary=(
                    f"Recent activity: {recent_activity} public events; "
                    f"{relevant} recently maintained/relevant Python or AI repositories."
                ),
                username=username,
            )
            self.cache[username] = result
            return result

        except requests.Timeout:
            result = GitHubResult(
                points=0,
                recent_activity_points=0,
                maintained_relevant_points=0,
                summary="GitHub enrichment failed: timeout.",
                username=username,
            )
        except Exception as exc:
            LOGGER.warning("GitHub enrichment failed for %s: %s", username, exc)
            result = GitHubResult(
                points=0,
                recent_activity_points=0,
                maintained_relevant_points=0,
                summary=f"GitHub enrichment failed: {exc}",
                username=username,
            )

        self.cache[username] = result
        return result
