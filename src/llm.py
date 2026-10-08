import json
import logging
import time
from typing import Optional, Any

from openai import OpenAI

from src.config import (
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)
from src.models import Candidate, ResumeAnalysis

LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert technical recruiter screening candidates for an SDE internship.

Evaluate ONLY evidence explicitly present in the resume.

You MUST return exactly ONE valid JSON object.

Do not return:
- Markdown
- ```json
- explanations outside JSON
- comments
- trailing text

IMPORTANT SCORING RULES:

AI PROJECT DEPTH: 0-40
- Give credit for meaningful LLM, RAG, agentic AI, LangChain,
  LangGraph, LlamaIndex, embeddings, vector databases, retrieval,
  tool/function calling, multi-agent workflows, LLM evaluation,
  AI application architecture, or similar work.
- Skills alone should not receive high AI project scores.
- A real project with architecture/workflow/retrieval/tools/state/evaluation
  should score higher.
- A simple "called OpenAI API and generated text" project is a thin wrapper
  and should receive very little AI project credit.
- CNN, RNN, LSTM, Random Forest, XGBoost, scikit-learn classification,
  regression, and other conventional ML projects do NOT count as LLM/agentic
  AI unless the resume clearly contains qualifying LLM/RAG/agentic work.

PYTHON/BACKEND: 0-30
- Python
- FastAPI
- Flask
- Django
- REST APIs
- databases
- backend architecture
- async/concurrency
- caching
- queues

CLOUD/FULL-STACK: 0-15
- Docker
- AWS
- GCP
- Azure
- Kubernetes
- deployment
- React
- Next.js
- full-stack applications

ENGINEERING DEPTH: 0-5
- testing
- architecture
- error handling
- retries
- logging
- monitoring
- caching
- concurrency
- production engineering

Never invent evidence.

Keep project_summary short.

Keep strengths and concerns short.

Keep evidence values short.

The response MUST be valid JSON.
"""


def _text(value: Any, limit: int) -> str:
    """Convert a value to text and limit prompt size."""
    if value is None:
        return ""

    text = str(value).strip()

    if len(text) > limit:
        return text[:limit] + "\n[TRUNCATED]"

    return text


def _extract_json(content: str) -> Optional[dict]:
    """Safely extract a JSON object from model output."""

    if not content:
        return None

    content = content.strip()

    # Remove Markdown fences if the model ignored the instruction.
    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    # Normal JSON.
    try:
        result = json.loads(content)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Try to locate the JSON object inside additional text.
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    json_text = content[start:end + 1]

    try:
        result = json.loads(json_text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        return None

    return None


def _clamp(value: Any, minimum: int, maximum: int) -> int:
    """Safely clamp a score."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return minimum

    return max(minimum, min(maximum, value))


def _normalise_result(data: dict) -> dict:
    """
    Normalise the model response before Pydantic validation.

    This protects the pipeline from slightly imperfect LLM output.
    """

    data["ai_project_depth_score"] = _clamp(
        data.get("ai_project_depth_score", 0),
        0,
        40,
    )

    data["python_backend_score"] = _clamp(
        data.get("python_backend_score", 0),
        0,
        30,
    )

    data["cloud_fullstack_score"] = _clamp(
        data.get("cloud_fullstack_score", 0),
        0,
        15,
    )

    data["engineering_depth_score"] = _clamp(
        data.get("engineering_depth_score", 0),
        0,
        5,
    )

    if not isinstance(data.get("project_summary"), str):
        data["project_summary"] = str(
            data.get("project_summary", "")
        )

    if not isinstance(data.get("strengths"), list):
        data["strengths"] = []

    if not isinstance(data.get("concerns"), list):
        data["concerns"] = []

    if not isinstance(data.get("evidence"), dict):
        data["evidence"] = {}

    data["is_thin_wrapper"] = bool(
        data.get("is_thin_wrapper", False)
    )

    return data


class LLMClient:
    def __init__(self) -> None:

        self.enabled = bool(OPENAI_API_KEY)
        self.client: Optional[OpenAI] = None

        if self.enabled:
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                timeout=LLM_TIMEOUT_SECONDS,
            )

            LOGGER.info(
                "LLM enabled: model=%s, base_url=%s",
                LLM_MODEL,
                OPENAI_BASE_URL,
            )

    def _build_prompt(
        self,
        candidate: Candidate,
        compact: bool = False,
    ) -> str:

        if compact:
            skills_limit = 1500
            projects_limit = 3500
            experience_limit = 2500
            education_limit = 1000
        else:
            skills_limit = 2500
            projects_limit = 5000
            experience_limit = 3500
            education_limit = 1500

        return f"""
Evaluate this candidate.

CANDIDATE:
{_text(candidate.candidate_name, 300)}

SKILLS:
{_text(", ".join(candidate.skills), skills_limit)}

PROJECTS:
{_text(candidate.projects, projects_limit)}

EXPERIENCE:
{_text(candidate.experience, experience_limit)}

EDUCATION:
{_text(candidate.education, education_limit)}

Return EXACTLY this JSON structure:

{{
  "ai_project_depth_score": 0,
  "python_backend_score": 0,
  "cloud_fullstack_score": 0,
  "engineering_depth_score": 0,
  "project_summary": "short summary",
  "strengths": ["short strength"],
  "concerns": ["short concern"],
  "is_thin_wrapper": false,
  "evidence": {{
    "ai": "short evidence",
    "python_backend": "short evidence",
    "cloud_fullstack": "short evidence",
    "engineering": "short evidence"
  }}
}}

SCORING LIMITS:
ai_project_depth_score: 0-40
python_backend_score: 0-30
cloud_fullstack_score: 0-15
engineering_depth_score: 0-5

IMPORTANT:
Return ONLY JSON.
Keep every text field concise.
Do not invent anything.
"""

    def _call_model(self, prompt: str) -> str:

        if not self.client:
            raise RuntimeError("LLM client is not configured.")

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=4096,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "resume_analysis",
                    "schema": ResumeAnalysis.model_json_schema(),
                },
            },
        )

        if not response.choices:
            raise ValueError("LLM returned no choices.")

        message = response.choices[0].message

        # Some NVIDIA responses can contain refusal/reasoning information
        # while leaving content empty.
        content = getattr(message, "content", None)

        if not content:
            raise ValueError("LLM returned empty content.")

        return content

    def analyze_resume(
        self,
        candidate: Candidate,
    ) -> Optional[ResumeAnalysis]:

        if not self.client:
            LOGGER.warning(
                "OpenAI API key not configured; using deterministic scoring."
            )
            return None

        prompt = self._build_prompt(
            candidate,
            compact=False,
        )

        for attempt in range(LLM_MAX_RETRIES + 1):

            try:
                raw_response = self._call_model(prompt)

                parsed = _extract_json(raw_response)

                if parsed is None:
                    raise ValueError(
                        "LLM returned malformed or incomplete JSON."
                    )

                parsed = _normalise_result(parsed)

                result = ResumeAnalysis.model_validate(parsed)

                return result

            except Exception as exc:

                LOGGER.warning(
                    "LLM attempt %d failed for %s: %s",
                    attempt + 1,
                    candidate.candidate_name,
                    exc,
                )

                if attempt < LLM_MAX_RETRIES:

                    # Make the next request smaller after a bad response.
                    prompt = self._build_prompt(
                        candidate,
                        compact=True,
                    )

                    time.sleep(1)

        LOGGER.warning(
            "LLM scoring failed for %s; deterministic fallback will be used.",
            candidate.candidate_name,
        )

        return None