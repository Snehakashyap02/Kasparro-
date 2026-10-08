import re
from typing import Dict, List, Tuple

from src.models import (
    Candidate,
    GitHubResult,
    ResumeAnalysis,
    ScoreBreakdown,
)


# =========================================================
# KEYWORD GROUPS
# =========================================================

AI_STRONG_SIGNALS = [
    "rag",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "langchain",
    "langgraph",
    "llamaindex",
    "llama index",
    "agentic ai",
    "agentic",
    "ai agent",
    "ai agents",
    "multi-agent",
    "multi agent",
    "tool calling",
    "function calling",
    "tool-calling",
    "embeddings",
    "embedding",
    "vector search",
    "vector database",
    "vector db",
    "vector store",
    "semantic search",
    "semantic retrieval",
    "faiss",
    "pinecone",
    "weaviate",
    "chroma",
    "chromadb",
    "qdrant",
    "llm",
    "large language model",
    "generative ai",
    "genai",
    "openai",
    "gpt",
    "claude",
    "anthropic",
    "gemini",
    "llama",
    "mistral",
    "hugging face",
    "huggingface",
    "transformers",
]


AI_DEPTH_SIGNALS = [
    "retrieval",
    "retrieval pipeline",
    "retrieval pipeline",
    "document retrieval",
    "semantic retrieval",
    "vector search",
    "vector database",
    "vector store",
    "embedding",
    "embeddings",
    "prompt engineering",
    "prompt template",
    "structured output",
    "tool calling",
    "function calling",
    "tools",
    "agent workflow",
    "agent workflows",
    "multi-agent",
    "multi agent",
    "orchestration",
    "workflow",
    "state management",
    "stateful",
    "conversation memory",
    "memory",
    "evaluation",
    "evaluation pipeline",
    "evaluation pipelines",
    "reranking",
    "re-ranking",
    "chunking",
    "document processing",
    "knowledge base",
    "context-aware",
    "context aware",
    "question answering",
    "document qa",
    "document question answering",
]


AI_BUSINESS_LOGIC_SIGNALS = [
    "backend",
    "api",
    "rest api",
    "restful api",
    "fastapi",
    "flask",
    "database",
    "postgresql",
    "postgres",
    "mongodb",
    "sqlite",
    "authentication",
    "authorization",
    "role-based",
    "role based",
    "dashboard",
    "upload",
    "file processing",
    "document processing",
    "pipeline",
    "workflow",
    "real-time",
    "realtime",
]


PYTHON_BACKEND_SIGNALS = [
    "python",
    "fastapi",
    "flask",
    "django",
    "backend",
    "rest api",
    "restful api",
    "api development",
    "microservice",
    "microservices",
    "asyncio",
    "async programming",
    "asynchronous",
    "postgresql",
    "postgres",
    "redis",
    "celery",
]


CLOUD_DEPLOYMENT_SIGNALS = [
    "gcp",
    "google cloud",
    "google cloud platform",
    "aws",
    "amazon web services",
    "azure",
    "docker",
    "dockerfile",
    "kubernetes",
    "k8s",
    "deployment",
    "deployed",
    "cloud",
    "cloud deployment",
    "ci/cd",
    "cicd",
    "github actions",
    "jenkins",
    "terraform",
    "vercel",
    "render",
    "railway",
]


FULLSTACK_SIGNALS = [
    "react",
    "react.js",
    "next.js",
    "nextjs",
    "frontend",
    "full stack",
    "full-stack",
]


ENGINEERING_DEPTH_SIGNALS = [
    "unit testing",
    "unit tests",
    "integration testing",
    "integration tests",
    "pytest",
    "testing",
    "test coverage",
    "architecture",
    "system design",
    "design patterns",
    "caching",
    "cache",
    "redis",
    "queue",
    "message queue",
    "rabbitmq",
    "kafka",
    "celery",
    "observability",
    "logging",
    "monitoring",
    "metrics",
    "tracing",
    "concurrency",
    "parallel processing",
    "async",
    "failure handling",
    "error handling",
    "retry",
    "retries",
    "fault tolerance",
    "scalability",
    "horizontal scaling",
    "rate limiting",
    "load balancing",
]


CLASSIC_ML_ONLY_SIGNALS = [
    "scikit-learn",
    "sklearn",
    "random forest",
    "xgboost",
    "lightgbm",
    "logistic regression",
    "linear regression",
    "decision tree",
    "support vector machine",
    "svm",
    "k-means",
    "kmeans",
    "knn",
    "cnn",
    "convolutional neural network",
    "rnn",
    "recurrent neural network",
    "lstm",
    "gru",
    "machine learning",
    "deep learning",
    "neural network",
    "classification",
    "regression",
    "clustering",
    "predictive modeling",
]


LLM_API_ONLY_SIGNALS = [
    "openai api",
    "openai api call",
    "gemini api",
    "anthropic api",
    "claude api",
    "gpt api",
    "llm api",
    "api call",
]


# =========================================================
# GENERIC HELPERS
# =========================================================

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _contains(text: str, keyword: str) -> bool:
    """
    Safe keyword detection.
    """
    if not text or not keyword:
        return False

    normalized_text = _normalize(text)
    normalized_keyword = _normalize(keyword)

    if len(normalized_keyword) <= 4:
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])",
                normalized_text,
            )
        )

    return normalized_keyword in normalized_text


def _count_matches(text: str, keywords: List[str]) -> int:
    return sum(1 for keyword in keywords if _contains(text, keyword))


def _matched_keywords(text: str, keywords: List[str]) -> List[str]:
    return [
        keyword
        for keyword in keywords
        if _contains(text, keyword)
    ]


def _has_python(text: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_])python(?![A-Za-z0-9_])",
            text or "",
            re.I,
        )
    )


# =========================================================
# PROJECT QUALITY ANALYSIS
# =========================================================

def _project_has_real_ai(project_text: str) -> bool:
    """
    Determines whether project/experience text has actual
    qualifying AI evidence.
    """

    if not project_text:
        return False

    strong_matches = _count_matches(
        project_text,
        AI_STRONG_SIGNALS,
    )

    return strong_matches >= 1


def _project_is_classic_ml_only(project_text: str) -> bool:
    """
    Returns True when the text looks like classic ML/deep
    learning and does not contain a strong LLM/RAG/agentic
    signal.
    """

    if not project_text:
        return False

    classic_matches = _count_matches(
        project_text,
        CLASSIC_ML_ONLY_SIGNALS,
    )

    ai_matches = _count_matches(
        project_text,
        AI_STRONG_SIGNALS,
    )

    return classic_matches >= 1 and ai_matches == 0


def _project_is_thin_wrapper(project_text: str) -> bool:
    """
    Detect shallow LLM/API-wrapper projects.

    Examples of thin projects:
        "Built chatbot using OpenAI API"
        "Created GPT chatbot with API"
        "Used Gemini API for text generation"

    A project containing retrieval, embeddings, tools, state,
    evaluation, backend logic, database work, etc. is treated
    as deeper.
    """

    if not project_text:
        return False

    normalized = _normalize(project_text)

    api_count = _count_matches(
        normalized,
        LLM_API_ONLY_SIGNALS,
    )

    ai_provider_count = _count_matches(
        normalized,
        [
            "openai",
            "gpt",
            "claude",
            "anthropic",
            "gemini",
            "llama",
            "mistral",
        ],
    )

    depth_count = _count_matches(
        normalized,
        AI_DEPTH_SIGNALS,
    )

    business_count = _count_matches(
        normalized,
        AI_BUSINESS_LOGIC_SIGNALS,
    )

    # Explicitly short, API-centric AI projects.
    if api_count >= 1 and depth_count == 0 and business_count <= 1:
        return True

    # Provider-only project with almost no implementation detail.
    if ai_provider_count >= 1 and depth_count == 0 and business_count == 0:
        return True

    # Very short project descriptions with only an API/model mention.
    words = normalized.split()

    if (
        len(words) < 45
        and ai_provider_count >= 1
        and depth_count <= 1
        and business_count <= 1
    ):
        return True

    return False


# =========================================================
# AI PROJECT DEPTH — 40 POINTS
# =========================================================

def _score_ai_project_depth(candidate: Candidate) -> Tuple[int, Dict[str, str]]:
    """
    Score real AI/agentic/RAG project depth.

    Important:
    - Skills alone do NOT receive the score.
    - Project/experience evidence is prioritized.
    - Classic ML alone receives zero.
    """

    project_text = "\n".join(
        [
            candidate.projects or "",
            candidate.experience or "",
        ]
    )

    if not project_text.strip():
        return 0, {
            "reason": "No project or experience evidence available."
        }

    if _project_is_classic_ml_only(project_text):
        return 0, {
            "reason": (
                "Project evidence is classic ML/deep learning only; "
                "no qualifying LLM/RAG/agentic depth found."
            )
        }

    strong_matches = _matched_keywords(
        project_text,
        AI_STRONG_SIGNALS,
    )

    depth_matches = _matched_keywords(
        project_text,
        AI_DEPTH_SIGNALS,
    )

    business_matches = _matched_keywords(
        project_text,
        AI_BUSINESS_LOGIC_SIGNALS,
    )

    if not strong_matches:
        return 0, {
            "reason": "No qualifying LLM/RAG/agentic project evidence."
        }

    # -----------------------------------------------------
    # Base score for genuine AI evidence
    # -----------------------------------------------------

    score = 10

    # Multiple AI signals indicate broader implementation.
    score += min(10, len(strong_matches) * 2)

    # Real implementation depth.
    score += min(12, len(depth_matches) * 2)

    # Product/backend/data-processing logic.
    score += min(8, len(business_matches) * 2)

    # -----------------------------------------------------
    # Thin wrapper penalty
    # -----------------------------------------------------

    thin_wrapper = _project_is_thin_wrapper(project_text)

    if thin_wrapper:
        score -= 10

    score = max(0, min(40, score))

    return score, {
        "strong_ai_signals": ", ".join(strong_matches[:12]) or "None",
        "depth_signals": ", ".join(depth_matches[:12]) or "None",
        "business_logic_signals": ", ".join(business_matches[:10]) or "None",
        "thin_wrapper_penalty": "-10" if thin_wrapper else "0",
        "summary": (
            f"{len(strong_matches)} qualifying AI signals, "
            f"{len(depth_matches)} depth signals, "
            f"{len(business_matches)} implementation signals."
        ),
    }


# =========================================================
# PYTHON + BACKEND — 30 POINTS
# =========================================================

def _score_python_backend(candidate: Candidate) -> Tuple[int, Dict[str, str]]:
    """
    Reward Python/backend evidence primarily from projects
    and experience rather than a skills-only list.
    """

    project_text = "\n".join(
        [
            candidate.projects or "",
            candidate.experience or "",
        ]
    )

    skills_text = ", ".join(candidate.skills or [])

    project_python = _has_python(project_text)
    skills_python = _has_python(skills_text)

    score = 0

    if project_python:
        score += 12
    elif skills_python:
        score += 7

    project_matches = _matched_keywords(
        project_text,
        PYTHON_BACKEND_SIGNALS,
    )

    skills_matches = _matched_keywords(
        skills_text,
        PYTHON_BACKEND_SIGNALS,
    )

    # Backend/project evidence gets stronger weight.
    for keyword in project_matches:
        if keyword == "python":
            continue

        if keyword in {"fastapi", "flask", "django"}:
            score += 4

        elif keyword in {
            "postgresql",
            "postgres",
            "redis",
            "celery",
            "asyncio",
            "async programming",
            "asynchronous",
        }:
            score += 3

        else:
            score += 2

    # Skills are supporting evidence only.
    skill_only_count = 0

    for keyword in skills_matches:
        if keyword in project_matches:
            continue

        if keyword == "python":
            continue

        score += 1
        skill_only_count += 1

    score = min(30, score)

    return score, {
        "python_in_project_or_experience": str(project_python),
        "python_in_skills": str(skills_python),
        "project_backend_signals": ", ".join(project_matches[:12]) or "None",
        "skill_backend_signals": ", ".join(skills_matches[:12]) or "None",
        "skill_only_supporting_signals": str(skill_only_count),
    }


# =========================================================
# CLOUD / DEPLOYMENT / FULL STACK — 15 POINTS
# =========================================================

def _score_cloud_fullstack(candidate: Candidate) -> Tuple[int, Dict[str, str]]:
    project_text = "\n".join(
        [
            candidate.projects or "",
            candidate.experience or "",
        ]
    )

    deployment_matches = _matched_keywords(
        project_text,
        CLOUD_DEPLOYMENT_SIGNALS,
    )

    fullstack_matches = _matched_keywords(
        project_text,
        FULLSTACK_SIGNALS,
    )

    score = 0

    # Strong deployment/cloud signals.
    strong_deployment = {
        "gcp",
        "google cloud",
        "google cloud platform",
        "aws",
        "amazon web services",
        "azure",
        "docker",
        "kubernetes",
        "k8s",
        "deployment",
        "deployed",
        "ci/cd",
        "cicd",
        "github actions",
        "jenkins",
        "terraform",
    }

    for keyword in deployment_matches:
        if keyword in strong_deployment:
            score += 3
        else:
            score += 2

    # React/Next/full-stack are supporting signals.
    score += min(4, len(fullstack_matches))

    score = min(15, score)

    return score, {
        "deployment_signals": ", ".join(deployment_matches[:12]) or "None",
        "fullstack_signals": ", ".join(fullstack_matches[:8]) or "None",
    }


# =========================================================
# ENGINEERING DEPTH — 5 POINTS
# =========================================================

def _score_engineering_depth(candidate: Candidate) -> Tuple[int, Dict[str, str]]:
    project_text = "\n".join(
        [
            candidate.projects or "",
            candidate.experience or "",
        ]
    )

    matches = _matched_keywords(
        project_text,
        ENGINEERING_DEPTH_SIGNALS,
    )

    score = min(5, len(matches))

    return score, {
        "signals": ", ".join(matches[:10]) or "None",
    }


# =========================================================
# DETERMINISTIC SCORE
# =========================================================

def deterministic_score(
    candidate: Candidate,
    github: GitHubResult,
) -> Tuple[ScoreBreakdown, Dict[str, str]]:

    ai_score, ai_evidence = _score_ai_project_depth(candidate)

    python_score, python_evidence = _score_python_backend(candidate)

    cloud_score, cloud_evidence = _score_cloud_fullstack(candidate)

    engineering_score, engineering_evidence = _score_engineering_depth(
        candidate
    )

    github_score = max(0, min(10, int(github.points)))

    breakdown = ScoreBreakdown(
        ai_project_depth=ai_score,
        python_backend=python_score,
        cloud_fullstack=cloud_score,
        github=github_score,
        engineering_depth=engineering_score,
    )

    evidence: Dict[str, str] = {
        "ai_project_depth": (
            ai_evidence.get("summary")
            or ai_evidence.get("reason")
            or ""
        ),
        "ai_strong_signals": ai_evidence.get(
            "strong_ai_signals",
            "",
        ),
        "ai_depth_signals": ai_evidence.get(
            "depth_signals",
            "",
        ),
        "ai_business_logic_signals": ai_evidence.get(
            "business_logic_signals",
            "",
        ),
        "thin_wrapper_penalty": ai_evidence.get(
            "thin_wrapper_penalty",
            "0",
        ),
        "python_backend": (
            f"Python in project/experience="
            f"{python_evidence['python_in_project_or_experience']}; "
            f"backend signals="
            f"{python_evidence['project_backend_signals']}; "
            f"supporting skills="
            f"{python_evidence['skill_backend_signals']}"
        ),
        "cloud_fullstack": (
            f"Deployment/cloud="
            f"{cloud_evidence['deployment_signals']}; "
            f"full-stack="
            f"{cloud_evidence['fullstack_signals']}"
        ),
        "github": github.summary,
        "engineering_depth": (
            f"Engineering signals="
            f"{engineering_evidence['signals']}"
        ),
    }

    return breakdown, evidence


# =========================================================
# UTILITY
# =========================================================

def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(
        minimum,
        min(maximum, int(value)),
    )


# =========================================================
# LLM RESULT APPLICATION
# =========================================================

def apply_llm_analysis(
    analysis: ResumeAnalysis,
    github: GitHubResult,
) -> Tuple[ScoreBreakdown, Dict[str, str]]:
    """
    Apply structured LLM scoring while keeping all score
    categories within the assignment limits.

    GitHub remains deterministic and is never controlled by
    the LLM.
    """

    ai = clamp(
        analysis.ai_project_depth_score,
        0,
        40,
    )

    py = clamp(
        analysis.python_backend_score,
        0,
        30,
    )

    cloud = clamp(
        analysis.cloud_fullstack_score,
        0,
        15,
    )

    eng = clamp(
        analysis.engineering_depth_score,
        0,
        5,
    )

    concern = ""

    # -----------------------------------------------------
    # Thin-wrapper penalty
    # -----------------------------------------------------

    if analysis.is_thin_wrapper:
        penalty = 10

        ai = max(
            0,
            ai - penalty,
        )

        concern = (
            f"Thin LLM/API-wrapper penalty applied: "
            f"-{penalty} points."
        )

    github_points = clamp(
        github.points,
        0,
        10,
    )

    breakdown = ScoreBreakdown(
        ai_project_depth=ai,
        python_backend=py,
        cloud_fullstack=cloud,
        github=github_points,
        engineering_depth=eng,
    )

    evidence = dict(
        analysis.evidence or {}
    )

    if concern:
        evidence["quality_penalty"] = concern

    evidence["github"] = github.summary

    return breakdown, evidence