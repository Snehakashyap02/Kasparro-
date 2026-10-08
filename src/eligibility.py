import re
from typing import Iterable, List, Tuple

from src.models import Candidate, EligibilityResult


# ---------------------------------------------------------------------------
# Qualifying AI / LLM / Agentic signals
# ---------------------------------------------------------------------------

AI_KEYWORDS = [
    "large language model",
    "large language models",
    "llm",
    "llms",
    "generative ai",
    "genai",

    "rag",
    "retrieval augmented generation",
    "retrieval-augmented generation",

    "langchain",
    "langgraph",
    "llamaindex",
    "llama index",
    "google adk",
    "agentic ai",
    "agentic",
    "ai agent",
    "ai agents",
    "multi-agent",
    "multi agent",
    "multi-agent systems",
    "tool calling",
    "tool-calling",
    "function calling",
    "function-calling",

    "embeddings",
    "embedding",
    "vector search",
    "vector database",
    "vector databases",
    "vector db",
    "vector store",
    "semantic search",
    "semantic retrieval",
    "hybrid retrieval",

    "document qa",
    "document question answering",
    "question answering",
    "context-aware",
    "context aware",

    "llama",
    "mistral",
    "claude",
    "openai",
    "gpt",
    "gemini",
    "qwen",

    "prompt engineering",
    "prompt orchestration",
    "llmops",
    "llm ops",
    "llm evaluation",
    "llm evaluations",
    "evaluation pipeline",
    "evaluation pipelines",
    "evals",

    "ai workflow",
    "ai workflows",
    "agent workflow",
    "agent workflows",
    "ai orchestration",
    "agent orchestration",

    "retrieval pipeline",
    "rag pipeline",
    "rag pipelines",
]


# ---------------------------------------------------------------------------
# Conventional/classic ML signals
# These must not count as qualifying LLM/agentic AI by themselves.
# ---------------------------------------------------------------------------

CLASSIC_ML_KEYWORDS = [
    "machine learning",
    "machine-learning",
    "deep learning",
    "deep-learning",

    "scikit-learn",
    "sklearn",

    "xgboost",
    "lightgbm",
    "catboost",

    "random forest",
    "randomforest",

    "decision tree",
    "decision trees",

    "gradient boosting",
    "linear regression",
    "logistic regression",
    "ridge regression",
    "lasso regression",

    "classification",
    "classifiication",
    "regression",
    "clustering",

    "cnn",
    "cnns",
    "convolutional neural network",
    "convolutional neural networks",

    "rnn",
    "rnns",
    "recurrent neural network",
    "recurrent neural networks",

    "lstm",
    "lstms",

    "gru",

    "neural network",
    "neural networks",

    "computer vision",
    "image classification",
    "object detection",
    "image segmentation",

    "tensorflow",
    "keras",
    "pytorch",
]


# ---------------------------------------------------------------------------
# Python signals
# ---------------------------------------------------------------------------

PYTHON_KEYWORDS = [
    "python",
    "python3",
    "python 3",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    # Normalize common PDF extraction variants.
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2011", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _contains_keyword(text: str, keyword: str) -> bool:
    """
    Boundary-aware matching.

    Short signals such as 'rag', 'gpt', 'cnn', etc. should not accidentally
    match inside unrelated words.
    """
    text = _normalize_text(text)
    keyword = _normalize_text(keyword)

    if not text or not keyword:
        return False

    escaped = re.escape(keyword)

    # Words made only of letters/numbers use word-ish boundaries.
    if re.fullmatch(r"[a-z0-9 ]+", keyword):
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        return re.search(pattern, text) is not None

    return keyword in text


def _matching_keywords(
    text: str,
    keywords: Iterable[str],
) -> List[str]:
    matches: List[str] = []

    for keyword in keywords:
        if _contains_keyword(text, keyword):
            matches.append(keyword)

    return matches


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        normalized = _normalize_text(item)

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(item)

    return result


def _snippet(
    text: str,
    keyword: str,
    max_len: int = 240,
) -> str:
    """
    Return a short evidence snippet containing the matched keyword.
    """
    if not text:
        return ""

    lower_text = text.lower()
    lower_keyword = keyword.lower()

    position = lower_text.find(lower_keyword)

    if position == -1:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned[:max_len]

    start = max(0, position - 100)
    end = min(len(text), position + len(keyword) + 140)

    snippet = text[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()

    if start > 0:
        snippet = "..." + snippet

    if end < len(text):
        snippet = snippet + "..."

    return snippet[:max_len]


def _section_text(candidate: Candidate, section_name: str) -> str:
    """
    Prefer the explicitly extracted section field, then sections dict.
    """
    value = ""

    if section_name == "projects":
        value = candidate.projects
    elif section_name == "experience":
        value = candidate.experience
    elif section_name == "skills":
        value = " ".join(candidate.skills)

    if value:
        return value

    return candidate.sections.get(section_name, "")


def _combined_qualifying_text(candidate: Candidate) -> str:
    """
    Only sections that are relevant for hard eligibility.

    Education/coursework are intentionally excluded.
    """
    projects = _section_text(candidate, "projects")
    experience = _section_text(candidate, "experience")

    return "\n".join(
        part for part in [projects, experience]
        if part and part.strip()
    )


def _python_evidence_text(candidate: Candidate) -> str:
    """
    Python can legitimately appear in skills, projects, or experience.
    """
    skills = _section_text(candidate, "skills")
    projects = _section_text(candidate, "projects")
    experience = _section_text(candidate, "experience")

    return "\n".join(
        part for part in [skills, projects, experience]
        if part and part.strip()
    )


def _ai_evidence_is_meaningful(
    project_experience_text: str,
    matches: List[str],
) -> bool:
    """
    Determine whether an AI keyword represents meaningful implementation
    evidence rather than a stray skills keyword.

    We intentionally require at least one qualifying AI signal in Projects
    or Experience. We then look for implementation context.
    """
    if not project_experience_text or not matches:
        return False

    normalized = _normalize_text(project_experience_text)

    # Strong architectural/implementation signals.
    strong_signals = [
        "built",
        "developed",
        "implemented",
        "designed",
        "architected",
        "engineered",
        "integrated",
        "deployed",
        "created",
        "orchestrated",
        "implemented",
        "using",
        "used",
        "pipeline",
        "backend",
        "system",
        "platform",
        "application",
        "assistant",
        "chatbot",
        "agent",
        "retrieval",
        "generation",
        "workflow",
        "api",
    ]

    has_implementation_context = any(
        _contains_keyword(normalized, signal)
        for signal in strong_signals
    )

    # Strong AI architecture signals are meaningful even if the prose
    # extraction is sparse.
    architecture_signals = [
        "langchain",
        "langgraph",
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "multi-agent",
        "multi agent",
        "tool calling",
        "function calling",
        "vector search",
        "vector database",
        "vector store",
        "semantic retrieval",
        "hybrid retrieval",
        "llmops",
        "evaluation pipeline",
        "evaluation pipelines",
        "agent orchestration",
        "ai orchestration",
    ]

    has_architecture_signal = any(
        _contains_keyword(normalized, signal)
        for signal in architecture_signals
    )

    # Model/provider names alone are weaker:
    # Gemini / OpenAI / Claude / GPT by themselves should not make
    # an otherwise conventional project eligible.
    provider_only = {
        "gemini",
        "openai",
        "claude",
        "gpt",
        "llama",
        "mistral",
        "qwen",
    }

    non_provider_matches = [
        item
        for item in matches
        if item.lower() not in provider_only
    ]

    if has_architecture_signal:
        return True

    if non_provider_matches and has_implementation_context:
        return True

    return False


# ---------------------------------------------------------------------------
# Main eligibility function
# ---------------------------------------------------------------------------

def check_eligibility(candidate: Candidate) -> EligibilityResult:
    """
    Apply the assignment's hard eligibility rules.

    Required:
      1. Python evidence.
      2. Meaningful AI/LLM/RAG/agentic evidence.

    Important:
      - AI evidence is NOT accepted from a standalone Skills keyword.
      - Education and Coursework are excluded.
      - Classic ML/deep learning alone does not qualify.
    """

    python_text = _python_evidence_text(candidate)
    ai_project_text = _combined_qualifying_text(candidate)

    # ------------------------------------------------------------------
    # Python detection
    # ------------------------------------------------------------------

    python_matches = _matching_keywords(
        python_text,
        PYTHON_KEYWORDS,
    )

    has_python = bool(python_matches)

    # ------------------------------------------------------------------
    # AI detection
    # ------------------------------------------------------------------

    # AI signals in Skills are useful for reporting, but do NOT satisfy
    # meaningful-AI eligibility by themselves.
    skills_text = _section_text(candidate, "skills")

    skill_ai_matches = _matching_keywords(
        skills_text,
        AI_KEYWORDS,
    )

    # AI signals must come from Projects or Experience.
    project_experience_ai_matches = _matching_keywords(
        ai_project_text,
        AI_KEYWORDS,
    )

    has_meaningful_ai = _ai_evidence_is_meaningful(
        ai_project_text,
        project_experience_ai_matches,
    )

    # ------------------------------------------------------------------
    # Classic ML detection
    # ------------------------------------------------------------------

    classic_ml_matches = _matching_keywords(
        ai_project_text,
        CLASSIC_ML_KEYWORDS,
    )

    has_classic_ml = bool(classic_ml_matches)

    # ------------------------------------------------------------------
    # Matched skills for output
    # ------------------------------------------------------------------

    matched_skills = _dedupe_preserve_order(
        candidate.skills
    )

    # Add Python explicitly when detected.
    if has_python and not any(
        _normalize_text(skill) == "python"
        for skill in matched_skills
    ):
        matched_skills.insert(0, "Python")

    # ------------------------------------------------------------------
    # Rejection reasons
    # ------------------------------------------------------------------

    rejection_reasons: List[str] = []
    evidence_snippets: List[str] = []

    if not has_python:
        rejection_reasons.append(
            "No evidence of Python in skills, projects, or experience"
        )

    if not has_meaningful_ai:
        rejection_reasons.append(
            "No meaningful AI/LLM/RAG/agentic evidence in skills, "
            "projects, or experience"
        )

    # If AI-looking keywords appear only in Skills, explain why they
    # were not enough.
    if (
        not has_meaningful_ai
        and skill_ai_matches
        and not project_experience_ai_matches
    ):
        rejection_reasons.append(
            "AI keywords were found only in the Skills section; "
            "a meaningful AI implementation/project is required"
        )

    # Classic ML should be called out explicitly when it is the only
    # substantive AI-like evidence.
    if (
        has_classic_ml
        and not has_meaningful_ai
    ):
        rejection_reasons.append(
            "Detected only classic ML/deep-learning signals "
            f"({', '.join(classic_ml_matches[:8])}); "
            "these do not qualify as LLM/RAG/agentic AI"
        )

    eligible = has_python and has_meaningful_ai

    # ------------------------------------------------------------------
    # Evidence snippets
    # ------------------------------------------------------------------

    if has_python:
        snippet = _snippet(
            python_text,
            "python",
        )

        if snippet:
            evidence_snippets.append(snippet)

    for keyword in project_experience_ai_matches[:3]:
        snippet = _snippet(
            ai_project_text,
            keyword,
        )

        if snippet:
            evidence_snippets.append(snippet)

    evidence_snippets = _dedupe_preserve_order(
        evidence_snippets
    )[:5]

    return EligibilityResult(
        eligible=eligible,
        rejection_reasons=_dedupe_preserve_order(
            rejection_reasons
        ),
        matched_skills=matched_skills[:100],
        evidence_snippets=evidence_snippets,
    )