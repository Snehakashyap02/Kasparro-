
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader

from src.models import Candidate

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTION_ALIASES: Dict[str, List[str]] = {
    "summary": [
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "objective",
        "career objective",
    ],
    "skills": [
        "skills",
        "technical skills",
        "technical skill",
        "core skills",
        "skills & technologies",
        "technologies",
        "technical expertise",
        "key skills",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "internship experience",
    ],
    "projects": [
        "projects",
        "project",
        "academic projects",
        "personal projects",
        "key projects",
        "technical projects",
    ],
    "education": [
        "education",
        "educational background",
        "academic background",
        "qualifications",
        "academic qualifications",
    ],
    "coursework": [
        "coursework",
        "relevant coursework",
        "courses",
        "relevant courses",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses & certifications",
    ],
    "achievements": [
        "achievements",
        "awards",
        "accomplishments",
    ],
}

SECTION_ORDER = [
    "summary",
    "skills",
    "experience",
    "projects",
    "education",
    "coursework",
    "certifications",
    "achievements",
]


# Common things which are almost never candidate names.
NAME_EXCLUSION_WORDS = {
    "resume",
    "curriculum",
    "vitae",
    "cv",
    "summary",
    "profile",
    "objective",
    "education",
    "experience",
    "projects",
    "project",
    "skills",
    "technical",
    "languages",
    "language",
    "technologies",
    "technology",
    "coursework",
    "certifications",
    "certification",
    "achievements",
    "references",
    "contact",
    "details",
    "linkedin",
    "linkedincom",
    "github",
    "portfolio",
    "developer",
    "engineer",
    "student",
    "phone",
    "mobile",
    "email",
    "address",
    "source",
    "contributions",
    "contribution",
    "the",
    "production",
    "handwritten",
    "digit",
    "recognition",
    "involvement",
    "interest",
    "languages",
    "programming",
    "backend",
    "frontend",
    "frameworks",
    "libraries",
    "databases",
    "engineering",
    "tools",
    "platforms",
    "devops",
    "full-stack",
    "fullstack",
    "touch",
}

LOCATION_WORDS = {
    "india",
    "indian",
    "delhi",
    "new delhi",
    "bengaluru",
    "bangalore",
    "mumbai",
    "kolkata",
    "bhubaneswar",
    "hyderabad",
    "pune",
    "chennai",
    "ranchi",
    "noida",
    "gurgaon",
    "gurugram",
    "lucknow",
    "jaipur",
    "ahmedabad",
    "surat",
    "kanpur",
    "patna",
    "dehradun",
    "mathura",
    "belagavi",
    "belgaum",
    "karnataka",
    "uttar pradesh",
    "west bengal",
    "maharashtra",
    "odisha",
    "jharkhand",
    "tamil nadu",
    "telangana",
    "gujarat",
    "rajasthan",
    "indore",
    "mysore",
    "mangalore",
    "kochi",
    "cochin",
    "thiruvananthapuram",
    "visakhapatnam",
    "vizag",
    "coimbatore",
    "madurai",
    "nashik",
    "nagpur",
    "vadodara",
    "rajkot",
    "jodhpur",
    "udaipur",
    "agra",
    "varanasi",
    "prayagraj",
    "allahabad",
    "meerut",
    "ghaziabad",
    "faridabad",
    "amritsar",
    "ludhiana",
    "chandigarh",
    "guwahati",
    "siliguri",
    "durgapur",
    "rourkela",
    "cuttack",
    "hubli",
    "dharwad",
    "davangere",
    "tumkur",
    "tumakuru",
    "shivamogga",
    "udupi",
    "ballari",
    "gokak",
}

NAME_CATEGORY_WORDS = {
    "area",
    "awards",
    "certification",
    "certifications",
    "coursework",
    "data",
    "database",
    "education",
    "experience",
    "framework",
    "frameworks",
    "get in touch",
    "linked in",
    "linkedin",
    "github",
    "github profile",
    "portfolio",
    "involvement",
    "language",
    "languages",
    "skills",
    "tools",
}

COMPANY_WORDS = {
    "company",
    "corporation",
    "corp",
    "inc",
    "llc",
    "llp",
    "limited",
    "ltd",
    "pvt",
    "private",
    "technologies",
    "technology",
    "solutions",
    "systems",
    "consulting",
    "services",
    "software",
    "labs",
    "laboratories",
    "university",
    "college",
    "institute",
    "school",
    "enterprise",
    "enterprises",
}

KNOWN_ORGANIZATIONS = {
    "accenture",
    "adobe",
    "amazon",
    "byjus",
    "capgemini",
    "cognizant",
    "deloitte",
    "facebook",
    "flipkart",
    "freshworks",
    "google",
    "hcl",
    "hcltech",
    "ibm",
    "infosys",
    "linkedin",
    "microsoft",
    "meta",
    "nvidia",
    "oracle",
    "razorpay",
    "salesforce",
    "swiggy",
    "tata",
    "tcs",
    "uber",
    "wipro",
    "zoho",
    "zomato",
}

# Canonical forms reduce noisy duplicate skills.
SKILL_CANONICAL_MAP = {
    "embedding": "Embeddings",
    "embeddings": "Embeddings",
    "vector db": "Vector Database",
    "vector database": "Vector Database",
    "vector store": "Vector Store",
    "vector stores": "Vector Store",
    "chromadb": "ChromaDB",
    "chroma": "Chroma",
    "lang graph": "LangGraph",
    "langgraph": "LangGraph",
    "lang chain": "LangChain",
    "langchain": "LangChain",
    "llama index": "LlamaIndex",
    "llamaindex": "LlamaIndex",
    "openai api": "OpenAI API",
    "openai": "OpenAI",
    "gpt": "GPT",
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "gemini": "Gemini",
    "claude": "Claude",
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "react": "React",
    "fast api": "FastAPI",
    "fastapi": "FastAPI",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo db": "MongoDB",
    "mongodb": "MongoDB",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "rag": "RAG",
    "llm": "LLM",
    "llms": "LLM",
    "large language models": "LLM",
    "react.js": "React",
    "reactjs": "React",
    "rest apis": "REST API",
    "restful api": "REST API",
    "restful apis": "REST API",
    "nodejs": "Node.js",
    "next js": "Next.js",
    "postgres": "PostgreSQL",
    "multi-agent": "AI Agents",
    "multi agent": "AI Agents",
    "agentic ai": "AI Agents",
    "ai agent": "AI Agents",
    "ai agents": "AI Agents",
}


KNOWN_SKILLS = [
    "Python",
    "C",
    "C++",
    "Java",
    "JavaScript",
    "TypeScript",
    "SQL",
    "HTML",
    "CSS",
    "React",
    "Next.js",
    "Node.js",
    "FastAPI",
    "Flask",
    "Django",
    "REST API",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "SQLite",
    "Redis",
    "Docker",
    "Kubernetes",
    "AWS",
    "GCP",
    "Google Cloud",
    "Azure",
    "LangChain",
    "LangGraph",
    "LlamaIndex",
    "RAG",
    "Embeddings",
    "Vector Database",
    "Vector Store",
    "FAISS",
    "Chroma",
    "ChromaDB",
    "Pinecone",
    "OpenAI",
    "OpenAI API",
    "GPT",
    "GPT-4",
    "GPT-4o",
    "Gemini",
    "Claude",
    "Llama",
    "Mistral",
    "Hugging Face",
    "Transformers",
    "Agentic AI",
    "AI Agents",
    "Multi-Agent",
    "Tool Calling",
    "Function Calling",
    "Prompt Engineering",
    "Machine Learning",
    "Deep Learning",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "XGBoost",
    "LightGBM",
    "Random Forest",
    "CNN",
    "RNN",
    "LSTM",
    "Git",
    "GitHub",
    "pytest",
    "React.js",
    "ReactJS",
    "Lang Chain",
    "Fast API",
    "REST APIs",
    "RESTful API",
    "Postgres",
    "Embedding",
    "LLM",
    "LLMs",
    "Large Language Models",
    "NodeJS",
    "Next JS",
    "CrewAI",
    "AutoGen",
    "MCP",
    "LiteLLM",
    "Haystack",
    "Ollama",
    "Qdrant",
    "Weaviate",
    "LlamaParse",
    "Prisma",
    "QStash",
    "NestJS",
    "Express.js",
    "SQLAlchemy",
    "Celery",
    "Kafka",
    "RabbitMQ",
    "Pandas",
    "NumPy",
    "Tailwind CSS",
    "React Native",
    "Streamlit",
    "Gradio",
    "Jenkins",
    "Terraform",
    "Prometheus",
    "Grafana",
    "OpenTelemetry",
    "GitHub Actions",
    "WebSockets",
    "WebRTC",
    "SSE",
    "Postman",
    "Linux",
]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_line(line: str) -> str:
    """Normalize a single extracted PDF line."""
    line = line.replace("\u00a0", " ")
    line = line.replace("\u200b", "")
    line = line.replace("\ufeff", "")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip(" \t|•·")


def normalize_text(text: str) -> str:
    """Normalize complete extracted text while preserving useful line breaks."""
    lines = []

    for raw_line in text.splitlines():
        line = clean_line(raw_line)

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    reader = PdfReader(str(pdf_path))

    page_text: List[str] = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            LOGGER.warning(
                "Could not extract one page from %s: %s",
                pdf_path.name,
                exc,
            )
            text = ""

        if text:
            page_text.append(text)

    return normalize_text("\n".join(page_text))


# ---------------------------------------------------------------------------
# Contact extraction
# ---------------------------------------------------------------------------

def extract_email(text: str) -> Optional[str]:
    match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text,
    )

    return match.group(0).strip() if match else None


def extract_github(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract GitHub URL and username."""
    url_match = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/"
        r"([A-Za-z0-9][A-Za-z0-9-]{0,38})"
        r"(?:[/?#\s]|$)",
        text,
        flags=re.IGNORECASE,
    )

    if not url_match:
        return None, None

    username = url_match.group(1).strip()

    url = f"https://github.com/{username}"

    return url, username


def extract_phone(text: str) -> Optional[str]:
    """Best-effort phone extraction."""
    patterns = [
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        r"(?<!\d)(?:\+\d{1,3}[\s-]?)?\d{10}(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0).strip()

    return None


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _normalized_heading(text: str) -> str:
    text = text.lower().strip()

    text = re.sub(r"[:|]+$", "", text)
    text = re.sub(r"[^a-z0-9+#&./ -]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


_HEADING_TO_SECTION: Dict[str, str] = {}

for section, aliases in SECTION_ALIASES.items():
    for alias in aliases:
        _HEADING_TO_SECTION[_normalized_heading(alias)] = section


def detect_section_heading(line: str) -> Optional[str]:
    """
    Return normalized section name when a line looks like a section heading.
    """
    normalized = _normalized_heading(line)

    if not normalized:
        return None

    # Direct match.
    if normalized in _HEADING_TO_SECTION:
        return _HEADING_TO_SECTION[normalized]

    # Handle headings such as "TECHNICAL SKILLS:"
    words = normalized.split()

    if len(words) <= 5:
        for alias, section in _HEADING_TO_SECTION.items():
            if normalized == alias:
                return section

    return None


def split_sections(text: str) -> Dict[str, str]:
    """Split resume into logical sections."""
    sections: Dict[str, List[str]] = {}
    current_section: Optional[str] = None

    for raw_line in text.splitlines():
        line = clean_line(raw_line)

        if not line:
            continue

        detected = detect_section_heading(line)

        if detected:
            current_section = detected
            sections.setdefault(current_section, [])
            continue

        if current_section:
            sections.setdefault(current_section, []).append(line)

    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
        if lines
    }


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------

def _looks_like_email(line: str) -> bool:
    return bool(
        re.search(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            line,
        )
    )


def _looks_like_url(line: str) -> bool:
    return bool(
        re.search(
            r"(https?://|www\.|linkedin\.com|github\.com|"
            r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b|@)",
            line,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_phone(line: str) -> bool:
    digits = re.sub(r"\D", "", line)

    return len(digits) >= 10


def _is_location_line(line: str) -> bool:
    normalized = re.sub(r"[^a-z ]", " ", line.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Explicit location punctuation/patterns.
    if re.search(
        r"\b(india|karnataka|odisha|jharkhand|"
        r"uttar pradesh|west bengal|maharashtra|"
        r"tamil nadu|telangana|gujarat|rajasthan)\b",
        normalized,
    ):
        return True

    # Most address/location lines contain a comma and a known location word.
    if "," in line:
        for location in LOCATION_WORDS:
            if location in normalized:
                return True

    return normalized in LOCATION_WORDS


def _looks_like_heading(line: str) -> bool:
    normalized = _normalized_heading(line)

    if normalized in _HEADING_TO_SECTION:
        return True

    upper_words = sum(
        1 for word in line.split()
        if word.isupper() and len(word) >= 2
    )

    if upper_words >= 2 and len(line.split()) <= 5:
        # Do not automatically reject every all-caps name.
        if any(
            word.lower() in NAME_EXCLUSION_WORDS
            for word in line.split()
        ):
            return True

    return False


def _clean_possible_name(line: str) -> str:
    """Remove obvious contact prefixes/suffixes from a possible name."""
    line = (
        line.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    line = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", line)
    name = line.strip()

    # Resume headers often place the name, location, and contact links on
    # one PDF-extracted line. Keep the leading name segment.
    name = re.split(r"\s*[|•]+\s*", name, maxsplit=1)[0]

    # Remove common labels.
    name = re.sub(
        r"^(name|candidate name|full name)\s*[:\-]\s*",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Remove bullets.
    name = name.strip("•·|:- ")

    # Remove trailing contact information.
    name = re.split(
        r"\s+(?:email|phone|mobile|linkedin|github)\s*[:\-]?",
        name,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Remove emails.
    name = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "",
        name,
    )

    # Remove phone numbers.
    name = re.sub(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        "",
        name,
    )

    name = re.sub(r"\s+", " ", name).strip(" ,|-")

    for location in sorted(LOCATION_WORDS, key=len, reverse=True):
        match = re.search(
            rf"(?<![A-Za-z]){re.escape(location)}(?![A-Za-z])",
            name,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        prefix = name[:match.start()].strip(" ,|-")
        suffix = name[match.end():].strip(" ,|-")

        if prefix and (
            not suffix
            or re.fullmatch(r"[A-Z]{2}", suffix, flags=re.IGNORECASE)
        ):
            name = prefix
            break

    return name


def _name_score(line: str, index: int) -> float:
    """
    Score how likely a line is to be a candidate name.

    Higher = more likely.
    """
    cleaned = _clean_possible_name(line)

    if not _is_safe_name_text(cleaned):
        return -1000

    # Remove obvious punctuation-heavy lines.
    punctuation_ratio = sum(
        1 for char in cleaned
        if not char.isalnum() and not char.isspace()
    ) / max(len(cleaned), 1)

    if punctuation_ratio > 0.25:
        return -100

    words = cleaned.split()

    # Candidate names are generally 2-4 words.
    if len(words) == 2:
        score = 80
    elif len(words) == 3:
        score = 82
    elif len(words) == 4:
        score = 68
    elif len(words) == 1:
        score = 0
    else:
        score = 5

    # Strong preference for lines made mostly of alphabetic tokens.
    valid_word_count = 0

    for word in words:
        token = re.sub(r"[^A-Za-z'-]", "", word)

        if 1 < len(token) <= 30:
            valid_word_count += 1

    if valid_word_count == len(words):
        score += 15
    else:
        score -= 20

    # Names frequently use title case or all caps.
    if cleaned.isupper():
        score += 4
    elif all(
        word[:1].isupper()
        for word in words
        if word and word[0].isalpha()
    ):
        score += 8

    # Earlier lines are more likely to be names,
    # but do not overpower content-based scoring.
    score += max(0, 18 - index * 2)

    return score


def _is_safe_name_text(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 60:
        return False

    if (
        _looks_like_email(name)
        or _looks_like_url(name)
        or _looks_like_phone(name)
        or _is_location_line(name)
        or _looks_like_heading(name)
    ):
        return False

    normalized = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    words = set(normalized.split())

    if normalized in NAME_CATEGORY_WORDS:
        return False

    if words.intersection(NAME_EXCLUSION_WORDS | COMPANY_WORDS):
        return False

    if words.intersection(KNOWN_ORGANIZATIONS):
        return False

    if any(
        re.search(
            rf"(?<![A-Za-z0-9+#]){re.escape(skill)}(?![A-Za-z0-9+#])",
            name,
            flags=re.IGNORECASE,
        )
        for skill in KNOWN_SKILLS
    ):
        return False

    return all(
        re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", word) is not None
        for word in name.split()
    )


def _extract_name_from_contact_context(lines: List[str]) -> Optional[str]:
    """
    Look near email / phone / LinkedIn contact information.
    Some resumes place the name on the line immediately above the contacts.
    """
    header_end = min(len(lines), 40)

    for index, line in enumerate(lines[:header_end]):
        if detect_section_heading(line):
            header_end = index
            break

    if header_end == 0:
        return None

    contact_indices = []

    for index, line in enumerate(lines[:header_end]):
        if (
            _looks_like_email(line)
            or _looks_like_phone(line)
            or _looks_like_url(line)
        ):
            contact_indices.append(index)

    candidates: List[Tuple[float, str]] = []

    for contact_index in contact_indices:
        start = max(0, contact_index - 5)

        for first_index in range(start, contact_index + 1):
            for last_index in range(first_index, min(contact_index, first_index + 1) + 1):
                parts = [
                    _clean_possible_name(lines[index])
                    for index in range(first_index, last_index + 1)
                ]

                if not all(_is_safe_name_text(part) for part in parts):
                    continue

                cleaned = " ".join(parts)

                if len(cleaned.split()) < 2:
                    continue

                score = _name_score(cleaned, first_index)
                distance = contact_index - last_index
                score += max(0, 24 - distance * 8)
                candidates.append((score, cleaned))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda item: item[0])

    best_score, best_name = candidates[0]

    return best_name if best_score >= 70 else None


def extract_candidate_name(
    text: str,
    source_file: Optional[str] = None,
) -> str:
    """
    Robust candidate name extraction.

    Priority:
    1. Name adjacent to email/phone/contact block.
    2. Strong candidate among first ~25 lines.
    3. Reasonable filename fallback.
    """
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return _filename_name_fallback(source_file)

    # An explicit name label is the strongest available signal.
    for line in lines[:20]:
        match = re.match(
            r"^(?:(?:candidate|full)\s+)?name\s*[:\-]\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )

        if match:
            explicit_name = _clean_possible_name(match.group(1))

            if _is_safe_name_text(explicit_name):
                return explicit_name

    # First, use contact block context.
    contact_name = _extract_name_from_contact_context(lines)

    if contact_name:
        return contact_name

    # Score early lines.
    candidates: List[Tuple[float, str]] = []

    for index, line in enumerate(lines[:30]):
        score = _name_score(line, index)

        if score <= 0:
            continue

        cleaned = _clean_possible_name(line)

        if cleaned:
            candidates.append((score, cleaned))

    if candidates:
        candidates.sort(reverse=True, key=lambda item: item[0])
        best_score, best_name = candidates[0]

        # Only trust name extraction when confidence is reasonable.
        if best_score >= 70:
            return best_name

    return _filename_name_fallback(source_file)


def _filename_name_fallback(source_file: Optional[str]) -> str:
    """
    Fallback for resumes where the PDF does not expose a readable name.
    """
    if not source_file:
        return "Unknown Candidate"

    name = Path(source_file).stem

    # Candidate_13 -> Candidate 13
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name.title() or "Unknown Candidate"


# ---------------------------------------------------------------------------
# Skills extraction
# ---------------------------------------------------------------------------

def _canonicalize_skill(skill: str) -> str:
    raw = skill.strip()

    key = raw.lower().strip()

    if key in SKILL_CANONICAL_MAP:
        return SKILL_CANONICAL_MAP[key]

    return raw


def _skill_key(skill: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", "", skill.lower())


def extract_skills(text: str, sections: Dict[str, str]) -> List[str]:
    """
    Extract recognized technologies from relevant sections only.

    Unknown comma-separated fragments are deliberately omitted rather than
    being treated as skills.
    """
    relevant_sections = [
        sections.get(section, "")
        for section in ("skills", "projects", "experience", "summary")
    ]
    search_text = "\n".join(
        section for section in relevant_sections if section.strip()
    )

    # For resumes without recognizable headings, retain a conservative
    # technology scan over the available text.
    if not search_text:
        search_text = text

    found: List[str] = []
    seen = set()

    for skill in KNOWN_SKILLS:
        if skill == "C" and not sections.get("skills"):
            continue

        pattern = re.escape(skill)

        if re.search(
            rf"(?<![A-Za-z0-9+#]){pattern}(?![A-Za-z0-9+#])",
            search_text,
            flags=re.IGNORECASE,
        ):
            canonical = _canonicalize_skill(skill)
            key = _skill_key(canonical)

            if key not in seen:
                seen.add(key)
                found.append(canonical)

    return found


# ---------------------------------------------------------------------------
# Section content helpers
# ---------------------------------------------------------------------------

def _section_or_empty(
    sections: Dict[str, str],
    *names: str,
) -> str:
    for name in names:
        value = sections.get(name, "")

        if value.strip():
            return value.strip()

    return ""


def build_project_text(
    sections: Dict[str, str],
    full_text: str,
) -> str:
    projects = _section_or_empty(
        sections,
        "projects",
    )

    if projects:
        return projects

    # Some resumes do not have a Projects heading but mention projects
    # in experience/summary.
    experience = sections.get("experience", "")

    if experience:
        project_keywords = re.compile(
            r"\b(project|developed|built|implemented|created|"
            r"designed|application|system|platform)\b",
            re.IGNORECASE,
        )

        relevant_lines = [
            line.strip()
            for line in experience.splitlines()
            if project_keywords.search(line)
        ]

        if relevant_lines:
            return "\n".join(relevant_lines)

    # Last resort: do not make the entire resume the project section.
    return ""


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def extract_candidate(pdf_path: Path) -> Candidate:
    """
    Extract one resume into the Candidate Pydantic model.
    """
    full_text = extract_pdf_text(pdf_path)

    return extract_candidate_from_text(pdf_path.name, full_text)


def extract_candidate_from_text(source_file: str, full_text: str) -> Candidate:
    """Build a candidate from text already extracted from a resume."""
    if not full_text:
        raise ValueError(
            f"No readable text found in {source_file}"
        )

    sections = split_sections(full_text)

    candidate_name = extract_candidate_name(
        full_text,
        source_file=source_file,
    )

    email = extract_email(full_text)

    github_url, github_username = extract_github(full_text)

    skills = extract_skills(
        full_text,
        sections,
    )

    projects = build_project_text(
        sections,
        full_text,
    )

    experience = _section_or_empty(
        sections,
        "experience",
    )

    education = _section_or_empty(
        sections,
        "education",
    )

    coursework = _section_or_empty(
        sections,
        "coursework",
    )

    # Improve extraction for resumes where sections were weakly detected.
    if not projects:
        projects = _find_projects_from_text(full_text)

    if not experience:
        experience = _find_experience_from_text(full_text)

    return Candidate(
        source_file=source_file,
        candidate_name=candidate_name,
        email=email,
        github_url=github_url,
        github_username=github_username,
        full_text=full_text,
        sections=sections,
        skills=skills,
        projects=projects,
        experience=experience,
        education=education,
        coursework=coursework,
    )


def _find_projects_from_text(text: str) -> str:
    """
    Lightweight fallback for resumes where a PROJECTS heading
    was not extracted correctly.
    """
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    project_lines: List[str] = []

    keywords = re.compile(
        r"\b("
        r"project|developed|built|implemented|designed|created|"
        r"application|platform|system|dashboard|website|"
        r"assistant|chatbot|pipeline"
        r")\b",
        re.IGNORECASE,
    )

    for line in lines:
        if keywords.search(line):
            project_lines.append(line)

    return "\n".join(project_lines[:50])


def _find_experience_from_text(text: str) -> str:
    """
    Lightweight fallback for resumes where an EXPERIENCE heading
    was not extracted correctly.
    """
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    experience_lines: List[str] = []

    keywords = re.compile(
        r"\b("
        r"intern|internship|engineer|developer|software engineer|"
        r"worked|employment|experience|trainee|associate"
        r")\b",
        re.IGNORECASE,
    )

    for line in lines:
        if keywords.search(line):
            experience_lines.append(line)

    return "\n".join(experience_lines[:50])