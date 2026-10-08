import os

from dotenv import load_dotenv

load_dotenv()

AI_PROJECT_WEIGHT = 40
PYTHON_BACKEND_WEIGHT = 30
CLOUD_FULLSTACK_WEIGHT = 15
GITHUB_WEIGHT = 10
ENGINEERING_DEPTH_WEIGHT = 5

THIN_WRAPPER_PENALTY_MIN = 5
THIN_WRAPPER_PENALTY_MAX = 15

AI_KEYWORDS = [
    "langchain",
    "langgraph",
    "llamaindex",
    "llama index",
    "google adk",
    "rag",
    "retrieval augmented generation",
    "embeddings",
    "embedding",
    "vector search",
    "vector database",
    "vector db",
    "tool calling",
    "tool-calling",
    "agents",
    "agentic",
    "multi-agent",
    "multi agent",
    "llm",
    "openai",
    "gemini",
    "claude",
    "evaluation pipeline",
    "evaluation pipelines",
]

PYTHON_KEYWORD = "python"

BACKEND_KEYWORDS = [
    "fastapi",
    "flask",
    "django",
    "asyncio",
    "async",
    "postgresql",
    "postgres",
    "redis",
    "rest api",
    "api",
]

CLOUD_FULLSTACK_KEYWORDS = [
    "docker",
    "gcp",
    "google cloud",
    "aws",
    "azure",
    "kubernetes",
    "react",
    "next.js",
    "nextjs",
    "full stack",
    "full-stack",
    "deployment",
]

ENGINEERING_KEYWORDS = [
    "pytest",
    "testing",
    "unit test",
    "architecture",
    "caching",
    "cache",
    "queue",
    "queues",
    "observability",
    "logging",
    "monitoring",
    "concurrency",
    "failure handling",
    "retry",
    "retries",
]

PROJECT_QUALITY_KEYWORDS = [
    "retrieval",
    "chunking",
    "reranking",
    "reranker",
    "vector",
    "state",
    "orchestration",
    "workflow",
    "tool",
    "evaluation",
    "evaluation pipeline",
    "agent",
    "multi-agent",
    "memory",
    "backend",
    "database",
]

LLM_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://openrouter.ai/api/v1",
)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

GITHUB_RECENT_DAYS = 90
REQUEST_TIMEOUT_SECONDS = 10
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2
