from src.eligibility import check_eligibility
from src.models import Candidate


def make_candidate(skills="", projects="", experience="", education="", coursework=""):
    return Candidate(
        source_file="test.pdf",
        candidate_name="Test Candidate",
        full_text="\n".join([skills, projects, experience, education, coursework]),
        skills=[x.strip() for x in skills.split(",") if x.strip()],
        projects=projects,
        experience=experience,
        education=education,
        coursework=coursework,
    )


def test_react_only_rejected():
    result = check_eligibility(make_candidate(skills="React, JavaScript, Next.js"))
    assert result.eligible is False


def test_python_without_ai_rejected():
    result = check_eligibility(
        make_candidate(skills="Python, Flask", projects="Expense tracker using Flask")
    )
    assert result.eligible is False


def test_python_and_provider_name_only_are_rejected():
    result = check_eligibility(
        make_candidate(skills="Python, OpenAI")
    )
    assert result.eligible is False
    assert any("AI/LLM/RAG/agentic" in reason for reason in result.rejection_reasons)


def test_python_langchain_rag_eligible():
    result = check_eligibility(
        make_candidate(
            skills="Python, LangChain",
            projects="Built a RAG chatbot using LangChain and embeddings."
        )
    )
    assert result.eligible is True


def test_ai_only_in_coursework_rejected():
    result = check_eligibility(
        make_candidate(
            skills="Python, Flask",
            coursework="Artificial Intelligence, LLMs, LangChain"
        )
    )
    assert result.eligible is False


def test_python_react_langgraph_eligible():
    result = check_eligibility(
        make_candidate(
            skills="Python, React, LangGraph",
            projects="Built a LangGraph agentic workflow with Python."
        )
    )
    assert result.eligible is True
    assert {"Python", "React", "LangGraph"}.issubset(result.matched_skills)


def test_matched_skills_exclude_project_keyword_duplicates():
    result = check_eligibility(
        make_candidate(
            skills="Python, RAG, Embeddings, LangGraph",
            projects=(
                "Built a LangGraph RAG workflow using vector embeddings."
            ),
        )
    )

    assert result.eligible is True
    assert result.matched_skills == ["Python", "RAG", "Embeddings", "LangGraph"]
