from src.extract import (
    extract_candidate_from_text,
    extract_candidate_name,
    extract_skills,
    split_sections,
)


def test_name_extraction_skips_location_line():
    text = "Bengaluru, India\nJane Doe\njane@example.com"

    assert extract_candidate_name(text, "candidate.pdf") == "Jane Doe"


def test_name_extraction_rejects_domain_and_partial_surname():
    assert (
        extract_candidate_name(
            "abhinavmishra.in\ncontact@example.com",
            "candidate_10.pdf",
        )
        == "Candidate 10"
    )
    assert (
        extract_candidate_name(
            "Prathamesh\nPatil\ncontact@example.com",
            "candidate.pdf",
        )
        == "Prathamesh Patil"
    )


def test_name_extraction_rejects_labels_and_preserves_uppercase_names():
    assert (
        extract_candidate_name("Linked In\nemail@example.com", "candidate.pdf")
        == "Candidate"
    )
    assert extract_candidate_name("VISHWAS BADIGER") == "VISHWAS BADIGER"
    assert extract_candidate_name("abhinavMishra") == "abhinav Mishra"


def test_name_extraction_salvages_name_from_contact_header_line():
    abhinav_header = (
        "ABHINAV MISHRA Ghaziabad, UP | +91 7701956829 | "
        "abhinavmishra3322@gmail.com | abhinavmishra.in | "
        "github.com/AbhinavMishra32"
    )
    kiran_header = (
        "Kiran Naregal Bangalore | 9964923041 | "
        "kiran@example.com | linkedin.com/in/kiran-naregal"
    )

    assert extract_candidate_name(abhinav_header) == "ABHINAV MISHRA"
    assert extract_candidate_name(kiran_header) == "Kiran Naregal"


def test_project_links_do_not_override_header_name():
    text = (
        "PAVANI A\n"
        "Bengaluru, Karnataka, India pavani@example.com\n"
        "SUMMARY\n"
        "Software engineer with backend experience.\n"
        "PROJECTS\n"
        "Handwritten Digit Recognition\n"
        "https://github.com/example/project"
    )

    assert extract_candidate_name(text, "candidate.pdf") == "PAVANI A"


def test_candidate_can_be_built_from_ingested_text():
    candidate = extract_candidate_from_text(
        "candidate.pdf",
        "Jane Doe\n"
        "jane@example.com\n"
        "SKILLS\nPython, FastAPI\n"
        "PROJECTS\nBuilt a service using Python and FastAPI.",
    )

    assert candidate.candidate_name == "Jane Doe"
    assert candidate.email == "jane@example.com"
    assert set(candidate.skills) == {"Python", "FastAPI"}
    assert "Built a service" in candidate.projects


def test_skills_are_known_canonical_technologies_only():
    text = (
        "SKILLS\n"
        "Python, Fast API, LangChain, LangGraph, RAG, embedding, Embeddings, "
        "Docker, AWS, Postgres, React.js\n"
        "Full-Stack: React js\n"
        "Engineering: pytest\n"
        "Databases: PostgreSQL, MongoDB\n"
        "Tools: GitHub Actions\n"
        "EDUCATION\n"
        "Bachelor of Engineering, May 2025"
    )

    skills = extract_skills(text, split_sections(text))

    assert {
        "GitHub Actions",
        "PostgreSQL",
        "MongoDB",
        "pytest",
        "LangChain",
        "LangGraph",
        "FastAPI",
        "Docker",
        "Python",
        "Embeddings",
        "AWS",
        "React",
    }.issubset(skills)
    assert len({skill.casefold() for skill in skills}) == len(skills)
    assert not any(":" in skill or skill.lower() in {"step", "functions"} for skill in skills)