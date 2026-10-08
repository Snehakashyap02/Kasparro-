# AI Resume Screening & Ranking System

A Python CLI for screening resumes against hard Python + AI/agentic requirements, ranking eligible candidates, enriching scores with public GitHub activity, and producing `results.json`.

## Requirements

- Python 3.11 recommended
- OpenAI API key for LLM-assisted scoring (optional; deterministic scoring is used if unavailable)
- Internet access for GitHub enrichment

## Setup

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
GITHUB_TOKEN=
```

`OPENAI_BASE_URL` and `OPENAI_MODEL` can target another OpenAI-compatible
provider, including NVIDIA NIM. The key is optional; without it the CLI uses
deterministic scoring.

Put PDF resumes in:

```text
resumes/
```

## Run

```bash
python main.py --input ./resumes --output ./output/results.json
```

## Tests

```bash
pytest
```

## Project Structure

```text
main.py
src/
  config.py
  models.py
  ingest.py
  extract.py
  eligibility.py
  scoring.py
  github.py
  llm.py
  pipeline.py
tests/
  test_eligibility.py
  test_scoring.py
  test_llm.py
  test_pipeline.py
resumes/
output/
```

## Design Decisions

### Filtering strategy

Hard eligibility is rule-based and happens before ranking. A candidate must have genuine Python evidence in skills, projects, or experience and meaningful AI/LLM/RAG/agentic evidence. Education/coursework is not used as the sole AI qualification.

This prevents a polished but irrelevant resume from receiving a high ranking.

### Scoring strategy

Eligible candidates receive up to 100 points:

- AI / Agentic / RAG Project Depth: 40
- Python & Backend Engineering: 30
- Cloud / Deployment / Full Stack: 15
- GitHub Activity: 10
- Engineering Depth: 5

Project and experience evidence is preferred over skill-list keywords.

### LLM usage

The OpenAI-compatible client is isolated in `src/llm.py`. The model returns a
Pydantic JSON-Schema-constrained result with concise evidence. If the API is
unavailable or the response fails validation, the deterministic scorer is used
instead.

The LLM does not make the hard eligibility decision.

### GitHub scoring

GitHub is an additional signal only. Recent public activity receives up to 5 points and maintained/relevant repositories receive up to 5 points. Missing profiles, API errors, timeouts, and rate limits do not fail the screening batch.

### Output

The JSON includes all parsed candidates, ranked eligible candidates first,
rejected candidates with reasons, score breakdowns, matched skills, evidence,
contact/GitHub details when found, and a batch summary. If interrupted with
Ctrl+C, completed candidates are written as partial results and marked in the
batch summary.

## If I Had More Time

1. Improve resume section detection for unusual PDF layouts.
2. Add persistent caching for LLM and GitHub results.
3. Add more nuanced project-quality detection and duplicate candidate detection.
4. Add an optional FastAPI endpoint and a small terminal report.

## Notes

The system intentionally favors a reliable, explainable backend over a frontend or database because the assignment scope does not require either.
