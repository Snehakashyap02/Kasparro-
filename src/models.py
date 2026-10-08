from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    source_file: str
    candidate_name: str
    email: Optional[str] = None
    github_url: Optional[str] = None
    github_username: Optional[str] = None
    full_text: str
    sections: Dict[str, str] = Field(default_factory=dict)
    skills: List[str] = Field(default_factory=list)
    projects: str = ""
    experience: str = ""
    education: str = ""
    coursework: str = ""


class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0


class ResumeAnalysis(BaseModel):
    ai_project_depth_score: int = Field(ge=0, le=40)
    python_backend_score: int = Field(ge=0, le=30)
    cloud_fullstack_score: int = Field(ge=0, le=15)
    engineering_depth_score: int = Field(ge=0, le=5)
    project_summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    is_thin_wrapper: bool = False
    evidence: Dict[str, str] = Field(default_factory=dict)


class GitHubResult(BaseModel):
    points: int = Field(ge=0, le=10)
    recent_activity_points: int = Field(ge=0, le=5)
    maintained_relevant_points: int = Field(ge=0, le=5)
    summary: str
    username: Optional[str] = None


class ScreeningResult(BaseModel):
    rank: Optional[int] = None
    candidate_name: str
    email: Optional[str] = None
    github_url: Optional[str] = None
    github_username: Optional[str] = None
    eligible: bool
    total_score: int = 0
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    matched_skills: List[str] = Field(default_factory=list)
    evidence: Dict[str, str] = Field(default_factory=dict)
    evidence_snippets: List[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    source_file: Optional[str] = None
