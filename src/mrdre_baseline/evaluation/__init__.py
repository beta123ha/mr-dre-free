"""
Evaluation module for deep research reports.
"""

from .checklist import evaluate_checklist, aggregate_checklist_metrics
from .citations import evaluate_citations, aggregate_citation_metrics, extract_claims, judge_claims
from .presentation import evaluate_rubric, aggregate_rubric_metrics

__all__ = [
    "evaluate_checklist",
    "aggregate_checklist_metrics",
    "evaluate_citations",
    "aggregate_citation_metrics",
    "extract_claims",
    "judge_claims",
    "evaluate_rubric",
    "aggregate_rubric_metrics",
]

