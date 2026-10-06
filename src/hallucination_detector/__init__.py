"""Core domain and baseline pipeline for the research project."""

from .domain import Claim, Evidence, VerificationLabel, VerificationResult

__all__ = ["Claim", "Evidence", "VerificationLabel", "VerificationResult"]