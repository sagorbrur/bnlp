"""
Bengali Language Detection Module

This module provides language detection capabilities for Bengali text
using FastText's language identification model.
"""

from bnlp.langdetect.detector import (
    LanguageDetector,
    DetectionResult,
    detect_language,
    is_bengali,
)

__all__ = [
    "LanguageDetector",
    "DetectionResult",
    "detect_language",
    "is_bengali",
]
