"""
Language Detection for Bengali Text

This module provides language detection using FastText's pre-trained
language identification model. FastText's lid.176.bin model can identify
176 languages with high accuracy.

FastText is chosen for its:
- High accuracy (>95% on most languages)
- Fast inference speed
- Support for 176 languages including Bengali
- Small model size options available

References:
- FastText: https://fasttext.cc/docs/en/language-identification.html
- Paper: "Bag of Tricks for Efficient Text Classification" (Joulin et al., 2016)
"""

import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Lazy import for fasttext
_fasttext = None
_model = None

# Model URLs
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_COMPRESSED_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def _ensure_fasttext():
    """Lazily import fasttext when needed."""
    global _fasttext
    if _fasttext is None:
        try:
            import fasttext
            _fasttext = fasttext
            # Suppress fasttext warnings about loading model
            _fasttext.FastText.eprint = lambda x: None
        except ImportError:
            raise ImportError(
                "fasttext is required for language detection. "
                "Install it with: pip install fasttext"
            )


def _get_default_model_path() -> Path:
    """Get the default path for storing the FastText model."""
    # Use user's home directory for model storage
    home = Path.home()
    model_dir = home / ".bnlp" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "lid.176.ftz"


def _download_model(model_path: Path, use_compressed: bool = True) -> None:
    """Download the FastText language identification model.

    Args:
        model_path: Path where the model should be saved
        use_compressed: If True, download the compressed .ftz model (917KB)
                       If False, download the full .bin model (126MB)
    """
    url = FASTTEXT_MODEL_COMPRESSED_URL if use_compressed else FASTTEXT_MODEL_URL
    print(f"Downloading FastText language model from {url}...")
    print(f"This is a one-time download. Model will be saved to: {model_path}")

    try:
        urllib.request.urlretrieve(url, str(model_path))
        print("Download complete!")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download FastText model: {e}\n"
            f"Please download manually from {url} and place at {model_path}"
        )


@dataclass
class DetectionResult:
    """Result of language detection.

    Attributes:
        language: ISO 639-1 language code (e.g., 'bn', 'en', 'hi')
        confidence: Confidence score (0.0 to 1.0)
        is_bengali: True if detected language is Bengali
        all_predictions: List of (language, confidence) tuples for top predictions
    """
    language: str
    confidence: float
    is_bengali: bool
    all_predictions: List[Tuple[str, float]]

    def __repr__(self) -> str:
        return f"DetectionResult(language='{self.language}', confidence={self.confidence:.4f}, is_bengali={self.is_bengali})"


class LanguageDetector:
    """Language detector using FastText's language identification model.

    This class provides fast and accurate language detection for text,
    with special support for Bengali language detection.

    Example:
        >>> from bnlp.langdetect import LanguageDetector
        >>>
        >>> detector = LanguageDetector()
        >>>
        >>> # Detect language
        >>> result = detector.detect("আমি বাংলায় গান গাই")
        >>> print(result.language)  # 'bn'
        >>> print(result.confidence)  # ~0.99
        >>> print(result.is_bengali)  # True
        >>>
        >>> # Check if text is Bengali
        >>> detector.is_bengali("Hello world")  # False
        >>> detector.is_bengali("আমি বাংলায় গান গাই")  # True
        >>>
        >>> # Get multiple predictions
        >>> result = detector.detect("আমি বাংলায় গান গাই", top_k=3)
        >>> print(result.all_predictions)

    Attributes:
        model_path: Path to the FastText model file
        threshold: Minimum confidence threshold for detection
    """

    # Common language code mappings
    LANGUAGE_NAMES: Dict[str, str] = {
        'bn': 'Bengali',
        'en': 'English',
        'hi': 'Hindi',
        'ur': 'Urdu',
        'ar': 'Arabic',
        'zh': 'Chinese',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'ja': 'Japanese',
        'ko': 'Korean',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ta': 'Tamil',
        'te': 'Telugu',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'or': 'Odia',
        'as': 'Assamese',
        'ne': 'Nepali',
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        auto_download: bool = True,
        threshold: float = 0.5,
    ):
        """Initialize the language detector.

        Args:
            model_path: Path to FastText model file. If None, uses default location.
            auto_download: If True, automatically download model if not found.
            threshold: Minimum confidence threshold for reliable detection.
        """
        _ensure_fasttext()

        self.threshold = threshold

        # Determine model path
        if model_path:
            self._model_path = Path(model_path)
        else:
            self._model_path = _get_default_model_path()

        # Check if model exists, download if needed
        if not self._model_path.exists():
            if auto_download:
                _download_model(self._model_path, use_compressed=True)
            else:
                raise FileNotFoundError(
                    f"FastText model not found at {self._model_path}. "
                    f"Set auto_download=True or download manually from {FASTTEXT_MODEL_COMPRESSED_URL}"
                )

        # Load the model
        self._model = _fasttext.load_model(str(self._model_path))

    def detect(self, text: str, top_k: int = 1) -> DetectionResult:
        """Detect the language of the given text.

        Args:
            text: Text to detect language for
            top_k: Number of top predictions to return

        Returns:
            DetectionResult with detected language and confidence
        """
        # Clean text - FastText expects single line
        text = self._preprocess(text)

        if not text.strip():
            return DetectionResult(
                language='unknown',
                confidence=0.0,
                is_bengali=False,
                all_predictions=[],
            )

        # Get predictions
        predictions = self._model.predict(text, k=top_k)
        labels, scores = predictions

        # Parse results (FastText returns labels like '__label__en')
        all_predictions = []
        for label, score in zip(labels, scores):
            lang_code = label.replace('__label__', '')
            all_predictions.append((lang_code, float(score)))

        # Get top prediction
        top_lang = all_predictions[0][0] if all_predictions else 'unknown'
        top_confidence = all_predictions[0][1] if all_predictions else 0.0

        return DetectionResult(
            language=top_lang,
            confidence=top_confidence,
            is_bengali=(top_lang == 'bn'),
            all_predictions=all_predictions,
        )

    def is_bengali(self, text: str, threshold: Optional[float] = None) -> bool:
        """Check if text is primarily Bengali.

        Args:
            text: Text to check
            threshold: Confidence threshold (uses instance threshold if None)

        Returns:
            True if text is detected as Bengali with sufficient confidence
        """
        if threshold is None:
            threshold = self.threshold

        result = self.detect(text)
        return result.is_bengali and result.confidence >= threshold

    def detect_batch(self, texts: List[str], top_k: int = 1) -> List[DetectionResult]:
        """Detect languages for multiple texts.

        Args:
            texts: List of texts to detect
            top_k: Number of top predictions per text

        Returns:
            List of DetectionResult objects
        """
        return [self.detect(text, top_k=top_k) for text in texts]

    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from ISO 639-1 code.

        Args:
            lang_code: Two-letter language code

        Returns:
            Full language name or the code if unknown
        """
        return self.LANGUAGE_NAMES.get(lang_code, lang_code)

    def detect_mixed(self, text: str, threshold: float = 0.3) -> Dict[str, float]:
        """Detect if text contains multiple languages.

        This method is useful for detecting code-mixed text (e.g., Bengali-English).

        Args:
            text: Text to analyze
            threshold: Minimum proportion threshold for a language

        Returns:
            Dictionary of detected languages and their proportions
        """
        # Split text into segments and detect each
        # Simple sentence-based splitting
        segments = re.split(r'[।.!?\n]+', text)
        segments = [s.strip() for s in segments if s.strip()]

        if not segments:
            return {}

        # Count language occurrences
        lang_counts: Dict[str, int] = {}
        total = 0

        for segment in segments:
            result = self.detect(segment)
            if result.confidence >= 0.5:  # Only count confident predictions
                lang_counts[result.language] = lang_counts.get(result.language, 0) + 1
                total += 1

        if total == 0:
            return {}

        # Calculate proportions
        proportions = {
            lang: count / total
            for lang, count in lang_counts.items()
            if count / total >= threshold
        }

        return proportions

    def _preprocess(self, text: str) -> str:
        """Preprocess text for FastText.

        Args:
            text: Raw text

        Returns:
            Preprocessed text (single line, normalized whitespace)
        """
        # Replace newlines with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def __call__(self, text: str) -> str:
        """Callable interface - returns language code.

        Args:
            text: Text to detect

        Returns:
            Detected language code
        """
        return self.detect(text).language


# Module-level convenience functions with lazy initialization
_default_detector: Optional[LanguageDetector] = None


def _get_default_detector() -> LanguageDetector:
    """Get or create the default language detector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = LanguageDetector()
    return _default_detector


def detect_language(text: str, top_k: int = 1) -> DetectionResult:
    """Detect the language of text using the default detector.

    This is a convenience function that uses a shared LanguageDetector instance.

    Args:
        text: Text to detect language for
        top_k: Number of top predictions to return

    Returns:
        DetectionResult with detected language and confidence

    Example:
        >>> from bnlp.langdetect import detect_language
        >>> result = detect_language("আমি বাংলায় গান গাই")
        >>> print(result.language)  # 'bn'
    """
    return _get_default_detector().detect(text, top_k=top_k)


def is_bengali(text: str, threshold: float = 0.5) -> bool:
    """Check if text is primarily Bengali.

    This is a convenience function that uses a shared LanguageDetector instance.

    Args:
        text: Text to check
        threshold: Confidence threshold for detection

    Returns:
        True if text is detected as Bengali with sufficient confidence

    Example:
        >>> from bnlp.langdetect import is_bengali
        >>> is_bengali("আমি বাংলায় গান গাই")  # True
        >>> is_bengali("Hello world")  # False
    """
    return _get_default_detector().is_bengali(text, threshold=threshold)
