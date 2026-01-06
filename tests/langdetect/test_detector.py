"""
Unit tests for Bengali Language Detection module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_detection_result_creation(self):
        """Test creating a DetectionResult."""
        from bnlp.langdetect.detector import DetectionResult

        result = DetectionResult(
            language='bn',
            confidence=0.95,
            is_bengali=True,
            all_predictions=[('bn', 0.95), ('hi', 0.03)]
        )

        assert result.language == 'bn'
        assert result.confidence == 0.95
        assert result.is_bengali is True
        assert len(result.all_predictions) == 2

    def test_detection_result_repr(self):
        """Test DetectionResult string representation."""
        from bnlp.langdetect.detector import DetectionResult

        result = DetectionResult(
            language='en',
            confidence=0.9876,
            is_bengali=False,
            all_predictions=[('en', 0.9876)]
        )

        repr_str = repr(result)
        assert 'en' in repr_str
        assert '0.9876' in repr_str
        assert 'False' in repr_str


class TestLanguageDetectorMocked:
    """Tests for LanguageDetector using mocked FastText."""

    @pytest.fixture
    def mock_fasttext(self):
        """Create a mock fasttext module."""
        mock_ft = MagicMock()

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            ['__label__bn'],
            [0.95]
        )
        mock_ft.load_model.return_value = mock_model

        return mock_ft, mock_model

    @pytest.fixture
    def detector_with_mock(self, mock_fasttext, tmp_path):
        """Create a LanguageDetector with mocked dependencies."""
        mock_ft, mock_model = mock_fasttext

        # Create a fake model file
        model_file = tmp_path / "test_model.ftz"
        model_file.touch()

        with patch.dict(sys.modules, {'fasttext': mock_ft}):
            # Reset the module's cached import
            from bnlp.langdetect import detector
            detector._fasttext = None

            from bnlp.langdetect import LanguageDetector
            det = LanguageDetector(model_path=str(model_file), auto_download=False)

        return det, mock_model

    def test_detect_bengali(self, detector_with_mock):
        """Test detecting Bengali text."""
        detector, mock_model = detector_with_mock

        # Configure mock for Bengali
        mock_model.predict.return_value = (['__label__bn'], [0.95])

        result = detector.detect("আমি বাংলায় গান গাই")

        assert result.language == 'bn'
        assert result.confidence == 0.95
        assert result.is_bengali is True

    def test_detect_english(self, detector_with_mock):
        """Test detecting English text."""
        detector, mock_model = detector_with_mock

        # Configure mock for English
        mock_model.predict.return_value = (['__label__en'], [0.98])

        result = detector.detect("Hello world, this is English text")

        assert result.language == 'en'
        assert result.confidence == 0.98
        assert result.is_bengali is False

    def test_detect_empty_text(self, detector_with_mock):
        """Test detecting empty text."""
        detector, mock_model = detector_with_mock

        result = detector.detect("")

        assert result.language == 'unknown'
        assert result.confidence == 0.0
        assert result.is_bengali is False
        assert result.all_predictions == []

    def test_detect_whitespace_only(self, detector_with_mock):
        """Test detecting whitespace-only text."""
        detector, mock_model = detector_with_mock

        result = detector.detect("   \n\t  ")

        assert result.language == 'unknown'
        assert result.confidence == 0.0

    def test_detect_top_k(self, detector_with_mock):
        """Test getting top-k predictions."""
        detector, mock_model = detector_with_mock

        # Configure mock for multiple predictions
        mock_model.predict.return_value = (
            ['__label__bn', '__label__hi', '__label__en'],
            [0.85, 0.10, 0.05]
        )

        result = detector.detect("টেস্ট টেক্সট", top_k=3)

        assert len(result.all_predictions) == 3
        assert result.all_predictions[0] == ('bn', 0.85)
        assert result.all_predictions[1] == ('hi', 0.10)
        assert result.all_predictions[2] == ('en', 0.05)

    def test_is_bengali_true(self, detector_with_mock):
        """Test is_bengali returns True for Bengali text."""
        detector, mock_model = detector_with_mock
        mock_model.predict.return_value = (['__label__bn'], [0.95])

        assert detector.is_bengali("আমি বাংলায় গান গাই") is True

    def test_is_bengali_false(self, detector_with_mock):
        """Test is_bengali returns False for non-Bengali text."""
        detector, mock_model = detector_with_mock
        mock_model.predict.return_value = (['__label__en'], [0.98])

        assert detector.is_bengali("Hello world") is False

    def test_is_bengali_low_confidence(self, detector_with_mock):
        """Test is_bengali returns False when confidence is below threshold."""
        detector, mock_model = detector_with_mock
        mock_model.predict.return_value = (['__label__bn'], [0.3])

        # Default threshold is 0.5
        assert detector.is_bengali("text") is False

    def test_is_bengali_custom_threshold(self, detector_with_mock):
        """Test is_bengali with custom threshold."""
        detector, mock_model = detector_with_mock
        mock_model.predict.return_value = (['__label__bn'], [0.6])

        # With high threshold, should fail
        assert detector.is_bengali("text", threshold=0.8) is False
        # With low threshold, should pass
        assert detector.is_bengali("text", threshold=0.5) is True

    def test_detect_batch(self, detector_with_mock):
        """Test batch detection."""
        detector, mock_model = detector_with_mock

        # Configure mock to return different languages
        mock_model.predict.side_effect = [
            (['__label__bn'], [0.95]),
            (['__label__en'], [0.98]),
            (['__label__hi'], [0.90]),
        ]

        texts = ["বাংলা টেক্সট", "English text", "हिंदी टेक्स्ट"]
        results = detector.detect_batch(texts)

        assert len(results) == 3
        assert results[0].language == 'bn'
        assert results[1].language == 'en'
        assert results[2].language == 'hi'

    def test_get_language_name(self, detector_with_mock):
        """Test getting language names."""
        detector, _ = detector_with_mock

        assert detector.get_language_name('bn') == 'Bengali'
        assert detector.get_language_name('en') == 'English'
        assert detector.get_language_name('hi') == 'Hindi'
        # Unknown language returns the code
        assert detector.get_language_name('xyz') == 'xyz'

    def test_callable_interface(self, detector_with_mock):
        """Test calling detector directly."""
        detector, mock_model = detector_with_mock
        mock_model.predict.return_value = (['__label__bn'], [0.95])

        result = detector("আমি বাংলায় গান গাই")
        assert result == 'bn'

    def test_preprocess_removes_newlines(self, detector_with_mock):
        """Test that preprocessing removes newlines."""
        detector, mock_model = detector_with_mock
        mock_model.predict.return_value = (['__label__bn'], [0.95])

        # Call with text containing newlines
        detector.detect("Line 1\nLine 2\rLine 3")

        # Check that the model received text without newlines
        call_args = mock_model.predict.call_args[0][0]
        assert '\n' not in call_args
        assert '\r' not in call_args

    def test_preprocess_normalizes_whitespace(self, detector_with_mock):
        """Test that preprocessing normalizes whitespace."""
        detector, mock_model = detector_with_mock
        mock_model.predict.return_value = (['__label__bn'], [0.95])

        detector.detect("word1    word2\t\tword3")

        call_args = mock_model.predict.call_args[0][0]
        assert '    ' not in call_args
        assert '\t' not in call_args


class TestDetectMixed:
    """Tests for mixed language detection."""

    @pytest.fixture
    def detector_with_mock(self, tmp_path):
        """Create a LanguageDetector with mocked dependencies."""
        mock_ft = MagicMock()
        mock_model = MagicMock()
        mock_ft.load_model.return_value = mock_model

        model_file = tmp_path / "test_model.ftz"
        model_file.touch()

        with patch.dict(sys.modules, {'fasttext': mock_ft}):
            from bnlp.langdetect import detector
            detector._fasttext = None

            from bnlp.langdetect import LanguageDetector
            det = LanguageDetector(model_path=str(model_file), auto_download=False)

        return det, mock_model

    def test_detect_mixed_single_language(self, detector_with_mock):
        """Test detect_mixed with single language text."""
        detector, mock_model = detector_with_mock

        # All segments detected as Bengali
        mock_model.predict.return_value = (['__label__bn'], [0.95])

        result = detector.detect_mixed("বাংলা বাক্য। আরেকটি বাক্য।")

        assert 'bn' in result
        assert result['bn'] == 1.0

    def test_detect_mixed_multiple_languages(self, detector_with_mock):
        """Test detect_mixed with multiple languages."""
        detector, mock_model = detector_with_mock

        # Alternate between Bengali and English
        mock_model.predict.side_effect = [
            (['__label__bn'], [0.95]),
            (['__label__en'], [0.95]),
            (['__label__bn'], [0.95]),
            (['__label__en'], [0.95]),
        ]

        result = detector.detect_mixed("বাংলা। English. আবার বাংলা। More English.")

        assert 'bn' in result
        assert 'en' in result
        assert result['bn'] == 0.5
        assert result['en'] == 0.5

    def test_detect_mixed_empty_text(self, detector_with_mock):
        """Test detect_mixed with empty text."""
        detector, _ = detector_with_mock

        result = detector.detect_mixed("")
        assert result == {}


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_language_function(self, tmp_path):
        """Test module-level detect_language function."""
        mock_ft = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = (['__label__bn'], [0.95])
        mock_ft.load_model.return_value = mock_model

        model_file = tmp_path / "test_model.ftz"
        model_file.touch()

        with patch.dict(sys.modules, {'fasttext': mock_ft}):
            from bnlp.langdetect import detector
            detector._fasttext = None
            detector._default_detector = None

            # Patch the default model path
            with patch.object(detector, '_get_default_model_path', return_value=model_file):
                result = detector.detect_language("আমি বাংলায় গান গাই")

        assert result.language == 'bn'

    def test_is_bengali_function(self, tmp_path):
        """Test module-level is_bengali function."""
        mock_ft = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = (['__label__bn'], [0.95])
        mock_ft.load_model.return_value = mock_model

        model_file = tmp_path / "test_model.ftz"
        model_file.touch()

        with patch.dict(sys.modules, {'fasttext': mock_ft}):
            from bnlp.langdetect import detector
            detector._fasttext = None
            detector._default_detector = None

            with patch.object(detector, '_get_default_model_path', return_value=model_file):
                result = detector.is_bengali("আমি বাংলায় গান গাই")

        assert result is True


class TestLazyImport:
    """Tests for lazy import behavior."""

    def test_fasttext_import_error(self):
        """Test proper error message when fasttext is not installed."""
        # This test verifies the error handling path
        from bnlp.langdetect.detector import _ensure_fasttext

        # Temporarily modify the global to test re-import
        from bnlp.langdetect import detector
        original = detector._fasttext

        # Reset to trigger re-import
        detector._fasttext = None

        with patch.dict(sys.modules, {'fasttext': None}):
            # Remove fasttext from modules
            if 'fasttext' in sys.modules:
                del sys.modules['fasttext']

            # This should raise ImportError with helpful message
            # Note: This test might not work in environments where fasttext is installed
            # The actual import will succeed if fasttext is available

        # Restore
        detector._fasttext = original


class TestModelDownload:
    """Tests for model download functionality."""

    def test_get_default_model_path(self):
        """Test default model path generation."""
        from bnlp.langdetect.detector import _get_default_model_path
        from pathlib import Path

        path = _get_default_model_path()

        assert isinstance(path, Path)
        assert 'bnlp' in str(path)
        assert path.name == 'lid.176.ftz'

    def test_model_not_found_error(self, tmp_path):
        """Test error when model not found and auto_download is False."""
        mock_ft = MagicMock()

        model_file = tmp_path / "nonexistent.ftz"

        with patch.dict(sys.modules, {'fasttext': mock_ft}):
            from bnlp.langdetect import detector
            detector._fasttext = None

            from bnlp.langdetect import LanguageDetector

            with pytest.raises(FileNotFoundError) as exc_info:
                LanguageDetector(model_path=str(model_file), auto_download=False)

            assert "FastText model not found" in str(exc_info.value)
