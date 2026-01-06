import unittest
from bnlp.core import (
    BNLPException,
    ModelNotFoundError,
    ModelLoadError,
    TokenizationError,
    EmbeddingError,
    TaggingError,
    DownloadError,
    PipelineError,
    InvalidInputError,
)


class TestBNLPException(unittest.TestCase):
    def test_base_exception(self):
        """Test base BNLPException."""
        exc = BNLPException("Test error")
        self.assertEqual(exc.message, "Test error")
        self.assertEqual(str(exc), "Test error")

    def test_base_exception_with_details(self):
        """Test BNLPException with details."""
        exc = BNLPException("Test error", details="Additional info")
        self.assertEqual(exc.details, "Additional info")
        self.assertIn("Additional info", str(exc))

    def test_exception_is_exception(self):
        """Test that BNLPException is an Exception."""
        exc = BNLPException("Test")
        self.assertIsInstance(exc, Exception)


class TestModelNotFoundError(unittest.TestCase):
    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        exc = ModelNotFoundError("test_model")
        self.assertEqual(exc.model_name, "test_model")
        self.assertIn("test_model", str(exc))
        self.assertIn("not found", str(exc))

    def test_model_not_found_with_path(self):
        """Test ModelNotFoundError with path."""
        exc = ModelNotFoundError("test_model", model_path="/path/to/model")
        self.assertEqual(exc.model_path, "/path/to/model")
        self.assertIn("/path/to/model", str(exc))

    def test_model_not_found_suggests_download(self):
        """Test that ModelNotFoundError suggests downloading."""
        exc = ModelNotFoundError("test_model")
        self.assertIn("download", str(exc).lower())


class TestModelLoadError(unittest.TestCase):
    def test_model_load_error(self):
        """Test ModelLoadError."""
        exc = ModelLoadError("test_model")
        self.assertEqual(exc.model_name, "test_model")
        self.assertIn("test_model", str(exc))
        self.assertIn("load", str(exc).lower())

    def test_model_load_error_with_reason(self):
        """Test ModelLoadError with reason."""
        exc = ModelLoadError("test_model", reason="File corrupted")
        self.assertIn("File corrupted", str(exc))


class TestTokenizationError(unittest.TestCase):
    def test_tokenization_error(self):
        """Test TokenizationError."""
        exc = TokenizationError()
        self.assertIn("Tokenization", str(exc))

    def test_tokenization_error_with_text(self):
        """Test TokenizationError with text."""
        exc = TokenizationError(text="Test text")
        self.assertIn("Test text", str(exc))

    def test_tokenization_error_truncates_long_text(self):
        """Test that long text is truncated."""
        long_text = "a" * 100
        exc = TokenizationError(text=long_text)
        # Should truncate and add ...
        self.assertIn("...", str(exc))


class TestEmbeddingError(unittest.TestCase):
    def test_embedding_error(self):
        """Test EmbeddingError."""
        exc = EmbeddingError()
        self.assertIn("Embedding", str(exc))

    def test_embedding_error_with_word(self):
        """Test EmbeddingError with word."""
        exc = EmbeddingError(word="বাংলা")
        self.assertEqual(exc.word, "বাংলা")
        self.assertIn("বাংলা", str(exc))


class TestTaggingError(unittest.TestCase):
    def test_tagging_error(self):
        """Test TaggingError."""
        exc = TaggingError()
        self.assertIn("tagging", str(exc).lower())

    def test_tagging_error_with_type(self):
        """Test TaggingError with tag type."""
        exc = TaggingError(tag_type="NER")
        self.assertIn("NER", str(exc))

    def test_tagging_error_with_text(self):
        """Test TaggingError with text."""
        exc = TaggingError(text="Test text", tag_type="POS")
        self.assertIn("Test text", str(exc))


class TestDownloadError(unittest.TestCase):
    def test_download_error(self):
        """Test DownloadError."""
        exc = DownloadError("test_model")
        self.assertEqual(exc.model_name, "test_model")
        self.assertIn("test_model", str(exc))
        self.assertIn("download", str(exc).lower())

    def test_download_error_with_url(self):
        """Test DownloadError with URL."""
        exc = DownloadError("test_model", url="https://example.com/model")
        self.assertEqual(exc.url, "https://example.com/model")
        self.assertIn("https://example.com/model", str(exc))

    def test_download_error_suggests_retry(self):
        """Test that DownloadError suggests checking connection."""
        exc = DownloadError("test_model")
        self.assertIn("internet", str(exc).lower())


class TestPipelineError(unittest.TestCase):
    def test_pipeline_error(self):
        """Test PipelineError."""
        exc = PipelineError()
        self.assertIn("Pipeline", str(exc))

    def test_pipeline_error_with_step(self):
        """Test PipelineError with step name."""
        exc = PipelineError(step_name="tokenize")
        self.assertEqual(exc.step_name, "tokenize")
        self.assertIn("tokenize", str(exc))

    def test_pipeline_error_with_reason(self):
        """Test PipelineError with reason."""
        exc = PipelineError(step_name="clean", reason="Invalid input")
        self.assertIn("Invalid input", str(exc))


class TestInvalidInputError(unittest.TestCase):
    def test_invalid_input_error(self):
        """Test InvalidInputError."""
        exc = InvalidInputError("text", expected="string")
        self.assertEqual(exc.param_name, "text")
        self.assertIn("text", str(exc))
        self.assertIn("string", str(exc))

    def test_invalid_input_error_with_received(self):
        """Test InvalidInputError with received value."""
        exc = InvalidInputError("count", expected="int", received="string")
        self.assertIn("int", str(exc))
        self.assertIn("string", str(exc))


class TestExceptionHierarchy(unittest.TestCase):
    def test_all_exceptions_inherit_from_base(self):
        """Test that all exceptions inherit from BNLPException."""
        exceptions = [
            ModelNotFoundError("test"),
            ModelLoadError("test"),
            TokenizationError(),
            EmbeddingError(),
            TaggingError(),
            DownloadError("test"),
            PipelineError(),
            InvalidInputError("test", "test"),
        ]

        for exc in exceptions:
            self.assertIsInstance(exc, BNLPException)
            self.assertIsInstance(exc, Exception)


if __name__ == "__main__":
    unittest.main()
