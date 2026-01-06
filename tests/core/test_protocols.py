import unittest
from typing import List, Tuple
import numpy as np
from bnlp import BasicTokenizer, CleanText
from bnlp.core import (
    TokenizerProtocol,
    EmbeddingProtocol,
    TaggerProtocol,
    TextProcessorProtocol,
)


class TestTokenizerProtocol(unittest.TestCase):
    def test_basic_tokenizer_implements_protocol(self):
        """Test that BasicTokenizer implements TokenizerProtocol."""
        tokenizer = BasicTokenizer()
        self.assertTrue(isinstance(tokenizer, TokenizerProtocol))

    def test_custom_tokenizer_implements_protocol(self):
        """Test a custom class implementing TokenizerProtocol."""

        class CustomTokenizer:
            def tokenize(self, text: str) -> List[str]:
                return text.split()

            def __call__(self, text: str) -> List[str]:
                return self.tokenize(text)

        tokenizer = CustomTokenizer()
        self.assertTrue(isinstance(tokenizer, TokenizerProtocol))

    def test_incomplete_tokenizer_fails_protocol(self):
        """Test that incomplete implementation fails protocol check."""

        class IncompleteTokenizer:
            def tokenize(self, text: str) -> List[str]:
                return text.split()
            # Missing __call__

        tokenizer = IncompleteTokenizer()
        self.assertFalse(isinstance(tokenizer, TokenizerProtocol))


class TestTextProcessorProtocol(unittest.TestCase):
    def test_cleantext_implements_protocol(self):
        """Test that CleanText implements TextProcessorProtocol."""
        cleaner = CleanText()
        self.assertTrue(isinstance(cleaner, TextProcessorProtocol))

    def test_custom_processor_implements_protocol(self):
        """Test a custom class implementing TextProcessorProtocol."""

        class CustomProcessor:
            def __call__(self, text: str) -> str:
                return text.upper()

        processor = CustomProcessor()
        self.assertTrue(isinstance(processor, TextProcessorProtocol))


class TestEmbeddingProtocol(unittest.TestCase):
    def test_custom_embedding_implements_protocol(self):
        """Test a custom class implementing EmbeddingProtocol."""

        class CustomEmbedding:
            def get_word_vector(self, word: str) -> np.ndarray:
                return np.zeros(100)

        embedding = CustomEmbedding()
        self.assertTrue(isinstance(embedding, EmbeddingProtocol))

    def test_incomplete_embedding_fails_protocol(self):
        """Test that incomplete implementation fails protocol check."""

        class IncompleteEmbedding:
            def get_vector(self, word: str) -> np.ndarray:  # Wrong method name
                return np.zeros(100)

        embedding = IncompleteEmbedding()
        self.assertFalse(isinstance(embedding, EmbeddingProtocol))


class TestTaggerProtocol(unittest.TestCase):
    def test_custom_tagger_implements_protocol(self):
        """Test a custom class implementing TaggerProtocol."""

        class CustomTagger:
            def tag(self, text: str) -> List[Tuple[str, str]]:
                return [("word", "TAG")]

        tagger = CustomTagger()
        self.assertTrue(isinstance(tagger, TaggerProtocol))

    def test_incomplete_tagger_fails_protocol(self):
        """Test that incomplete implementation fails protocol check."""

        class IncompleteTagger:
            def predict(self, text: str) -> List[Tuple[str, str]]:  # Wrong method name
                return [("word", "TAG")]

        tagger = IncompleteTagger()
        self.assertFalse(isinstance(tagger, TaggerProtocol))


if __name__ == "__main__":
    unittest.main()
