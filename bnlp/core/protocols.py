"""
BNLP Protocol Definitions

This module defines Protocol classes for type safety and polymorphism.
These protocols define the expected interfaces for tokenizers, embeddings, and taggers.
"""

from typing import (
    Protocol,
    List,
    Tuple,
    Union,
    Callable,
    Optional,
    Any,
    runtime_checkable,
)
import numpy as np


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for tokenizer classes.

    Any class implementing this protocol should provide:
    - tokenize(text) -> List[str]: Tokenize text into tokens
    - __call__(text) -> List[str]: Callable interface for tokenization
    """

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        ...

    def __call__(self, text: str) -> List[str]:
        """Callable interface for tokenization.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        ...


@runtime_checkable
class BatchTokenizerProtocol(Protocol):
    """Protocol for tokenizers with batch processing support."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize single text."""
        ...

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            List of token lists
        """
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for word embedding classes.

    Any class implementing this protocol should provide:
    - get_word_vector(word) -> np.ndarray: Get embedding for a word
    """

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get the embedding vector for a word.

        Args:
            word: Input word

        Returns:
            Embedding vector as numpy array
        """
        ...


@runtime_checkable
class BatchEmbeddingProtocol(Protocol):
    """Protocol for embeddings with batch processing support."""

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get embedding for single word."""
        ...

    def get_word_vectors(self, words: List[str]) -> np.ndarray:
        """Get embeddings for multiple words.

        Args:
            words: List of input words

        Returns:
            2D numpy array of shape (n_words, embedding_dim)
        """
        ...


@runtime_checkable
class SimilarityEmbeddingProtocol(Protocol):
    """Protocol for embeddings with similarity search support."""

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get embedding for a word."""
        ...

    def get_most_similar_words(
        self, word: str, topn: int = 10
    ) -> List[Tuple[str, float]]:
        """Get most similar words.

        Args:
            word: Input word
            topn: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples
        """
        ...


@runtime_checkable
class DocumentEmbeddingProtocol(Protocol):
    """Protocol for document embedding classes."""

    def get_document_vector(self, document: str) -> np.ndarray:
        """Get embedding vector for a document.

        Args:
            document: Input document text

        Returns:
            Document embedding vector
        """
        ...

    def get_document_similarity(self, document_1: str, document_2: str) -> float:
        """Get similarity between two documents.

        Args:
            document_1: First document
            document_2: Second document

        Returns:
            Similarity score (0-1)
        """
        ...


@runtime_checkable
class TaggerProtocol(Protocol):
    """Protocol for sequence tagging classes (POS, NER).

    Any class implementing this protocol should provide:
    - tag(text) -> List[Tuple[str, str]]: Tag tokens in text
    """

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """Tag tokens in text.

        Args:
            text: Input text

        Returns:
            List of (token, tag) tuples
        """
        ...


@runtime_checkable
class BatchTaggerProtocol(Protocol):
    """Protocol for taggers with batch processing support."""

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """Tag single text."""
        ...

    def tag_batch(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """Tag multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            List of tag lists
        """
        ...


@runtime_checkable
class TextProcessorProtocol(Protocol):
    """Protocol for text processing/cleaning classes."""

    def __call__(self, text: str) -> str:
        """Process/clean text.

        Args:
            text: Input text

        Returns:
            Processed text
        """
        ...


@runtime_checkable
class PipelineStepProtocol(Protocol):
    """Protocol for pipeline steps.

    Any class that can be used in a Pipeline should implement __call__.
    """

    def __call__(self, input_data: Any) -> Any:
        """Process input and return output.

        Args:
            input_data: Input from previous step (or initial input)

        Returns:
            Processed output for next step
        """
        ...
