"""
BNLP Batch Processing Utilities

This module provides batch processing capabilities for BNLP operations.
"""

from typing import List, Tuple, Callable, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np


class BatchProcessor:
    """A utility class for batch processing of NLP operations.

    This class provides methods to process multiple inputs efficiently
    using parallel processing.

    Example:
        >>> from bnlp import BasicTokenizer
        >>> from bnlp.core import BatchProcessor
        >>> tokenizer = BasicTokenizer()
        >>> batch = BatchProcessor(tokenizer.tokenize)
        >>> texts = ["আমি বাংলায় গান গাই।", "তুমি কোথায় যাও?"]
        >>> results = batch.process(texts)
    """

    def __init__(
        self,
        func: Callable[[Any], Any],
        use_multiprocessing: bool = False,
        max_workers: Optional[int] = None,
    ):
        """Initialize BatchProcessor.

        Args:
            func: The function to apply to each input
            use_multiprocessing: Use ProcessPoolExecutor instead of ThreadPoolExecutor
            max_workers: Maximum number of workers (default: None = auto)
        """
        self.func = func
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers

    def process(
        self,
        inputs: List[Any],
        show_progress: bool = False,
    ) -> List[Any]:
        """Process multiple inputs in parallel.

        Args:
            inputs: List of inputs to process
            show_progress: Show progress bar (requires tqdm)

        Returns:
            List of results
        """
        if not inputs:
            return []

        # For small batches, use simple loop
        if len(inputs) <= 2:
            return [self.func(x) for x in inputs]

        executor_class = (
            ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.max_workers) as executor:
            if show_progress:
                try:
                    from tqdm import tqdm

                    results = list(tqdm(executor.map(self.func, inputs), total=len(inputs)))
                except ImportError:
                    results = list(executor.map(self.func, inputs))
            else:
                results = list(executor.map(self.func, inputs))

        return results

    def __call__(self, inputs: List[Any]) -> List[Any]:
        """Callable interface for batch processing."""
        return self.process(inputs)


def tokenize_batch(
    tokenizer: Callable[[str], List[str]],
    texts: List[str],
    max_workers: Optional[int] = None,
) -> List[List[str]]:
    """Tokenize multiple texts in batch.

    Args:
        tokenizer: Tokenizer function or callable
        texts: List of texts to tokenize
        max_workers: Maximum number of workers

    Returns:
        List of token lists
    """
    processor = BatchProcessor(tokenizer, max_workers=max_workers)
    return processor.process(texts)


def embed_batch(
    embedder: Callable[[str], np.ndarray],
    words: List[str],
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """Get embeddings for multiple words.

    Args:
        embedder: Embedding function that returns numpy array
        words: List of words
        max_workers: Maximum number of workers

    Returns:
        2D numpy array of shape (n_words, embedding_dim)
    """
    processor = BatchProcessor(embedder, max_workers=max_workers)
    vectors = processor.process(words)
    return np.vstack(vectors)


def tag_batch(
    tagger: Callable[[str], List[Tuple[str, str]]],
    texts: List[str],
    max_workers: Optional[int] = None,
) -> List[List[Tuple[str, str]]]:
    """Tag multiple texts in batch.

    Args:
        tagger: Tagging function
        texts: List of texts to tag
        max_workers: Maximum number of workers

    Returns:
        List of tag lists
    """
    processor = BatchProcessor(tagger, max_workers=max_workers)
    return processor.process(texts)


def clean_batch(
    cleaner: Callable[[str], str],
    texts: List[str],
    max_workers: Optional[int] = None,
) -> List[str]:
    """Clean multiple texts in batch.

    Args:
        cleaner: Text cleaning function
        texts: List of texts to clean
        max_workers: Maximum number of workers

    Returns:
        List of cleaned texts
    """
    processor = BatchProcessor(cleaner, max_workers=max_workers)
    return processor.process(texts)
