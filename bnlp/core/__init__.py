"""
BNLP Core Module

This module provides core abstractions, protocols, and utilities for BNLP.
"""

from bnlp.core.protocols import (
    TokenizerProtocol,
    BatchTokenizerProtocol,
    EmbeddingProtocol,
    BatchEmbeddingProtocol,
    SimilarityEmbeddingProtocol,
    DocumentEmbeddingProtocol,
    TaggerProtocol,
    BatchTaggerProtocol,
    TextProcessorProtocol,
    PipelineStepProtocol,
)

from bnlp.core.pipeline import (
    Pipeline,
    PipelineStep,
    PipelineResult,
    create_tokenization_pipeline,
    create_ner_pipeline,
    create_pos_pipeline,
)

from bnlp.core.exceptions import (
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

from bnlp.core.batch import (
    BatchProcessor,
    tokenize_batch,
    embed_batch,
    tag_batch,
    clean_batch,
)

from bnlp.core.async_loader import (
    AsyncModelLoader,
    LazyModelLoader,
    LoadingStatus,
    LoadingProgress,
    load_model_async,
)

__all__ = [
    # Protocols
    "TokenizerProtocol",
    "BatchTokenizerProtocol",
    "EmbeddingProtocol",
    "BatchEmbeddingProtocol",
    "SimilarityEmbeddingProtocol",
    "DocumentEmbeddingProtocol",
    "TaggerProtocol",
    "BatchTaggerProtocol",
    "TextProcessorProtocol",
    "PipelineStepProtocol",
    # Pipeline
    "Pipeline",
    "PipelineStep",
    "PipelineResult",
    "create_tokenization_pipeline",
    "create_ner_pipeline",
    "create_pos_pipeline",
    # Exceptions
    "BNLPException",
    "ModelNotFoundError",
    "ModelLoadError",
    "TokenizationError",
    "EmbeddingError",
    "TaggingError",
    "DownloadError",
    "PipelineError",
    "InvalidInputError",
    # Batch Processing
    "BatchProcessor",
    "tokenize_batch",
    "embed_batch",
    "tag_batch",
    "clean_batch",
    # Async Loading
    "AsyncModelLoader",
    "LazyModelLoader",
    "LoadingStatus",
    "LoadingProgress",
    "load_model_async",
]
