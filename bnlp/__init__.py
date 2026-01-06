
__version__ = "4.4.0"

import os
from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.tokenizer.nltk import NLTKTokenizer
from bnlp.tokenizer.sentencepiece import (
    SentencepieceTokenizer,
    SentencepieceTrainer,
)

from bnlp.embedding.word2vec import (
    BengaliWord2Vec,
    Word2VecTraining,
)
from bnlp.embedding.glove import BengaliGlove

from bnlp.embedding.doc2vec import (
    BengaliDoc2vec,
    BengaliDoc2vecTrainer,
)

from bnlp.token_classification.pos import BengaliPOS
from bnlp.token_classification.ner import BengaliNER
from bnlp.token_classification.token_classification_trainer import CRFTaggerTrainer

from bnlp.cleantext.clean import CleanText

from bnlp.corpus.corpus import BengaliCorpus

# Lazy imports for optional dependencies (spell checking, language detection)
# These are loaded on-demand to avoid requiring symspellpy/fasttext at package load
def __getattr__(name):
    """Lazy load optional modules."""
    # Spell checking (requires symspellpy)
    if name == "BengaliSpellChecker":
        from bnlp.spellcheck import BengaliSpellChecker
        return BengaliSpellChecker
    elif name == "SpellingError":
        from bnlp.spellcheck import SpellingError
        return SpellingError
    # Language detection (requires fasttext)
    elif name == "LanguageDetector":
        from bnlp.langdetect import LanguageDetector
        return LanguageDetector
    elif name == "DetectionResult":
        from bnlp.langdetect import DetectionResult
        return DetectionResult
    elif name == "detect_language":
        from bnlp.langdetect import detect_language
        return detect_language
    elif name == "is_bengali":
        from bnlp.langdetect import is_bengali
        return is_bengali
    raise AttributeError(f"module 'bnlp' has no attribute '{name}'")

# Core module - Protocols, Pipeline, Exceptions, Batch Processing, Async Loading
from bnlp.core import (
    # Pipeline
    Pipeline,
    PipelineStep,
    PipelineResult,
    create_tokenization_pipeline,
    create_ner_pipeline,
    create_pos_pipeline,
    # Batch Processing
    BatchProcessor,
    tokenize_batch,
    embed_batch,
    tag_batch,
    clean_batch,
    # Async Loading
    AsyncModelLoader,
    LazyModelLoader,
    load_model_async,
    # Exceptions
    BNLPException,
    ModelNotFoundError,
    ModelLoadError,
)
