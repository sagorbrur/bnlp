# BNLP Package Refactoring Analysis

## Executive Summary

**Package**: BNLP (Bengali Natural Language Processing Toolkit)
**Version**: 4.0.3
**Total Source Files**: 21 Python modules
**Test Files**: 9 test modules
**Lines of Code**: ~1,600 (excluding stopwords data)

---

## 1. Architecture Overview

### Current Structure
```
bnlp/
├── tokenizer/          # 3 tokenizers (Basic, NLTK, SentencePiece)
├── embedding/          # 4 embeddings (Word2Vec, FastText, GloVe, Doc2Vec)
├── token_classification/  # 2 taggers (POS, NER) + trainer
├── cleantext/          # Text cleaning utilities
├── corpus/             # Static Bengali linguistic data
└── utils/              # Config, downloader, feature extraction
```

---

## 2. Code Quality Issues & Refactoring Opportunities

### 2.1 Legacy Python 2 Compatibility Code

**Location**: `bnlp/tokenizer/basic.py:7-28`

```python
import six  # Python 2/3 compatibility library

def convert_to_unicode(text):
    if six.PY3:
        ...
    elif six.PY2:  # Dead code - Python 2 is EOL
        ...
```

**Issue**: The `six` library and Python 2 code paths are unnecessary since `python_requires=">=3.6"`.

**Backward Compatibility**: Safe to remove - no Python 2 users.

---

### 2.2 Duplicate DUMMYTOKEN Constants

**Location**: `bnlp/tokenizer/nltk.py:10-12`

```python
DUMMYTOKEN = "XTEMPTOKEN"
DUMMYTOKEN = "XTEMPDOT"  # Overwrites previous line!
```

**Issue**: Dead code - first assignment is immediately overwritten.

---

### 2.3 Inconsistent Warning Suppression

**Locations**:
- `bnlp/embedding/word2vec.py:6-7`
- `bnlp/embedding/doc2vec.py:1-2`

```python
import warnings
warnings.filterwarnings("ignore")  # Suppresses ALL warnings globally
```

**Issue**: Global warning suppression hides important deprecation/runtime warnings from users.

---

### 2.4 Missing Abstract Base Classes / Protocol Definitions

**Current State**: No common interface for tokenizers or embeddings.

**Impact**: Each tokenizer has different method signatures:

| Class | Methods |
|-------|---------|
| `BasicTokenizer` | `tokenize()`, `__call__()` |
| `NLTKTokenizer` | `word_tokenize()`, `sentence_tokenize()` |
| `SentencepieceTokenizer` | `tokenize()`, `text2id()`, `id2text()` |

**Opportunity**: Define `Protocol` classes for type safety and polymorphism.

---

### 2.5 Unused Imports

**Location**: `bnlp/embedding/word2vec.py:11`
```python
import multiprocessing  # Never used
import sys  # Never used
```

**Location**: `bnlp/token_classification/ner.py:8`
```python
from sklearn_crfsuite import CRF  # Imported but only pickle model used
```

---

### 2.6 Module-Level Default Instance

**Location**: `bnlp/embedding/doc2vec.py:17`

```python
default_tokenizer = BasicTokenizer()  # Created at import time
```

**Issue**: Side effect at import time, harder to test/mock.

---

### 2.7 Hardcoded Print Statements

**Locations**: Multiple files use `print()` for logging instead of the `logging` module.

Examples:
- `bnlp/embedding/word2vec.py:109-110`: `print("training started.......")`
- `bnlp/embedding/fasttext.py:7-10`: `print("fasttext not installed...")`
- `bnlp/utils/downloader.py`: Multiple print statements

**Issue**: No control over verbosity, cannot redirect to log files.

---

### 2.8 File Handle Resource Management

**Location**: `bnlp/embedding/fasttext.py:37-49`

```python
output_vector = open(vector_name, "w")
# ... operations ...
output_vector.close()  # Not in try/finally or context manager
```

**Issue**: File may not be closed if exception occurs.

---

### 2.9 Bare Exception Handling

**Location**: `bnlp/utils/utils.py:34`

```python
except Exception as e:
    print(e)  # Swallows all exceptions silently
```

**Location**: `bnlp/utils/downloader.py:51`

```python
except Exception as zip_error:  # Too broad
```

---

### 2.10 Type Hint Inconsistencies

**Current Coverage**: Partial - some functions have hints, others don't.

Examples of missing hints:
- `bnlp/tokenizer/basic.py:40`: `_is_punctuation(char)` - no return type
- `bnlp/embedding/doc2vec.py:20`: `_read_corpus()` - generator not typed

---

## 3. Testability Issues

### 3.1 Network-Dependent Tests

**All current tests require network access** to download pre-trained models:

```python
# test_word2vec.py
def setUp(self):
    self.word2vec = BengaliWord2Vec()  # Downloads ~500MB model!
```

**Problems**:
- Tests are slow (download time)
- Tests fail without internet
- Cannot run in CI/CD without caching
- No isolation between test runs

---

### 3.2 No Mock/Fixture Infrastructure

**Current State**: No `conftest.py`, no pytest fixtures, no mocking.

**Missing**:
- Model mocking for unit tests
- Fixture for sample Bengali text
- Parametrized test cases
- Test data files

---

### 3.3 Limited Test Coverage

| Module | Test Coverage |
|--------|--------------|
| `tokenizer/basic.py` | 3 tests |
| `tokenizer/nltk.py` | 1 test (likely) |
| `tokenizer/sentencepiece.py` | 1 test (likely) |
| `embedding/word2vec.py` | 2 tests |
| `embedding/fasttext.py` | 1 test |
| `embedding/glove.py` | 1 test |
| `embedding/doc2vec.py` | 1 test |
| `token_classification/pos.py` | 1 test |
| `token_classification/ner.py` | 1 test |
| **cleantext/** | **0 tests** |
| **corpus/** | **0 tests** |
| **utils/** | **0 tests** |
| **trainers** | **0 tests** |

**Estimated Coverage**: ~30-40%

---

### 3.4 No Edge Case Testing

Current tests only cover happy paths:
- No empty string handling tests
- No Unicode edge case tests
- No error condition tests
- No boundary value tests

---

## 4. Backward Compatibility Considerations

### 4.1 Public API Surface

**Stable Exports** (from `__init__.py`):
```python
BasicTokenizer, NLTKTokenizer, SentencepieceTokenizer, SentencepieceTrainer
BengaliWord2Vec, Word2VecTraining, BengaliGlove, BengaliFasttext
BengaliDoc2vec, BengaliDoc2vecTrainer
BengaliPOS, BengaliNER, CRFTaggerTrainer
CleanText, BengaliCorpus
```

**Safe to Change** (internal):
- `utils/` module internals
- Private methods (prefixed with `_`)
- Internal constants like `DUMMYTOKEN`

**Must Preserve**:
- All class names and their method signatures
- Constructor parameters (especially `model_path`, `tokenizer`)
- Return types of all public methods

---

### 4.2 Deprecation Strategy for Changes

For any signature changes, use:
```python
import warnings

def old_method(self, ...):
    warnings.warn(
        "old_method is deprecated, use new_method instead",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method(...)
```

---

## 5. Recommended New Features

### 5.1 Pipeline API

**Rationale**: Users often chain operations (clean -> tokenize -> embed).

```python
# Proposed usage
pipeline = bnlp.Pipeline([
    CleanText(remove_punct=True),
    BasicTokenizer(),
    BengaliWord2Vec()
])
result = pipeline("আমি বাংলায় গান গাই।")
```

---

### 5.2 Batch Processing Support

**Current State**: All methods process single inputs.

**Proposed Addition**:
```python
def get_word_vectors(self, words: List[str]) -> np.ndarray:
    """Vectorized batch processing"""
    ...

def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
    """Parallel tokenization"""
    ...
```

---

### 5.3 Transformer-Based Models Integration

**Opportunity**: Add support for modern Bengali transformer models.

```python
# Proposed
from bnlp.transformers import BanglaBERT, BanglaT5

model = BanglaBERT()
embeddings = model.encode("আমি বাংলায় গান গাই।")
```

**Models to Consider**:
- `sagorsarker/bangla-bert-base`
- `csebuetnlp/banglabert`
- `csebuetnlp/banglat5`

---

### 5.4 Async Model Loading

**Rationale**: Models are large, async loading improves UX.

```python
# Proposed
async def load_model(model_type: str) -> Model:
    ...

# Or with progress callback
model = BengaliWord2Vec(
    on_progress=lambda pct: print(f"Loading: {pct}%")
)
```

---

### 5.5 Text Normalization Module

**Opportunity**: Bengali-specific normalization beyond cleaning.

```python
class BengaliNormalizer:
    def normalize_unicode(self, text: str) -> str:
        """Handle NFC/NFD normalization for Bengali"""

    def normalize_numbers(self, text: str) -> str:
        """Convert ১২৩ <-> 123"""

    def normalize_spelling(self, text: str) -> str:
        """Handle common spelling variations"""
```

---

### 5.6 Sentence Embeddings

**Current Gap**: Only word-level and document-level embeddings.

```python
# Proposed
class BengaliSentenceTransformer:
    def encode(self, sentences: List[str]) -> np.ndarray:
        """Sentence-level embeddings using transformer models"""
```

---

### 5.7 Text Augmentation

**Opportunity**: Data augmentation for Bengali NLP.

```python
class BengaliAugmenter:
    def synonym_replace(self, text: str) -> str: ...
    def random_swap(self, text: str) -> str: ...
    def back_translation(self, text: str) -> str: ...
```

---

### 5.8 Language Detection

**Opportunity**: Detect Bengali vs other languages/scripts.

```python
def detect_language(text: str) -> str:
    """Returns 'bn', 'en', 'mixed', etc."""

def is_bengali(text: str) -> bool:
    """Check if text is primarily Bengali"""
```

---

### 5.9 Spell Checking / Correction

```python
class BengaliSpellChecker:
    def check(self, text: str) -> List[SpellingError]: ...
    def correct(self, text: str) -> str: ...
    def suggestions(self, word: str) -> List[str]: ...
```

---

### 5.10 CLI Tool Enhancement

**Current**: No CLI interface.

**Proposed**:
```bash
$ bnlp tokenize "আমি বাংলায় গান গাই।"
["আমি", "বাংলায়", "গান", "গাই", "।"]

$ bnlp ner "সজীব ওয়াজেদ জয় ঢাকায় থাকেন।"
[("সজীব", "B-PER"), ("ওয়াজেদ", "I-PER"), ...]

$ bnlp download all
```

---

## 6. Dependency Updates

### 6.1 Pinned Version Risks

**Current Issues**:

| Dependency | Pinned | Latest | Risk |
|------------|--------|--------|------|
| `scipy>=1.11.0` | No | 1.13+ | Updated for Python 3.12+ |
| `gensim>=4.3.3` | No | 4.4+ | Updated for scipy compatibility |
| `emoji>=2.0.0` | No | 2.15+ | Updated, code migrated to new API |
| `sklearn-crfsuite>=0.5.0` | No | 0.5+ | Updated for Python 3.12+ |

### 6.2 Recommended Approach

```python
install_requires=[
    "sentencepiece>=0.2.0",
    "gensim>=4.3.3",
    "nltk",
    "numpy",
    "scipy>=1.11.0",
    "sklearn-crfsuite>=0.5.0",
    "tqdm>=4.66.3",
    "ftfy>=6.2.0",
    "emoji>=2.0.0",
    "requests",
    "symspellpy>=6.7.0",
],
```

---

## 7. Proposed Refactoring Phases

### Phase 1: Code Cleanup (Non-Breaking)
- Remove Python 2 compatibility code
- Fix duplicate constants
- Replace print statements with logging
- Add missing type hints
- Fix resource management (context managers)
- Remove unused imports

### Phase 2: Testing Infrastructure
- Add pytest configuration
- Create model mocking infrastructure
- Add fixtures for Bengali text samples
- Implement unit tests for all modules
- Add integration tests with small test models
- Set up CI/CD with cached models

### Phase 3: API Improvements (Backward Compatible)
- Add `Protocol` definitions for interfaces
- Add batch processing methods (new methods, don't change existing)
- Add async loading options
- Add Pipeline API
- Improve error messages and exceptions

### Phase 4: New Features
- Add transformer model support
- Add sentence embeddings
- Add text augmentation
- Add spell checking
- Enhance CLI

---

## 8. Proposed Directory Structure After Refactoring

```
bnlp/
├── __init__.py              # Public API exports
├── _version.py              # Version info
├── core/                    # NEW: Core abstractions
│   ├── protocols.py         # Protocol definitions
│   ├── pipeline.py          # Pipeline API
│   └── base.py              # Base classes
├── tokenizer/
│   ├── base.py              # NEW: TokenizerProtocol
│   ├── basic.py
│   ├── nltk.py
│   └── sentencepiece.py
├── embedding/
│   ├── base.py              # NEW: EmbeddingProtocol
│   ├── word2vec.py
│   ├── fasttext.py
│   ├── glove.py
│   ├── doc2vec.py
│   └── sentence.py          # NEW: Sentence embeddings
├── token_classification/
│   ├── base.py              # NEW: TaggerProtocol
│   ├── pos.py
│   ├── ner.py
│   └── trainer.py           # Renamed for clarity
├── transformers/            # NEW: Modern models
│   ├── bert.py
│   └── t5.py
├── cleantext/
│   ├── clean.py
│   ├── normalize.py         # NEW: Normalization
│   └── constants.py
├── augmentation/            # NEW
│   └── augmenter.py
├── corpus/
│   ├── corpus.py
│   └── _stopwords.py
├── utils/
│   ├── config.py
│   ├── downloader.py
│   ├── utils.py
│   └── logging.py           # NEW: Centralized logging
├── cli/                     # NEW: Command-line interface
│   └── main.py
└── py.typed                 # NEW: PEP 561 marker

tests/
├── conftest.py              # NEW: Pytest fixtures
├── fixtures/                # NEW: Test data
│   ├── sample_texts.py
│   └── mock_models.py
├── unit/                    # NEW: Unit tests (mocked)
│   ├── test_basic_tokenizer.py
│   ├── test_cleantext.py
│   └── ...
└── integration/             # NEW: Integration tests
    ├── test_word2vec_integration.py
    └── ...
```

---

## 9. Summary of Key Recommendations

| Priority | Category | Recommendation |
|----------|----------|----------------|
| **High** | Testing | Add pytest fixtures and model mocking |
| **High** | Testing | Achieve 80%+ code coverage |
| **High** | Cleanup | Remove Python 2 code, fix warnings |
| **Medium** | API | Add Protocol definitions for type safety |
| **Medium** | API | Add batch processing support |
| **Medium** | Logging | Replace print with logging module |
| **Medium** | Deps | Use version ranges instead of pins |
| **Low** | Features | Add transformer model support |
| **Low** | Features | Add CLI tool |
| **Low** | Features | Add Pipeline API |

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing user code | Strict semantic versioning, deprecation warnings |
| Large model downloads in tests | Mock models, cached fixtures |
| Dependency conflicts | Test against multiple Python/dep versions |
| Transformer models increase package size | Make optional (extras_require) |

---

*Analysis prepared: January 2026*
*Package version analyzed: 4.0.3*
