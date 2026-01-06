# Bengali Natural Language Processing(BNLP)

[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![Downloads](https://static.pepy.tech/badge/bnlp_toolkit)](https://pepy.tech/project/bnlp_toolkit)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **Embedding Bengali Document**, **Bengali POS Tagging**, **Bengali Name Entity Recognition**, **Bangla Text Cleaning** for Bengali NLP purposes.


## Features
- Tokenization
   - [Basic Tokenizer](./docs/README.md#basic-tokenizer)
   - [NLTK Tokenizer](./docs/README.md#nltk-tokenization)
   - [Sentencepiece Tokenizer](./docs/README.md#bengali-sentencepiece-tokenization)
- Embeddings
   - [Word2vec embedding](./docs/README.md#bengali-word2vec)
   - [Fasttext embedding](./docs/README.md#bengali-fasttext)
   - [Glove Embedding](./docs/README.md#bengali-glove-word-vectors)
   - [Doc2vec Document embedding](./docs/README.md#document-embedding)
- Part of speech tagging
   - [CRF-based POS tagging](./docs/README.md#bengali-crf-pos-tagging)
- Named Entity Recognition
   - [CRF-based NER](./docs/README.md#bengali-crf-ner)
- [Text Cleaning](./docs/README.md#text-cleaning)
- [Corpus](./docs/README.md#bengali-corpus-class)
   - Letters, vowels, punctuations, stopwords
- [Command Line Interface (CLI)](#command-line-interface)
- [Pipeline API](#pipeline-api)
- [Batch Processing](#batch-processing)
- [Async Model Loading](#async-model-loading)
- [Spell Checking](#spell-checking)

## Installation

### PIP installer

  ```
  pip install bnlp_toolkit
  ```
  **or Upgrade**

  ```
  pip install -U bnlp_toolkit
  ```
  - Python: 3.8, 3.9, 3.10, 3.11
  - OS: Linux, Windows, Mac

### Build from source
```
git clone https://github.com/sagorbrur/bnlp.git
cd bnlp
python setup.py install
```

## Sample Usage

```py
from bnlp import BasicTokenizer

tokenizer = BasicTokenizer()

raw_text = "আমি বাংলায় গান গাই।"
tokens = tokenizer(raw_text)
print(tokens)
# output: ["আমি", "বাংলায়", "গান", "গাই", "।"]
```

## Command Line Interface

BNLP provides a command-line interface for quick text processing without writing Python code.

### Basic Usage

```bash
# Tokenize text
bnlp tokenize "আমি বাংলায় গান গাই।"
# Output: ['আমি', 'বাংলায়', 'গান', 'গাই', '।']

# Named Entity Recognition
bnlp ner "সজীব ওয়াজেদ জয় ঢাকায় থাকেন।"

# Part-of-Speech Tagging
bnlp pos "আমি ভাত খাই।"

# Get word embeddings (similar words)
bnlp embedding "বাংলা" --similar

# Clean text
bnlp clean "hello@example.com আমি বাংলায়" --remove-email

# Download models
bnlp download all          # Download all models
bnlp download word2vec     # Download specific model

# List available models
bnlp list-models

# Access corpus data
bnlp corpus stopwords
bnlp corpus letters
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `tokenize` | Tokenize Bengali text (supports: basic, nltk, sentencepiece) |
| `ner` | Named Entity Recognition |
| `pos` | Part-of-Speech tagging |
| `embedding` | Word embeddings (supports: word2vec, fasttext, glove) |
| `clean` | Text cleaning and normalization |
| `download` | Download pre-trained models |
| `list-models` | List all available models |
| `corpus` | Access Bengali corpus data (stopwords, letters, digits, etc.) |

### CLI Options

```bash
# Get help
bnlp --help
bnlp tokenize --help

# Output as JSON
bnlp tokenize "আমি বাংলায় গান গাই।" --json

# Use different tokenizer
bnlp tokenize "আমি বাংলায় গান গাই।" --type nltk

# Sentence tokenization
bnlp tokenize "আমি বাংলায় গান গাই। তুমি কি গাও?" --type nltk --sentence

# Get similar words with custom count
bnlp embedding "বাংলা" --similar --topn 5
```

## Pipeline API

Chain multiple NLP operations together using the Pipeline API.

```python
from bnlp import Pipeline, CleanText, BasicTokenizer

# Create a pipeline
pipeline = Pipeline([
    CleanText(remove_url=True, remove_punct=True),
    BasicTokenizer(),
])

# Process text through the pipeline
result = pipeline("আমি বাংলায় গান গাই।")
print(result)
# Output: ['আমি', 'বাংলায়', 'গান', 'গাই']

# Get detailed results with intermediate outputs
result = pipeline.run("আমি বাংলায় গান গাই।", return_details=True)
print(result.intermediate_results)
```

### Pre-built Pipelines

```python
from bnlp import create_tokenization_pipeline, create_ner_pipeline, create_pos_pipeline

# Tokenization pipeline
tokenizer_pipeline = create_tokenization_pipeline(clean=True, tokenizer_type="basic")
tokens = tokenizer_pipeline("আমি বাংলায় গান গাই।")

# NER pipeline
ner_pipeline = create_ner_pipeline(clean=True)
entities = ner_pipeline("সজীব ঢাকায় থাকেন।")

# POS pipeline
pos_pipeline = create_pos_pipeline(clean=True)
tags = pos_pipeline("আমি ভাত খাই।")
```

## Batch Processing

Process multiple texts efficiently using batch processing utilities.

```python
from bnlp import BasicTokenizer, tokenize_batch, tag_batch, clean_batch
from bnlp import BengaliNER, CleanText

# Batch tokenization
tokenizer = BasicTokenizer()
texts = ["আমি বাংলায় গান গাই।", "তুমি কোথায় যাও?", "সে বই পড়ে।"]
results = tokenize_batch(tokenizer.tokenize, texts)
print(results)
# Output: [['আমি', 'বাংলায়', ...], ['তুমি', 'কোথায়', ...], ['সে', 'বই', ...]]

# Batch NER tagging
ner = BengaliNER()
texts = ["সজীব ঢাকায় থাকেন।", "রবীন্দ্রনাথ ঠাকুর কলকাতায় জন্মগ্রহণ করেন।"]
results = tag_batch(ner.tag, texts)

# Batch text cleaning
cleaner = CleanText(remove_url=True, remove_email=True)
texts = ["email@example.com আমি", "https://example.com তুমি"]
results = clean_batch(cleaner, texts)
```

### Using BatchProcessor

```python
from bnlp import BatchProcessor, BasicTokenizer

tokenizer = BasicTokenizer()
batch = BatchProcessor(tokenizer.tokenize, max_workers=4)

texts = ["আমি বাংলায় গান গাই।"] * 100
results = batch.process(texts, show_progress=True)
```

## Async Model Loading

Load large models in the background without blocking your application.

```python
from bnlp import AsyncModelLoader, BengaliWord2Vec

# Create async loader with callbacks
def on_progress(progress):
    print(f"Loading: {progress.progress * 100:.0f}% - {progress.message}")

loader = AsyncModelLoader(
    BengaliWord2Vec,
    on_progress=on_progress,
    on_complete=lambda m: print("Model ready!")
)

# Start loading in background
loader.start_loading()

# Do other work while model loads...
print("Doing other work...")

# Get model when needed (blocks until ready)
model = loader.get_model()
vector = model.get_word_vector("বাংলা")
```

### Lazy Loading

```python
from bnlp import LazyModelLoader, BengaliWord2Vec

# Model not loaded yet
lazy_model = LazyModelLoader(BengaliWord2Vec)

# Model loads on first access
model = lazy_model.get()
vector = model.get_word_vector("বাংলা")
```

### Quick Async Loading

```python
from bnlp import load_model_async, BengaliWord2Vec

# One-liner to start async loading
loader = load_model_async(BengaliWord2Vec)

# Get model when ready
model = loader.get_model()
```

## Spell Checking

Fast and accurate Bengali spell checking using the SymSpell algorithm.

```python
from bnlp import BengaliSpellChecker

# Create spell checker
checker = BengaliSpellChecker()

# Check if a word is spelled correctly
print(checker.is_correct("আমি"))  # True
print(checker.is_correct("আমর"))  # False (misspelled)

# Get spelling suggestions
suggestions = checker.suggestions("আমর")
print(suggestions)
# Output: [('আমি', 1), ('আমার', 2), ...]

# Check text for errors
text = "আমর বাংলায় গান গাই।"
errors = checker.check(text)
for error in errors:
    print(f"{error.word} -> {error.best_correction}")
# Output: আমর -> আমি

# Automatically correct text
corrected = checker.correct("আমর বাংলায় গান গাই।")
print(corrected)
# Output: আমি বাংলায় গান গাই।
```

### Custom Dictionary

```python
from bnlp import BengaliSpellChecker

# Add custom words
checker = BengaliSpellChecker()
checker.add_word("কাস্টমশব্দ", frequency=1000)
checker.add_words(["নতুনশব্দ", "আরেকটি"], default_frequency=500)

# Or pass custom words during initialization
custom_words = {"বিএনএলপি": 1000, "এনএলপি": 900}
checker = BengaliSpellChecker(custom_words=custom_words)

# Load dictionary from file
checker.load_dictionary("my_dictionary.txt")
```

### Spell Checker Options

```python
from bnlp import BengaliSpellChecker

# Customize edit distance (default: 2)
checker = BengaliSpellChecker(max_edit_distance=1)  # Faster, less suggestions

# Get word probability
prob = checker.word_probability("আমি")
print(prob)  # Higher = more common word
```

## Documentation
Full documentation are available [here](https://sagorbrur.github.io/bnlp/)

If you are using previous version of **bnlp** check the documentation [archive](https://sagorbrur.github.io/bnlp/docs/archive)

## Contributor Guide

Check [CONTRIBUTING.md](https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md) page for details.


## Thanks To

* [Semantics Lab](https://www.facebook.com/lab.semantics/)
* All the developers who are contributing to enrich Bengali NLP.
