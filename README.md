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

## Documentation
Full documentation are available [here](https://sagorbrur.github.io/bnlp/)

If you are using previous version of **bnlp** check the documentation [archive](https://sagorbrur.github.io/bnlp/docs/archive)

## Contributor Guide

Check [CONTRIBUTING.md](https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md) page for details.


## Thanks To

* [Semantics Lab](https://www.facebook.com/lab.semantics/)
* All the developers who are contributing to enrich Bengali NLP.
