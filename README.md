# Bengali Natural Language Processing(BNLP)

[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![Downloads](https://pepy.tech/badge/bnlp-toolkit)](https://pepy.tech/project/bnlp-toolkit)

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

## Installation

### PIP installer

  ```
  pip install bnlp_toolkit
  ```
  **or Upgrade**

  ```
  pip install -U bnlp_toolkit
  ```
  - Python: 3.6, 3.7, 3.8, 3.9, 3.10
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

## Documentation
Full documentation are available [here](https://github.com/sagorbrur/bnlp/tree/master/docs)

If you are using previous version of **bnlp** check the documentation [archive](https://github.com/sagorbrur/bnlp/tree/master/docs/archive)

## Contributor Guide

Check [CONTRIBUTING.md](https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md) page for details.


## Thanks To

* [Semantics Lab](https://www.facebook.com/lab.semantics/)
* All the developers who are contributing to enrich Bengali NLP.
