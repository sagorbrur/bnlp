# Bengali Natural Language Processing(BNLP)

[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![Downloads](https://pepy.tech/badge/bnlp-toolkit)](https://pepy.tech/project/bnlp-toolkit)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **Embedding Bengali Document**, **Bengali POS Tagging**, **Bengali Name Entity Recognition**, **Bangla Text Cleaning** for Bengali NLP purposes.


## Documentation
Full documentation are available [here](./docs/README.md)

## Features
- Tokenization
   - Basic Tokenizer
   - NLTK Tokenizer
   - Sentencepiece Tokenizer
- Embeddings
   - Word2vec embedding
   - Fasttext embedding
   - Glove Embedding
   - Doc2vec embedding
- Part of speech tagging
   - CRF-based POS tagging
- Named Entity Recognition
   - CRF-based NER
- Text Cleaning
- Corpus
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