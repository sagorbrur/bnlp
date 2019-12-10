# Bengali Natural Language Processing(BNLP)

[![Build Status](https://travis-ci.org/sagorbrur/bnlp.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlp)
[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![release version](https://img.shields.io/github/v/release/sagorbrur/bnlp)](https://github.com/sagorbrur/bnlp/releases/tag/1.1.0)
[![Support Python Version](https://img.shields.io/badge/python-3.6%7C3.7-brightgreen)](https://pypi.org/project/bnlp-toolkit/)
[![pypi Downloads](https://img.shields.io/pypi/dw/bnlp_toolkit?color=green)](https://pypi.org/project/bnlp-toolkit/)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **construct neural model** for Bengali NLP purposes.


## Current Features
* [Bengali Tokenization](#tokenization)
  - SentencePiece Tokenizer
  - Basic Tokenizer
  - NLTK Tokenizer
* [Bengali Word Embedding](#word-embedding)
  - Bengali Word2Vec
  - Bengali Fasttext
  - Bengali GloVe


## Installation

* pypi package installer(python 3.6, 3.7 tested okay)

  ```pip install bnlp_toolkit```


## Pretrained Model

### Download Link

* [Bengali SentencePiece](https://github.com/sagorbrur/bnlp/tree/master/model)
* [Bengali Word2Vec](https://drive.google.com/open?id=1DxR8Vw61zRxuUm17jzFnOX97j7QtNW7U)
* [Bengali FastText](https://drive.google.com/open?id=1DZ6i9G1CPRCk9-YBa9_P-vGPreK8hmw8)
* [Bengali GloVe Wordvectors](https://github.com/sagorbrur/GloVe-Bengali)

### Training Details
* All three model trained with **Bengali Wikipedia Dump Dataset**
  - [Bengali Wiki Dump](https://dumps.wikimedia.org/bnwiki/latest/)
* SentencePiece Training Vocab Size=50000
* Fasttext vocab size = 1508445, training loss = 0.840629, embedding dimension = 300
* Word2Vec word embedding dimension = 300
* To Know Bengali GloVe Wordvector and training process follow [this](https://github.com/sagorbrur/GloVe-Bengali) repository


## Tokenization

* **Bengali SentencePiece Tokenization**

  - tokenization using trained model
    ```py
    from bnlp.sentencepiece_tokenizer import SP_Tokenizer

    bsp = SP_Tokenizer()
    model_path = "./model/bn_spm.model"
    input_text = "আমি ভাত খাই। সে বাজারে যায়।"
    tokens = bsp.tokenize(model_path, input_text)
    print(tokens)

    ```
  - Training SentencePiece
    ```py
    from bnlp.sentencepiece_tokenizer import SP_Tokenizer
    
    bsp = SP_Tokenizer(is_train=True)
    data = "test.txt"
    model_prefix = "test"
    vocab_size = 5
    bsp.train_bsp(data, model_prefix, vocab_size) 

    ```

* **Basic Tokenizer**

 

  ```py
  from bnlp.basic_tokenizer import BasicTokenizer
  basic_t = BasicTokenizer(False)
  raw_text = "আমি বাংলায় গান গাই।"
  tokens = basic_t.tokenize(raw_text)
  print(tokens)
  
  # output: ["আমি", "বাংলায়", "গান", "গাই", "।"]

  ```

* **NLTK Tokenization**

  ```py
  from bnlp.nltk_tokenizer import NLTK_Tokenizer

  text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"
  bnltk = NLTK_Tokenizer(text)
  word_tokens = bnltk.word_tokenize()
  sentence_tokens = bnltk.sentence_tokenize()
  print(word_tokens)
  print(sentence_tokens)
  
  # output
  # word_token: ["আমি", "ভাত", "খাই", "।", "সে", "বাজারে", "যায়", "।", "তিনি", "কি", "সত্যিই", "ভালো", "মানুষ", "?"]
  # sentence_token: ["আমি ভাত খাই।", "সে বাজারে যায়।", "তিনি কি সত্যিই ভালো মানুষ?"]

  ```


## Word Embedding

* **Bengali Word2Vec**

  - Generate Vector using pretrain model

    ```py
    from bnlp.bengali_word2vec import Bengali_Word2Vec

    bwv = Bengali_Word2Vec()
    model_path = "model/bengali_word2vec.model"
    word = 'আমার'
    vector = bwv.generate_word_vector(model_path, word)
    print(vector.shape)
    print(vector)

    ```

  - Find Most Similar Word Using Pretrained Model

    ```py
    from bnlp.bengali_word2vec import Bengali_Word2Vec

    bwv = Bengali_Word2Vec()
    model_path = "model/bengali_word2vec.model"
    word = 'আমার'
    similar = bwv.most_similar(model_path, word)
    print(similar)

    ```
  - Train Bengali Word2Vec with your own data

    ```py
    from bnlp.bengali_word2vec import Bengali_Word2Vec
    bwv = Bengali_Word2Vec(is_train=True)
    data_file = "test.txt"
    model_name = "test_model.model"
    vector_name = "test_vector.vector"
    bwv.train_word2vec(data_file, model_name, vector_name)


    ```
    
 * **Bengali FastText**
 

    - Generate Vector Using Pretrained Model
      

      ```py
      from bnlp.bengali_fasttext import Bengali_Fasttext

      bft = Bengali_Fasttext()
      word = "গ্রাম"
      model_path = "model/bengali_fasttext.bin"
      word_vector = bft.generate_word_vector(model_path, word)
      print(word_vector.shape)
      print(word_vector)


      ```
    - Train Bengali FastText Model

      ```py
      from bnlp.bengali_fasttext import Bengali_Fasttext

      bft = Bengali_Fasttext(is_train=True)
      data = "data.txt"
      model_name = "saved_model.bin"
      bft.train_fasttext(data, model_name)

      ```

* **Bengali GloVe Word Vectors**

  We trained glove model with bengali data(wiki+news articles) and published bengali glove word vectors</br>
  You can download and use it on your different machine learning purposes.

  ```py
  from bnlp.glove_wordvector import BN_Glove
  glove_path = "bn_glove.39M.100d.txt"
  word = "গ্রাম"
  bng = BN_Glove()
  res = bng.closest_word(glove_path, word)
  print(res)
  vec = bng.word2vec(glove_path, word)
  print(vec)

  ```

## Issue
* if `ModuleNotFoundError: No module named 'fasttext'` problem arise please do the next line

```pip install fasttext```
* if `nltk` issue arise please do the following line before importing `bnlp`

```py
import nltk
nltk.download("punkt")
```


## Developer Guide

Check [CONTRIBUTING](https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md) page for details.


## Thanks To

* [Semantics Lab](http://semanticslab.net/)

## Contributor List

* [Sagor Sarker](https://github.com/sagorbrur)
* [Faruk Ahmad](https://github.com/faruk-ahmad)
* [Mehadi Hasan Menon](https://github.com/menon92)
* [Kazal Chandra Barman](https://github.com/kazalbrur)
* [Md Ibrahim](https://github.com/iriad11)
