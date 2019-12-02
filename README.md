# Bengali Natural Language Processing(BNLP)

[![Build Status](https://travis-ci.org/sagorbrur/bnlp.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlp)
[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![release version](https://img.shields.io/github/v/release/sagorbrur/bnlp)](https://github.com/sagorbrur/bnlp/releases/tag/1.1.0)
[![Support Python Version](https://img.shields.io/badge/python-3.6%7C3.7-blue)](https://pypi.org/project/bnlp-toolkit/)
[![pypi Downloads](https://img.shields.io/pypi/dw/bnlp_toolkit?color=green)](https://pypi.org/project/bnlp-toolkit/)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **construct neural model** for Bengali NLP purposes.


## Current Features
* Bengali Tokenization
* Bengali Word Embedding

## Installation

* pypi package installer(python 3.6, 3.7 tested okay)

  ```pip install bnlp_toolkit```


## Pretrained Model

### Download Link

* [Bengali SentencePiece](https://github.com/sagorbrur/bnlp/tree/master/model)
* [Bengali Word2Vec](https://drive.google.com/open?id=1DxR8Vw61zRxuUm17jzFnOX97j7QtNW7U)
* [Bengali FastText](https://drive.google.com/open?id=1KRA91w6dMpuQpowOwLCRplRgSdRzyOYz)

### Training Details
* All three model trained with **Bengali Wikipedia Dump Dataset**
  - [Bengali Wiki Dump](https://dumps.wikimedia.org/bnwiki/latest/)
* SentencePiece Training Vocab Size=50000
* Word2Vec and Fasttext word embedding dimension = 300


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

  ```


## Word Embedding

* **Bengali Word2Vec**

  - Generate Vector using pretrain model

    ```py
    from bnlp.bengali_word2vec import Bengali_Word2Vec

    bwv = Bengali_Word2Vec()
    model_path = "model/wiki.bn.text.model"
    word = 'আমার'
    vector = bwv.generate_word_vector(model_path, word)
    print(vector.shape)
    print(vector)

    ```

  - Find Most Similar Word Using Pretrained Model

    ```py
    from bnlp.bengali_word2vec import Bengali_Word2Vec

    bwv = Bengali_Word2Vec()
    model_path = "model/wiki.bn.text.model"
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
      model_path = "cc.bn.300.bin"
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

## Issue
* if `ModuleNotFoundError: No module named 'fasttext'` problem arise please do the next line

```pip install fasttext```
* if `nltk` issue arise please do the following line before importing `bnlp`

```py
import nltk
nltk.download("punkt")
```


## Developer Guide

* `Fork`
* `add` or `modify`
* send `pull request` for merging


## Thanks To

* [Semantics Lab](http://semanticslab.net/)

## Contributor List

* [Sagor Sarker](https://github.com/sagorbrur)
* [Faruk Ahmad](https://github.com/faruk-ahmad)
* [Mehadi Hasan Menon](https://github.com/menon92)
* [Kazal Chandra Barman](https://github.com/kazalbrur)
* [Md Ibrahim](https://github.com/iriad11)
