<img align="left" height="70" src="bnlp.svg" alt="bnlp"/>

# Bengali Natural Language Processing(BNLP)

[![Build Status](https://travis-ci.org/sagorbrur/bnlp.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlp)
[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![release version](https://img.shields.io/github/v/release/sagorbrur/bnlp)](https://github.com/sagorbrur/bnlp/releases/tag/2.0.0)
[![Support Python Version](https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-brightgreen)](https://pypi.org/project/bnlp-toolkit/)
[![Documentation Status](https://readthedocs.org/projects/bnlp/badge/?version=latest)](https://bnlp.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/bnlp_toolkit/community.svg)](https://gitter.im/bnlp_toolkit/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **Bengali POS Tagging**, **Construct Neural Model** for Bengali NLP purposes.


# Contents
- [Current Features](#current-features)
- [Installation](#installation)
- [Pretrained Model](#pretrained-model)
- [Tokenization](#tokenization)
- [Embedding](#word-embedding)
- [POS Tagging](#bengali-pos-tagging)
- [Issue](#issue)
- [Contributor Guide](#contributor-guide)
- [Contributor List](#contributor-list)


## Current Features
* [Bengali Tokenization](#tokenization)
  - SentencePiece Tokenizer
  - Basic Tokenizer
  - NLTK Tokenizer
* [Bengali Word Embedding](#word-embedding)
  - Bengali Word2Vec
  - Bengali Fasttext
  - Bengali GloVe
  
* [Bengali POS Tagging](#bengali-pos-tagging)


## Installation

### PIP installer(python 3.5, 3.6, 3.7 tested okay)

  ```pip install bnlp_toolkit```

### Local Installer
  ```
  $git clone https://github.com/sagorbrur/bnlp.git
  $cd bnlp
  $python setup.py install
  ```



## Pretrained Model

### Download Link

* [Bengali SentencePiece](https://github.com/sagorbrur/bnlp/tree/master/model)
* [Bengali Word2Vec](https://drive.google.com/open?id=1DxR8Vw61zRxuUm17jzFnOX97j7QtNW7U)
* [Bengali FastText](https://drive.google.com/open?id=1CFA-SluRyz3s5gmGScsFUcs7AjLfscm2)
* [Bengali GloVe Wordvectors](https://github.com/sagorbrur/GloVe-Bengali)
* [Bengali POS Tag model](https://github.com/sagorbrur/bnlp/blob/master/model/bn_pos_model.pkl)

### Training Details
* Sentencepiece, Word2Vec, Fasttext, GloVe model trained with **Bengali Wikipedia Dump Dataset**
  - [Bengali Wiki Dump](https://dumps.wikimedia.org/bnwiki/latest/)
* SentencePiece Training Vocab Size=50000
* Fasttext trained with total words = 20M, vocab size = 1171011, epoch=50, embedding dimension = 300 and the training loss = 0.318668,
* Word2Vec word embedding dimension = 300
* To Know Bengali GloVe Wordvector and training process follow [this](https://github.com/sagorbrur/GloVe-Bengali) repository
* Bengali CRF POS Tagging was training with [nltr](https://github.com/abhishekgupta92/bangla_pos_tagger/tree/master/data) dataset with 80% accuracy. 


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
    text2id = bsp.text2id(model_path, input_text)
    print(text2id)
    id2text = bsp.id2text(model_path, text2id)
    print(id2text)

    ```
  - Training SentencePiece
    ```py
    from bnlp.sentencepiece_tokenizer import SP_Tokenizer
    
    bsp = SP_Tokenizer()
    data = "test.txt"
    model_prefix = "test"
    vocab_size = 5
    bsp.train_bsp(data, model_prefix, vocab_size) 

    ```

* **Basic Tokenizer**

 

  ```py
  from bnlp.basic_tokenizer import BasicTokenizer
  basic_t = BasicTokenizer()
  raw_text = "আমি বাংলায় গান গাই।"
  tokens = basic_t.tokenize(raw_text)
  print(tokens)
  
  # output: ["আমি", "বাংলায়", "গান", "গাই", "।"]

  ```

* **NLTK Tokenization**

  ```py
  from bnlp.nltk_tokenizer import NLTK_Tokenizer

  text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"
  bnltk = NLTK_Tokenizer()
  word_tokens = bnltk.word_tokenize(text)
  sentence_tokens = bnltk.sentence_tokenize(text)
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
    bwv = Bengali_Word2Vec()
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

      bft = Bengali_Fasttext()
      data = "data.txt"
      model_name = "saved_model.bin"
      epoch = 50
      bft.train_fasttext(data, model_name, epoch)
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

## Bengali POS Tagging
* **Bengali CRF POS Tagging** 


  - Find Pos Tag Using Pretrained Model

    ```py
    from bnlp.bengali_pos import BN_CRF_POS
    bn_pos = BN_CRF_POS()
    model_path = "model/bn_pos_model.pkl"
    text = "আমি ভাত খাই।"
    res = bn_pos.pos_tag(model_path, text)
    print(res)
    # [('আমি', 'PPR'), ('ভাত', 'NC'), ('খাই', 'VM'), ('।', 'PU')]

    ```
  - Train POS Tag Model
  
    ```py
    from bnlp.bengali_pos import BN_CRF_POS
    bn_pos = BN_CRF_POS()
    model_name = "pos_model.pkl"
    tagged_sentences = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা', 'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

    bn_pos.training(model_name, tagged_sentences)

    ```

## Issue
* if `ModuleNotFoundError: No module named 'fasttext'` problem arise please do the next line

```pip install fasttext```
* if `nltk` issue arise please do the following line before importing `bnlp`

```py
import nltk
nltk.download("punkt")
```


## Contributor Guide

Check [CONTRIBUTING.md](https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md) page for details.


## Thanks To

* [Semantics Lab](http://semanticslab.net/)

## Contributor List

* [Sagor Sarker](https://github.com/sagorbrur)
* [Faruk Ahmad](https://github.com/faruk-ahmad)
* [Mehadi Hasan Menon](https://github.com/menon92)
* [Kazal Chandra Barman](https://github.com/kazalbrur)
* [Md Ibrahim](https://github.com/iriad11)
* [Abdullah Al Loman](https://github.com/aaloman)
