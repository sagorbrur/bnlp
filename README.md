<img align="left" height="70" src="bnlp.svg" alt="bnlp"/>

# Bengali Natural Language Processing(BNLP)

[![Build Status](https://travis-ci.org/sagorbrur/bnlp.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlp)
[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![release version](https://img.shields.io/github/v/release/sagorbrur/bnlp)](https://github.com/sagorbrur/bnlp/releases/tag/2.0.0)
[![Support Python Version](https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-brightgreen)](https://pypi.org/project/bnlp-toolkit/)
[![Documentation Status](https://readthedocs.org/projects/bnlp/badge/?version=latest)](https://bnlp.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/bnlp_toolkit/community.svg)](https://gitter.im/bnlp_toolkit/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **Bengali POS Tagging**, **Bengali Name Entity Recognition**, **Construct Neural Model** for Bengali NLP purposes.

**NB: Any Researcher who refer this tool in his/her paper please let us know, we will include paper link here**</br>


## Installation

### PIP installer(python 3.5, 3.6, 3.7 tested okay)

  ```
  pip install bnlp_toolkit
  ```

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
* [Bengali POS Tag model](https://github.com/sagorbrur/bnlp/blob/master/model/bn_pos.pkl)
* [Bengali NER model](https://github.com/sagorbrur/bnlp/blob/master/model/bn_ner.pkl)

### Training Details
* Sentencepiece, Word2Vec, Fasttext, GloVe model trained with **Bengali Wikipedia Dump Dataset**
  - [Bengali Wiki Dump](https://dumps.wikimedia.org/bnwiki/latest/)
* SentencePiece Training Vocab Size=50000
* Fasttext trained with total words = 20M, vocab size = 1171011, epoch=50, embedding dimension = 300 and the training loss = 0.318668,
* Word2Vec word embedding dimension = 300
* To Know Bengali GloVe Wordvector and training process follow [this](https://github.com/sagorbrur/GloVe-Bengali) repository
* Bengali CRF POS Tagging was training with [nltr](https://github.com/abhishekgupta92/bangla_pos_tagger/tree/master/data) dataset with 80% accuracy. 
* Bengali CRF NER Tagging was train with [this](https://github.com/MISabic/NER-Bangla-Dataset) data with 90% accuracy.


## Tokenization

* **Basic Tokenizer**

 

  ```py
  from bnlp import BasicTokenizer
  basic_tokenizer = BasicTokenizer()
  raw_text = "আমি বাংলায় গান গাই।"
  tokens = basic_tokenizer.tokenize(raw_text)
  print(tokens)
  
  # output: ["আমি", "বাংলায়", "গান", "গাই", "।"]

  ```

* **NLTK Tokenization**

  ```py
  from bnlp import NLTKTokenizer

  text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"
  bnltk = NLTKTokenizer()
  word_tokens = bnltk.word_tokenize(text)
  sentence_tokens = bnltk.sentence_tokenize(text)
  print(word_tokens)
  print(sentence_tokens)
  
  # output
  # word_token: ["আমি", "ভাত", "খাই", "।", "সে", "বাজারে", "যায়", "।", "তিনি", "কি", "সত্যিই", "ভালো", "মানুষ", "?"]
  # sentence_token: ["আমি ভাত খাই।", "সে বাজারে যায়।", "তিনি কি সত্যিই ভালো মানুষ?"]

  ```


* **Bengali SentencePiece Tokenization**

  - tokenization using trained model
    ```py
    from bnlp import SentencepieceTokenizer

    bsp = SentencepieceTokenizer()
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
    from bnlp import SentencepieceTokenizer
    
    bsp = SentencepieceTokenizer()
    data = "test.txt"
    model_prefix = "test"
    vocab_size = 5
    bsp.train(data, model_prefix, vocab_size) 

    ```



## Word Embedding

* **Bengali Word2Vec**

  - Generate Vector using pretrain model

    ```py
    from bnlp import BengaliWord2Vec

    bwv = BengaliWord2Vec()
    model_path = "bengali_word2vec.model"
    word = 'আমার'
    vector = bwv.generate_word_vector(model_path, word)
    print(vector.shape)
    print(vector)

    ```

  - Find Most Similar Word Using Pretrained Model

    ```py
    from bnlp import BengaliWord2Vec

    bwv = BengaliWord2Vec()
    model_path = "bengali_word2vec.model"
    word = 'গ্রাম'
    similar = bwv.most_similar(model_path, word)
    print(similar)

    ```
  - Train Bengali Word2Vec with your own data

    ```py
    from bnlp import BengaliWord2Vec
    bwv = BengaliWord2Vec()
    data_file = "sample.txt"
    model_name = "test_model.model"
    vector_name = "test_vector.vector"
    bwv.train(data_file, model_name, vector_name)


    ```
    
 * **Bengali FastText**
 
    To use `fasttext` you need to install fasttext manually by `pip install fasttext==0.9.2`
    
    NB: `fasttext` may not be worked in `windows`, it will only work in `linux`

    - Generate Vector Using Pretrained Model
      

      ```py
      from bnlp.embedding.fasttext import BengaliFasttext

      bft = BengaliFasttext()
      word = "গ্রাম"
      model_path = "bengali_fasttext_wiki.bin"
      word_vector = bft.generate_word_vector(model_path, word)
      print(word_vector.shape)
      print(word_vector)


      ```
    - Train Bengali FastText Model

      ```py
      from bnlp.embedding.fasttext import BengaliFasttext

      bft = BengaliFasttext()
      data = "sample.txt"
      model_name = "saved_model.bin"
      epoch = 50
      bft.train(data, model_name, epoch)
      ```

* **Bengali GloVe Word Vectors**

  We trained glove model with bengali data(wiki+news articles) and published bengali glove word vectors</br>
  You can download and use it on your different machine learning purposes.

  ```py
  from bnlp import BengaliGlove
  glove_path = "bn_glove.39M.100d.txt"
  word = "গ্রাম"
  bng = BengaliGlove()
  res = bng.closest_word(glove_path, word)
  print(res)
  vec = bng.word2vec(glove_path, word)
  print(vec)

  ```

## Bengali POS Tagging
* **Bengali CRF POS Tagging** 


  - Find Pos Tag Using Pretrained Model

    ```py
    from bnlp import POS
    bn_pos = POS()
    model_path = "model/bn_pos.pkl"
    text = "আমি ভাত খাই।"
    res = bn_pos.tag(model_path, text)
    print(res)
    # [('আমি', 'PPR'), ('ভাত', 'NC'), ('খাই', 'VM'), ('।', 'PU')]

    ```
  - Train POS Tag Model
  
    ```py
    from bnlp import POS
    bn_pos = POS()
    model_name = "pos_model.pkl"
    tagged_sentences = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা', 'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

    bn_pos.train(model_name, tagged_sentences)

    ```

## Bengali NER
* **Bengali CRF NER** 


  - Find NER Tag Using Pretrained Model

    ```py
    from bnlp import NER
    bn_ner = NER()
    model_path = "model/bn_ner.pkl"
    text = "সে ঢাকায় থাকে।"
    result = bn_ner.tag(model_path, text)
    print(result)
    # [('সে', 'O'), ('ঢাকায়', 'S-LOC'), ('থাকে', 'O')]

    ```
  - Train NER Tag Model
  
    ```py
    from bnlp import NER
    bn_ner = NER()
    model_name = "ner_model.pkl"
    tagged_sentences = [[('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')]]

    bn_ner.train(model_name, tagged_sentences)

    ```

## Bengali Corpus Class

* Stopwords and Punctuations
  ```py
  from bnlp.corpus import stopwords, punctuations

  stopwords = stopwords() 
  print(stopwords)
  print(punctuations)

  ```

* Remove stopwords from Text

    ```py
    from bnlp.corpus import stopwords
    from bnlp.corpus.util import remove_stopwords

    stopwords = stopwords()
    raw_text = 'আমি ভাত খাই।' 
    result = remove_stopwords(raw_text, stopwords)
    print(result)
    # ['ভাত', 'খাই', '।']
    ```


## Contributor Guide

Check [CONTRIBUTING.md](https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md) page for details.


## Thanks To

* [Semantics Lab](http://semanticslab.net/)

## Contributor List

[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/0)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/0)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/1)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/1)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/2)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/2)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/3)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/3)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/4)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/4)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/5)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/5)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/6)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/6)[![](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/images/7)](https://sourcerer.io/fame/sagorbrur/sagorbrur/bnlp/links/7)

### Extra Contributor
* [Mehadi Hasan Menon](https://github.com/menon92)
* [Kazal Chandra Barman](https://github.com/kazalbrur)
