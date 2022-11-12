<img align="left" height="70" src="bnlp.svg" alt="bnlp"/>

# Bengali Natural Language Processing(BNLP)

[![Build Status](https://travis-ci.org/sagorbrur/bnlp.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlp)
[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)
[![Downloads](https://pepy.tech/badge/bnlp-toolkit)](https://pepy.tech/project/bnlp-toolkit)
[![Documentation Status](https://readthedocs.org/projects/bnlp/badge/?version=latest)](https://bnlp.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/bnlp_toolkit/community.svg)](https://gitter.im/bnlp_toolkit/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **Embedding Bengali Document**, **Bengali POS Tagging**, **Bengali Name Entity Recognition**, **Construct Neural Model** for Bengali NLP purposes.



## Installation

### PIP installer(Python: 3.6, 3.7, 3.8 tested okay, OS: linux, windows tested okay )

  ```
  pip install bnlp_toolkit
  ```
  **or Upgrade**

  ```
  pip install -U bnlp_toolkit

  ```



## Pretrained Model

### Download Link

Large model published in [huggingface](https://huggingface.co/) model hub.

* [Bengali SentencePiece](https://github.com/sagorbrur/bnlp/tree/master/model)
* [Bengali Word2Vec](https://huggingface.co/sagorsarker/bangla_word2vec)
* [Bengali FastText](https://huggingface.co/sagorsarker/bangla-fasttext)
* [Bengali GloVe Wordvectors](https://huggingface.co/sagorsarker/bangla-glove-vectors)
* [Bengali POS Tag model](https://github.com/sagorbrur/bnlp/blob/master/model/bn_pos.pkl)
* [Bengali NER model](https://github.com/sagorbrur/bnlp/blob/master/model/bn_ner.pkl)
* [Bengali News article Doc2Vec model](https://huggingface.co/sagorsarker/news_article_doc2vec)
* [Bangla Wikipedia Doc2Vec model](https://huggingface.co/sagorsarker/bnwiki_doc2vec_model)

### Training Details
* Sentencepiece, Word2Vec, Fasttext, GloVe model trained with **Bengali Wikipedia Dump Dataset**
  - [Bengali Wiki Dump](https://dumps.wikimedia.org/bnwiki/latest/)
* SentencePiece Training Vocab Size=50000
* Fasttext trained with total words = 20M, vocab size = 1171011, epoch=50, embedding dimension = 300 and the training loss = 0.318668,
* Word2Vec word embedding dimension = 100, min_count=5, window=5, epochs=10
* To Know Bengali GloVe Wordvector and training process follow [this](https://github.com/sagorbrur/GloVe-Bengali) repository
* Bengali CRF POS Tagging was training with [nltr](https://github.com/abhishekgupta92/bangla_pos_tagger/tree/master/data) dataset with 80% accuracy. 
* Bengali CRF NER Tagging was train with [this](https://github.com/MISabic/NER-Bangla-Dataset) data with 90% accuracy.
* Bengali news article doc2vec model train with 8 jsons of [this](https://www.kaggle.com/datasets/ebiswas/bangla-largest-newspaper-dataset) corpus with epochs 40 vector size 100 min_count=2, total news article 400013
* Bengali wikipedia doc2vec model trained with wikipedia dump datasets. Total articles 110448, epochs: 40, vector_size: 100, min_count: 2


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
  
  bnltk = NLTKTokenizer()
  text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"
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
    data = "raw_text.txt"
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
    word = 'গ্রাম'
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
    similar = bwv.most_similar(model_path, word, topn=10)
    print(similar)

    ```
  - Train Bengali Word2Vec with your own data

    Train Bengali word2vec with your custom raw data or tokenized sentences.

    custom tokenized sentence format example:
    ```
    sentences = [['আমি', 'ভাত', 'খাই', '।'], ['সে', 'বাজারে', 'যায়', '।']]
    ```
    Check [gensim word2vec api](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec) for details of training parameter

    ```py
    from bnlp import BengaliWord2Vec
    bwv = BengaliWord2Vec()
    data_file = "raw_text.txt" # or you can pass custom sentence tokens as list of list
    model_name = "test_model.model"
    vector_name = "test_vector.vector"
    bwv.train(data_file, model_name, vector_name, epochs=5)


    ```
  - Pre-train or resume word2vec training with same or new corpus or tokenized sentences

    Check [gensim word2vec api](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec) for details of training parameter

    ```py
    from bnlp import BengaliWord2Vec
    bwv = BengaliWord2Vec()

    trained_model_path = "mytrained_model.model"
    data_file = "raw_text.txt"
    model_name = "test_model.model"
    vector_name = "test_vector.vector"
    bwv.pretrain(trained_model_path, data_file, model_name, vector_name, epochs=5)

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

      Check [fasttext documentation](https://fasttext.cc/docs/en/options.html) for details of training parameter

      ```py
      from bnlp.embedding.fasttext import BengaliFasttext

      bft = BengaliFasttext()
      data = "raw_text.txt"
      model_name = "saved_model.bin"
      epoch = 50
      bft.train(data, model_name, epoch)
      ```

    - Generate Vector File from Fasttext Binary Model
      ```py
      from bnlp.embedding.fasttext import BengaliFasttext

      bft = BengaliFasttext()

      model_path = "mymodel.bin"
      out_vector_name = "myvector.txt"
      bft.bin2vec(model_path, out_vector_name)
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

## Document Embedding
* __Bengali Doc2Vec__
  - Get document vector from input document

    ```py
    from bnlp import BengaliDoc2vec
    
    bn_doc2vec = BengaliDoc2vec()
    
    model_path = "bangla_news_article_doc2vec.model" # keep other .npy model files also in same folder
    document = "রাষ্ট্রবিরোধী ও উসকানিমূলক বক্তব্য দেওয়ার অভিযোগে গাজীপুরের গাছা থানায় ডিজিটাল নিরাপত্তা আইনে করা মামলায় আলোচিত ‘শিশুবক্তা’ রফিকুল ইসলামের বিরুদ্ধে অভিযোগ গঠন করেছেন আদালত। ফলে মামলার আনুষ্ঠানিক বিচার শুরু হলো। আজ বুধবার (২৬ জানুয়ারি) ঢাকার সাইবার ট্রাইব্যুনালের বিচারক আসসামছ জগলুল হোসেন এ অভিযোগ গঠন করেন। এর আগে, রফিকুল ইসলামকে কারাগার থেকে আদালতে হাজির করা হয়। এরপর তাকে নির্দোষ দাবি করে তার আইনজীবী শোহেল মো. ফজলে রাব্বি অব্যাহতি চেয়ে আবেদন করেন। অন্যদিকে, রাষ্ট্রপক্ষ অভিযোগ গঠনের পক্ষে শুনানি করেন। উভয় পক্ষের শুনানি শেষে আদালত অব্যাহতির আবেদন খারিজ করে অভিযোগ গঠনের মাধ্যমে বিচার শুরুর আদেশ দেন। একইসঙ্গে সাক্ষ্যগ্রহণের জন্য আগামী ২২ ফেব্রুয়ারি দিন ধার্য করেন আদালত।"

    vector = bn_doc2vec.get_document_vector(model_path, text)
    print(vector)
    ```

  - Find document similarity between two document

    ```py
    from bnlp import BengaliDoc2vec
    
    bn_doc2vec = BengaliDoc2vec()
    
    model_path = "bangla_news_article_doc2vec.model" # keep other .npy model files also in same folder
    article_1 = "রাষ্ট্রবিরোধী ও উসকানিমূলক বক্তব্য দেওয়ার অভিযোগে গাজীপুরের গাছা থানায় ডিজিটাল নিরাপত্তা আইনে করা মামলায় আলোচিত ‘শিশুবক্তা’ রফিকুল ইসলামের বিরুদ্ধে অভিযোগ গঠন করেছেন আদালত। ফলে মামলার আনুষ্ঠানিক বিচার শুরু হলো। আজ বুধবার (২৬ জানুয়ারি) ঢাকার সাইবার ট্রাইব্যুনালের বিচারক আসসামছ জগলুল হোসেন এ অভিযোগ গঠন করেন। এর আগে, রফিকুল ইসলামকে কারাগার থেকে আদালতে হাজির করা হয়। এরপর তাকে নির্দোষ দাবি করে তার আইনজীবী শোহেল মো. ফজলে রাব্বি অব্যাহতি চেয়ে আবেদন করেন। অন্যদিকে, রাষ্ট্রপক্ষ অভিযোগ গঠনের পক্ষে শুনানি করেন। উভয় পক্ষের শুনানি শেষে আদালত অব্যাহতির আবেদন খারিজ করে অভিযোগ গঠনের মাধ্যমে বিচার শুরুর আদেশ দেন। একইসঙ্গে সাক্ষ্যগ্রহণের জন্য আগামী ২২ ফেব্রুয়ারি দিন ধার্য করেন আদালত।"
    article_2 = "রাষ্ট্রবিরোধী ও উসকানিমূলক বক্তব্য দেওয়ার অভিযোগে গাজীপুরের গাছা থানায় ডিজিটাল নিরাপত্তা আইনে করা মামলায় আলোচিত ‘শিশুবক্তা’ রফিকুল ইসলামের বিরুদ্ধে অভিযোগ গঠন করেছেন আদালত। ফলে মামলার আনুষ্ঠানিক বিচার শুরু হলো। আজ বুধবার (২৬ জানুয়ারি) ঢাকার সাইবার ট্রাইব্যুনালের বিচারক আসসামছ জগলুল হোসেন এ অভিযোগ গঠন করেন। এর আগে, রফিকুল ইসলামকে কারাগার থেকে আদালতে হাজির করা হয়। এরপর তাকে নির্দোষ দাবি করে তার আইনজীবী শোহেল মো. ফজলে রাব্বি অব্যাহতি চেয়ে আবেদন করেন। অন্যদিকে, রাষ্ট্রপক্ষ অভিযোগ গঠনের পক্ষে শুনানি করেন। উভয় পক্ষের শুনানি শেষে আদালত অব্যাহতির আবেদন খারিজ করে অভিযোগ গঠনের মাধ্যমে বিচার শুরুর আদেশ দেন। একইসঙ্গে সাক্ষ্যগ্রহণের জন্য আগামী ২২ ফেব্রুয়ারি দিন ধার্য করেন আদালত।"

    similarity = bn_doc2vec.get_document_similarity(
      model_path,
      article_1,
      article_2
    )
    print(similarity)

    ```

  - Train doc2vec vector with custom text files

    ```py
    from bnlp import BengaliDoc2vec
      
    bn_doc2vec = BengaliDoc2vec()

    text_files = "path/myfiles"
    checkpoint_path = "msc/logs"

    bn_doc2vec.train_doc2vec(
      text_files, 
      checkpoint_path=checkpoint_path,
      vector_size=100,
      min_count=2,
      epochs=10
    )

    # it will train doc2vec with your text files and save the train model in checkpoint_path

    ```

## Bengali POS Tagging
* **Bengali CRF POS Tagging** 


  - Find Pos Tag Using Pretrained Model

    ```py
    from bnlp import POS
    bn_pos = POS()
    model_path = "model/bn_pos.pkl"
    text = "আমি ভাত খাই।" # or you can pass ['আমি', 'ভাত', 'খাই', '।']
    res = bn_pos.tag(model_path, text)
    print(res)
    # [('আমি', 'PPR'), ('ভাত', 'NC'), ('খাই', 'VM'), ('।', 'PU')]

    ```
  - Train POS Tag Model
  
    ```py
    from bnlp import POS
    bn_pos = POS()
    model_name = "pos_model.pkl"
    train_data = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা', 'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

    test_data = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা', 'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

    bn_pos.train(model_name, train_data, test_data)

    ```

## Bengali NER
* **Bengali CRF NER** 


  - Find NER Tag Using Pretrained Model

    ```py
    from bnlp import NER
    bn_ner = NER()
    model_path = "model/bn_ner.pkl"
    text = "সে ঢাকায় থাকে।" # or you can pass ['সে', 'ঢাকায়', 'থাকে', '।']
    result = bn_ner.tag(model_path, text)
    print(result)
    # [('সে', 'O'), ('ঢাকায়', 'S-LOC'), ('থাকে', 'O')]

    ```
  - Train NER Tag Model
  
    ```py
    from bnlp import NER
    bn_ner = NER()
    model_name = "ner_model.pkl"
    train_data = [[('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')]]

    test_data = [[('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')]]

    bn_ner.train(model_name, train_data, test_data)

    ```


## Bengali Corpus Class

* Stopwords and Punctuations
  ```py
  from bnlp.corpus import stopwords, punctuations, letters, digits

  print(stopwords)
  print(punctuations)
  print(letters)
  print(digits)

  ```

* Remove stopwords from Text

    ```py
    from bnlp.corpus import stopwords
    from bnlp.corpus.util import remove_stopwords

    raw_text = 'আমি ভাত খাই।' 
    result = remove_stopwords(raw_text, stopwords)
    print(result)
    # ['ভাত', 'খাই', '।']
    ```


## Contributor Guide

Check [CONTRIBUTING.md](https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md) page for details.


## Thanks To

* [Semantics Lab](https://www.facebook.com/lab.semantics/)
