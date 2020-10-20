Bengali Natural Language Processing(BNLP)
=========================================


.. image:: https://travis-ci.org/sagorbrur/bnlp.svg?branch=master
   :target: https://travis-ci.org/sagorbrur/bnlp
   :alt: Build Status


.. image:: https://img.shields.io/pypi/v/bnlp_toolkit
   :target: https://pypi.org/project/bnlp-toolkit/
   :alt: PyPI version


.. image:: https://img.shields.io/github/v/release/sagorbrur/bnlp
   :target: https://github.com/sagorbrur/bnlp/releases/tag/1.1.0
   :alt: release version


.. image:: https://img.shields.io/badge/python-3.5%7C3.6%7C3.7-brightgreen
   :target: https://pypi.org/project/bnlp-toolkit/
   :alt: Support Python Version


BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**\ , **Embedding Bengali words**\ , **Bengali POS Tagging**\ , **Construct Neural Model** for Bengali NLP purposes.




Installation
============


* 
  pypi package installer(python 3.5, 3.6, 3.7 tested okay)

  ``pip install bnlp_toolkit``

* 
  Local

  .. code-block:: py

     $git clone https://github.com/sagorbrur/bnlp.git
     $cd bnlp
     $python setup.py install

Pretrained Model
================

Download Link
^^^^^^^^^^^^^


* `Bengali SentencePiece <https://github.com/sagorbrur/bnlp/tree/master/model>`_
* `Bengali Word2Vec <https://drive.google.com/open?id=1DxR8Vw61zRxuUm17jzFnOX97j7QtNW7U>`_
* `Bengali FastText <https://drive.google.com/open?id=1CFA-SluRyz3s5gmGScsFUcs7AjLfscm2>`_
* `Bengali GloVe Wordvectors <https://github.com/sagorbrur/GloVe-Bengali>`_
* `Bengali POS Tag model <https://github.com/sagorbrur/bnlp/blob/master/model/bn_pos.pkl>`_
* `Bengali NER model <https://github.com/sagorbrur/bnlp/blob/master/model/bn_ner.pkl>`_

Training Details
^^^^^^^^^^^^^^^^


* Sentencepiece, Word2Vec, Fasttext, GloVe model trained with **Bengali Wikipedia Dump Dataset**

  * `Bengali Wiki Dump <https://dumps.wikimedia.org/bnwiki/latest/>`_

* SentencePiece Training Vocab Size=50000
* Fasttext trained with total words = 20M, vocab size = 1171011, epoch=50, embedding dimension = 300 and the training loss = 0.318668,
* Word2Vec word embedding dimension = 300
* To Know Bengali GloVe Wordvector and training process follow `this <https://github.com/sagorbrur/GloVe-Bengali>`_ repository
* Bengali CRF POS Tagging was training with `nltr <https://github.com/abhishekgupta92/bangla_pos_tagger/tree/master/data>`_ dataset with 80% accuracy. 
* Bengali CRF NER Tagging was train with `this <https://github.com/MISabic/NER-Bangla-Dataset>`_ data with 90% accuracy.


Tokenization
============


* 
  **Basic Tokenizer**

.. code-block:: py

     from bnlp import BasicTokenizer
     basic_t = BasicTokenizer()
     raw_text = "আমি বাংলায় গান গাই।"
     tokens = basic_t.tokenize(raw_text)
     print(tokens)

     # output: ["আমি", "বাংলায়", "গান", "গাই", "।"]


* 
  **NLTK Tokenization**

  .. code-block:: py

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


* 
  **Bengali SentencePiece Tokenization**


  * 
    tokenization using trained model

    .. code-block:: py

       from bnlp import SentencepieceTokenizer

       bsp = SentencepieceTokenizer()
       model_path = "./model/bn_spm.model"
       input_text = "আমি ভাত খাই। সে বাজারে যায়।"
       tokens = bsp.tokenize(model_path, input_text)
       print(tokens)

  * 
    Training SentencePiece

    .. code-block:: py

       from bnlp import SentencepieceTokenizer

       bsp = SentencepieceTokenizer()
       data = "sample.txt"
       model_prefix = "test"
       vocab_size = 5
       bsp.train(data, model_prefix, vocab_size)


Word Embedding
==============


* 
  **Bengali Word2Vec**


  * 
    Generate Vector using pretrain model

    .. code-block:: py

       from bnlp import BengaliWord2Vec

       bwv = BengaliWord2Vec()
       model_path = "model/bengali_word2vec.model"
       word = 'আমার'
       vector = bwv.generate_word_vector(model_path, word)
       print(vector.shape)
       print(vector)

  * 
    Find Most Similar Word Using Pretrained Model

    .. code-block:: py

       from bnlp import BengaliWord2Vec

       bwv = BengaliWord2Vec()
       model_path = "model/bengali_word2vec.model"
       word = 'আমার'
       similar = bwv.most_similar(model_path, word)
       print(similar)

  * 
    Train Bengali Word2Vec with your own data

    .. code-block:: py

       from bnlp import BengaliWord2Vec
       bwv = BengaliWord2Vec()
       data_file = "test.txt"
       model_name = "test_model.model"
       vector_name = "test_vector.vector"
       bwv.train(data_file, model_name, vector_name)




* 
  **Bengali FastText**
   Install fasttext first by pip install fasttext

   - Generate Vector Using Pretrained Model


     .. code-block:: py

        from bnlp import BengaliFasttext
   
        bft = BengaliFasttext()
        word = "গ্রাম"
        model_path = "model/bengali_fasttext.bin"
        word_vector = bft.generate_word_vector(model_path, word)
        print(word_vector.shape)
        print(word_vector)


   - Train Bengali FastText Model

     .. code-block:: py

        from bnlp import BengaliFasttext
   
        bft = BengaliFasttext()
        data = "data.txt"
        model_name = "saved_model.bin"
        epoch = 50
        bft.train(data, model_name, epoch)



* 
  **Bengali GloVe Word Vectors**

  We trained glove model with bengali data(wiki+news articles) and published bengali glove word vectors</br>
  You can download and use it on your different machine learning purposes.

  .. code-block:: py

     from bnlp import BengaliGlove

     bng = BengaliGlove()
     glove_path = "bn_glove.39M.100d.txt"
     word = "গ্রাম"
     res = bng.closest_word(glove_path, word)
     print(res)
     vec = bng.word2vec(glove_path, word)
     print(vec)

Bengali POS Tagging
===================


* **Bengali CRF POS Tagging** 


* 
  Find Pos Tag Using Pretrained Model

  .. code-block:: py

     from bnlp import POS
     bn_pos = POS()
     model_path = "model/bn_pos_model.pkl"
     text = "আমি ভাত খাই।"
     res = bn_pos.tag(model_path, text)
     print(res)
     # [('আমি', 'PPR'), ('ভাত', 'NC'), ('খাই', 'VM'), ('।', 'PU')]

* 
  Train POS Tag Model

  .. code-block:: py

     from bnlp import POS
     bn_pos = POS()
     model_name = "pos_model.pkl"
     tagged_sentences = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা', 'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

     bn_pos.train(model_name, tagged_sentences)


Bengali NER
===========


* **Bengali CRF NER** 


* 
  Find NER Tag Using Pretrained Model

  .. code-block:: py

     from bnlp import ner
     bn_ner = NER()
     model_path = "model/bn_pos_model.pkl"
     text = "সে ঢাকায় থাকে।"
     res = bn_ner.tag(model_path, text)
     print(res)
     # [('সে', 'O'), ('ঢাকায়', 'S-LOC'), ('থাকে', 'O')]

* 
  Train NER Model

  .. code-block:: py

     from bnlp import NER
     bn_ner = NER()
     model_name = "ner_model.pkl"
     tagged_sentences = [[('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')]]

     bn_ner.train(model_name, tagged_sentences)


Contributor Guide
=================

Check `CONTRIBUTING.md <https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md>`_ page for details.

