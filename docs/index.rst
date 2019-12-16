.. role:: raw-html-m2r(raw)
   :format: html


:raw-html-m2r:`<img align="left" height="70" src="../bnlp.svg" alt="bnlp"/>`

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


.. image:: https://img.shields.io/pypi/dw/bnlp_toolkit?color=green
   :target: https://pypi.org/project/bnlp-toolkit/
   :alt: pypi Downloads


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
* `Bengali POS Tag model <https://github.com/sagorbrur/bnlp/blob/master/model/bn_pos_model.pkl>`_

Training Details
^^^^^^^^^^^^^^^^


* Sentencepiece, Word2Vec, Fasttext, GloVe model trained with **Bengali Wikipedia Dump Dataset**

  * `Bengali Wiki Dump <https://dumps.wikimedia.org/bnwiki/latest/>`_

* SentencePiece Training Vocab Size=50000
* Fasttext trained with total words = 20M, vocab size = 1171011, epoch=50, embedding dimension = 300 and the training loss = 0.318668,
* Word2Vec word embedding dimension = 300
* To Know Bengali GloVe Wordvector and training process follow `this <https://github.com/sagorbrur/GloVe-Bengali>`_ repository
* Bengali CRF POS Tagging was training with `nltr <https://github.com/abhishekgupta92/bangla_pos_tagger/tree/master/data>`_ dataset with 80% accuracy. 

Tokenization
============


* 
  **Bengali SentencePiece Tokenization**


  * 
    tokenization using trained model

    .. code-block:: py

       from bnlp.sentencepiece_tokenizer import SP_Tokenizer

       bsp = SP_Tokenizer()
       model_path = "./model/bn_spm.model"
       input_text = "আমি ভাত খাই। সে বাজারে যায়।"
       tokens = bsp.tokenize(model_path, input_text)
       print(tokens)

  * 
    Training SentencePiece

    .. code-block:: py

       from bnlp.sentencepiece_tokenizer import SP_Tokenizer

       bsp = SP_Tokenizer(is_train=True)
       data = "test.txt"
       model_prefix = "test"
       vocab_size = 5
       bsp.train_bsp(data, model_prefix, vocab_size)

* 
  **Basic Tokenizer**

.. code-block:: py

     from bnlp.basic_tokenizer import BasicTokenizer
     basic_t = BasicTokenizer(False)
     raw_text = "আমি বাংলায় গান গাই।"
     tokens = basic_t.tokenize(raw_text)
     print(tokens)

     # output: ["আমি", "বাংলায়", "গান", "গাই", "।"]


* 
  **NLTK Tokenization**

  .. code-block:: py

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

Word Embedding
--------------


* 
  **Bengali Word2Vec**


  * 
    Generate Vector using pretrain model

    .. code-block:: py

       from bnlp.bengali_word2vec import Bengali_Word2Vec

       bwv = Bengali_Word2Vec()
       model_path = "model/bengali_word2vec.model"
       word = 'আমার'
       vector = bwv.generate_word_vector(model_path, word)
       print(vector.shape)
       print(vector)

  * 
    Find Most Similar Word Using Pretrained Model

    .. code-block:: py

       from bnlp.bengali_word2vec import Bengali_Word2Vec

       bwv = Bengali_Word2Vec()
       model_path = "model/bengali_word2vec.model"
       word = 'আমার'
       similar = bwv.most_similar(model_path, word)
       print(similar)

  * 
    Train Bengali Word2Vec with your own data

    .. code-block:: py

       from bnlp.bengali_word2vec import Bengali_Word2Vec
       bwv = Bengali_Word2Vec(is_train=True)
       data_file = "test.txt"
       model_name = "test_model.model"
       vector_name = "test_vector.vector"
       bwv.train_word2vec(data_file, model_name, vector_name)




* 
  **Bengali FastText**

   - Generate Vector Using Pretrained Model


     .. code-block:: py

        from bnlp.bengali_fasttext import Bengali_Fasttext
   
        bft = Bengali_Fasttext()
        word = "গ্রাম"
        model_path = "model/bengali_fasttext.bin"
        word_vector = bft.generate_word_vector(model_path, word)
        print(word_vector.shape)
        print(word_vector)


   - Train Bengali FastText Model

     .. code-block:: py

        from bnlp.bengali_fasttext import Bengali_Fasttext
   
        bft = Bengali_Fasttext(is_train=True)
        data = "data.txt"
        model_name = "saved_model.bin"
        epoch = 50
        bft.train_fasttext(data, model_name, epoch)



* 
  **Bengali GloVe Word Vectors**

  We trained glove model with bengali data(wiki+news articles) and published bengali glove word vectors</br>
  You can download and use it on your different machine learning purposes.

  .. code-block:: py

     from bnlp.glove_wordvector import BN_Glove
     glove_path = "bn_glove.39M.100d.txt"
     word = "গ্রাম"
     bng = BN_Glove()
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

     from bnlp.bengali_pos import BN_CRF_POS
     bn_pos = BN_CRF_POS()
     model_path = "model/bn_pos_model.pkl"
     text = "আমি ভাত খাই।"
     res = bn_pos.pos_tag(model_path, text)
     print(res)
     # [('আমি', 'PPR'), ('ভাত', 'NC'), ('খাই', 'VM'), ('।', 'PU')]

* 
  Train POS Tag Model

  .. code-block:: py

     from bnlp.bengali_pos import BN_CRF_POS
     bn_pos = BN_CRF_POS()
     model_name = "pos_model.pkl"
     tagged_sentences = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা', 'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

     bn_pos.training(model_name, tagged_sentences)

Issue
=====


* if ``ModuleNotFoundError: No module named 'fasttext'`` problem arise please do the next line

``pip install fasttext``


* if ``nltk`` issue arise please do the following line before importing ``bnlp``

.. code-block:: py

   import nltk
   nltk.download("punkt")

Contributor Guide
=================

Check `CONTRIBUTING.md <https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md>`_ page for details.

