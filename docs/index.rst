Bengali Natural Language Processing(BNLP)
=========================================

[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)

[![Downloads](https://pepy.tech/badge/bnlp-toolkit)](https://pepy.tech/project/bnlp-toolkit)

[![Documentation Status](https://readthedocs.org/projects/bnlp/badge/?version=latest)](https://bnlp.readthedocs.io/en/latest/?badge=latest)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **Embedding Bengali Document**, **Bengali POS Tagging**, **Bengali Name Entity Recognition**, **Construct Neural Model** for Bengali NLP purposes.

Table of contents
=================

<!--ts-->

   * `Installation <#installation>`_

	  \* [PIP installer](#pip\-installer)

   * `Pretrained Model <#pretrained-model>`_

	  \* [Download Links](#download\-links)

	  \* [Training Details](#training\-details)

   * `Tokenization <#tokenization>`_

	  \* [Basic Tokenizer](#basic\-tokenizer)

	  \* [NLTK Tokenization](#nltk\-tokenization)

	  \* [Bengali SentencePiece Tokenization](#bengali\-sentencepiece\-tokenization)

		 \* [Tokenization using trained model](#tokenization\-using\-trained\-model)

		 \* [Training SentencePiece](#training\-sentencepiece)

   * `Word Embedding <#word-embedding>`_

	  \* [Bengali Word2Vec](#bengali\-word2vec)

		 \* [Generate Vector using pretrain model](#generate\-vector\-using\-pretrain\-model)

		 \* [Find Most Similar Word Using Pretrained Model](#find\-most\-similar\-word\-using\-pretrained\-model)

		 \* [Train Bengali Word2Vec with your own data](#train\-bengali\-word2vec\-with\-your\-own\-data)

		 \* [Pre\-train or resume word2vec training with same or new corpus or tokenized sentences](#pre\-train\-or\-resume\-word2vec\-training\-with\-same\-or\-new\-corpus\-or\-tokenized\-sentences)

	 \* [Bengali FastText](#bengali\-fasttext)

		\* [Generate Vector Using Pretrained Model](#generate\-vector\-using\-pretrain\-model)

		\* [Train Bengali FastText Model](#train\-bengali\-fasttext\-model)

		\* [Generate Vector File from Fasttext Binary Model](#generate\-vector\-file\-from\-fasttext\-binary\-model)

	 \* [Bengali GloVe Word Vectors](#bengali\-glove\-word\-vectors)

   * `Document Embedding <#document-embedding>`_

	  \* [Bengali Doc2Vec](#bengali\-doc2vec)

		 \* [Get document vector from input document](#get\-document\-vector\-from\-input\-document)

		 \* [Find document similarity between two document](#find\-document\-similarity\-between\-two\-document)

		 \* [Train doc2vec vector with custom text files](#train\-doc2vec\-vector\-with\-custom\-text\-files)

   * `Bengali POS Tagging <#bengali-pos-tagging>`_

	  \* [Bengali CRF POS Tagging](#bengali\-crf\-pos\-tagging)

		 \* [Find Pos Tag Using Pretrained Model](#find\-pos\-tag\-using\-pretrained\-model)

		 \* [Train POS Tag Model](#train\-pos\-tag\-model)

   * `Bengali NER <#bengali-ner>`_

	  \* [Bengali CRF NER](#bengali\-crf\-ner)

		 \* [Find NER Tag Using Pretrained Model](#find\-ner\-tag\-using\-pretrained\-model)

		 \* [Train NER Tag Model](#train\-ner\-tag\-model)

   * `Bengali Corpus Class <#bengali-corpus-class>`_

	  \* [Stopwords and Punctuations](#stopwords\-and\-punctuations)

	  \* [Remove stopwords from Text](#remove\-stopwords\-from\-text)

   * `Bangla Text Cleaning <#text-cleaning>`_

   * `Contributor Guide <#contributor-guide>`_
<!--te-->
---

# Installation
==============

## PIP installer
================

  ```

  pip install bnlp_toolkit

  ```

  **or Upgrade**

  ```

  pip install -U bnlp_toolkit

  ```

  - Python: 3.6, 3.7, 3.8, 3.9

  - OS: Linux, Windows, Mac



# Pretrained Model
==================

## Download Links
=================

Large model published in `huggingface <https://huggingface.co/>`_ model hub.

* `Bengali SentencePiece <https://github.com/sagorbrur/bnlp/tree/master/model>`_

* `Bengali Word2Vec <https://huggingface.co/sagorsarker/bangla_word2vec>`_

* `Bengali FastText <https://huggingface.co/sagorsarker/bangla-fasttext>`_

* `Bengali GloVe Wordvectors <https://huggingface.co/sagorsarker/bangla-glove-vectors>`_

* `Bengali POS Tag model <https://github.com/sagorbrur/bnlp/blob/master/model/bn_pos.pkl>`_

* `Bengali NER model <https://github.com/sagorbrur/bnlp/blob/master/model/bn_ner.pkl>`_

* `Bengali News article Doc2Vec model <https://huggingface.co/sagorsarker/news_article_doc2vec>`_

* `Bangla Wikipedia Doc2Vec model <https://huggingface.co/sagorsarker/bnwiki_doc2vec_model>`_

## Training Details
===================

* Sentencepiece, Word2Vec, Fasttext, GloVe model trained with **Bengali Wikipedia Dump Dataset**

  - `Bengali Wiki Dump <https://dumps.wikimedia.org/bnwiki/latest/>`_
* SentencePiece Training Vocab Size=50000

* Fasttext trained with total words = 20M, vocab size = 1171011, epoch=50, embedding dimension = 300 and the training loss = 0.318668,

* Word2Vec word embedding dimension = 100, min_count=5, window=5, epochs=10

* To Know Bengali GloVe Wordvector and training process follow `this <https://github.com/sagorbrur/GloVe-Bengali>`_ repository

* Bengali CRF POS Tagging was training with `nltr <https://github.com/abhishekgupta92/bangla_pos_tagger/tree/master/data>`_ dataset with 80% accuracy.

* Bengali CRF NER Tagging was train with `this <https://github.com/MISabic/NER-Bangla-Dataset>`_ data with 90% accuracy.

* Bengali news article doc2vec model train with 8 jsons of `this <https://www.kaggle.com/datasets/ebiswas/bangla-largest-newspaper-dataset>`_ corpus with epochs 40 vector size 100 min_count=2, total news article 400013

* Bengali wikipedia doc2vec model trained with wikipedia dump datasets. Total articles 110448, epochs: 40, vector*size: 100, min*count: 2


# Tokenization
==============

## Basic Tokenizer
==================

  ```py

  from bnlp import BasicTokenizer
  
  basic_tokenizer = BasicTokenizer()

  raw_text = "আমি বাংলায় গান গাই।"

  tokens = basic*tokenizer.tokenize(raw*text)

  print(tokens)

  # output: ["আমি", "বাংলায়", "গান", "গাই", "।"]

  ```

## NLTK Tokenization
====================

  ```py

  from bnlp import NLTKTokenizer

  bnltk = NLTKTokenizer()

  text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"

  word*tokens = bnltk.word*tokenize(text)

  sentence*tokens = bnltk.sentence*tokenize(text)

  print(word_tokens)

  print(sentence_tokens)

  # output

  # word_token: ["আমি", "ভাত", "খাই", "।", "সে", "বাজারে", "যায়", "।", "তিনি", "কি", "সত্যিই", "ভালো", "মানুষ", "?"]

  # sentence_token: ["আমি ভাত খাই।", "সে বাজারে যায়।", "তিনি কি সত্যিই ভালো মানুষ?"]

  ```


## Bengali SentencePiece Tokenization
=====================================

### Tokenization using trained model
====================================

```py

from bnlp import SentencepieceTokenizer

bsp = SentencepieceTokenizer()

model*path = "./model/bn*spm.model"

input_text = "আমি ভাত খাই। সে বাজারে যায়।"

tokens = bsp.tokenize(model*path, input*text)

print(tokens)

text2id = bsp.text2id(model*path, input*text)

print(text2id)

id2text = bsp.id2text(model_path, text2id)

print(id2text)

```

### Training SentencePiece
==========================

```py

from bnlp import SentencepieceTokenizer

bsp = SentencepieceTokenizer()

data = "raw_text.txt"

model_prefix = "test"

vocab_size = 5

bsp.train(data, model*prefix, vocab*size)

```

# Word Embedding
================

## Bengali Word2Vec
===================

### Generate Vector using pretrain model
========================================

```py

from bnlp import BengaliWord2Vec

bwv = BengaliWord2Vec()

model*path = "bengali*word2vec.model"

word = 'গ্রাম'

vector = bwv.generate*word*vector(model_path, word)

print(vector.shape)

print(vector)

```

### Find Most Similar Word Using Pretrained Model
=================================================

```py

from bnlp import BengaliWord2Vec

bwv = BengaliWord2Vec()

model*path = "bengali*word2vec.model"

word = 'গ্রাম'

similar = bwv.most*similar(model*path, word, topn=10)

print(similar)

```
### Train Bengali Word2Vec with your own data
=============================================

Train Bengali word2vec with your custom raw data or tokenized sentences.

Custom tokenized sentence format example:

```py

sentences = [['আমি', 'ভাত', 'খাই', '।'], ['সে', 'বাজারে', 'যায়', '।']]

```

Check `gensim word2vec api <https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`_ for details of training parameter

```py

from bnlp import BengaliWord2Vec

bwv = BengaliWord2Vec()

data*file = "raw*text.txt" # or you can pass custom sentence tokens as list of list

model*name = "test*model.model"

vector*name = "test*vector.vector"

bwv.train(data*file, model*name, vector_name, epochs=5)

```

### Pre-train or resume word2vec training with same or new corpus or tokenized sentences
========================================================================================

Check `gensim word2vec api <https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`_ for details of training parameter

```py

from bnlp import BengaliWord2Vec

bwv = BengaliWord2Vec()

trained*model*path = "mytrained_model.model"

data*file = "raw*text.txt"

model*name = "test*model.model"

vector*name = "test*vector.vector"

bwv.pretrain(trained*model*path, data*file, model*name, vector_name, epochs=5)

```

## Bengali FastText
===================

To use `fasttext` you need to install fasttext manually by `pip install fasttext==0.9.2`

NB: `fasttext` may not be worked in `windows`, it will only work in `linux`

## Generate Vector Using Pretrained Model
=========================================

  ```py

  from bnlp.embedding.fasttext import BengaliFasttext

  bft = BengaliFasttext()

  word = "গ্রাম"

  model*path = "bengali*fasttext_wiki.bin"

  word*vector = bft.generate*word*vector(model*path, word)

  print(word_vector.shape)

  print(word_vector)

  ```

## Train Bengali FastText Model
===============================

Check `fasttext documentation <https://fasttext.cc/docs/en/options.html>`_ for details of training parameter

  ```py

  from bnlp.embedding.fasttext import BengaliFasttext

  bft = BengaliFasttext()

  data = "raw_text.txt"

  model*name = "saved*model.bin"

  epoch = 50

  bft.train(data, model_name, epoch)

  ```

## Generate Vector File from Fasttext Binary Model
==================================================

```py

from bnlp.embedding.fasttext import BengaliFasttext

bft = BengaliFasttext()

model_path = "mymodel.bin"

out*vector*name = "myvector.txt"

bft.bin2vec(model*path, out*vector_name)

```

# Bengali GloVe Word Vectors
============================

We trained glove model with bengali data(wiki+news articles) and published bengali glove word vectors</br>

You can download and use it on your different machine learning purposes.

```py

from bnlp import BengaliGlove

glove*path = "bn*glove.39M.100d.txt"

word = "গ্রাম"

bng = BengaliGlove()

res = bng.closest*word(glove*path, word)

print(res)

vec = bng.word2vec(glove_path, word)

print(vec)

```

# Document Embedding
====================

## Bengali Doc2Vec
==================
### Get document vector from input document
===========================================

```py

from bnlp import BengaliDoc2vec

bn_doc2vec = BengaliDoc2vec()

model*path = "bangla*news*article*doc2vec.model" # keep other .npy model files also in same folder

document = "রাষ্ট্রবিরোধী ও উসকানিমূলক বক্তব্য দেওয়ার অভিযোগে গাজীপুরের গাছা থানায় ডিজিটাল নিরাপত্তা আইনে করা মামলায় আলোচিত ‘শিশুবক্তা’ রফিকুল ইসলামের বিরুদ্ধে অভিযোগ গঠন করেছেন আদালত। ফলে মামলার আনুষ্ঠানিক বিচার শুরু হলো। আজ বুধবার (২৬ জানুয়ারি) ঢাকার সাইবার ট্রাইব্যুনালের বিচারক আসসামছ জগলুল হোসেন এ অভিযোগ গঠন করেন। এর আগে, রফিকুল ইসলামকে কারাগার থেকে আদালতে হাজির করা হয়। এরপর তাকে নির্দোষ দাবি করে তার আইনজীবী শোহেল মো. ফজলে রাব্বি অব্যাহতি চেয়ে আবেদন করেন। অন্যদিকে, রাষ্ট্রপক্ষ অভিযোগ গঠনের পক্ষে শুনানি করেন। উভয় পক্ষের শুনানি শেষে আদালত অব্যাহতির আবেদন খারিজ করে অভিযোগ গঠনের মাধ্যমে বিচার শুরুর আদেশ দেন। একইসঙ্গে সাক্ষ্যগ্রহণের জন্য আগামী ২২ ফেব্রুয়ারি দিন ধার্য করেন আদালত।"

vector = bn*doc2vec.get*document*vector(model*path, text)

print(vector)

```

### Find document similarity between two document
=================================================

```py

from bnlp import BengaliDoc2vec

bn_doc2vec = BengaliDoc2vec()

model*path = "bangla*news*article*doc2vec.model" # keep other .npy model files also in same folder

article_1 = "রাষ্ট্রবিরোধী ও উসকানিমূলক বক্তব্য দেওয়ার অভিযোগে গাজীপুরের গাছা থানায় ডিজিটাল নিরাপত্তা আইনে করা মামলায় আলোচিত ‘শিশুবক্তা’ রফিকুল ইসলামের বিরুদ্ধে অভিযোগ গঠন করেছেন আদালত। ফলে মামলার আনুষ্ঠানিক বিচার শুরু হলো। আজ বুধবার (২৬ জানুয়ারি) ঢাকার সাইবার ট্রাইব্যুনালের বিচারক আসসামছ জগলুল হোসেন এ অভিযোগ গঠন করেন। এর আগে, রফিকুল ইসলামকে কারাগার থেকে আদালতে হাজির করা হয়। এরপর তাকে নির্দোষ দাবি করে তার আইনজীবী শোহেল মো. ফজলে রাব্বি অব্যাহতি চেয়ে আবেদন করেন। অন্যদিকে, রাষ্ট্রপক্ষ অভিযোগ গঠনের পক্ষে শুনানি করেন। উভয় পক্ষের শুনানি শেষে আদালত অব্যাহতির আবেদন খারিজ করে অভিযোগ গঠনের মাধ্যমে বিচার শুরুর আদেশ দেন। একইসঙ্গে সাক্ষ্যগ্রহণের জন্য আগামী ২২ ফেব্রুয়ারি দিন ধার্য করেন আদালত।"

article_2 = "রাষ্ট্রবিরোধী ও উসকানিমূলক বক্তব্য দেওয়ার অভিযোগে গাজীপুরের গাছা থানায় ডিজিটাল নিরাপত্তা আইনে করা মামলায় আলোচিত ‘শিশুবক্তা’ রফিকুল ইসলামের বিরুদ্ধে অভিযোগ গঠন করেছেন আদালত। ফলে মামলার আনুষ্ঠানিক বিচার শুরু হলো। আজ বুধবার (২৬ জানুয়ারি) ঢাকার সাইবার ট্রাইব্যুনালের বিচারক আসসামছ জগলুল হোসেন এ অভিযোগ গঠন করেন। এর আগে, রফিকুল ইসলামকে কারাগার থেকে আদালতে হাজির করা হয়। এরপর তাকে নির্দোষ দাবি করে তার আইনজীবী শোহেল মো. ফজলে রাব্বি অব্যাহতি চেয়ে আবেদন করেন। অন্যদিকে, রাষ্ট্রপক্ষ অভিযোগ গঠনের পক্ষে শুনানি করেন। উভয় পক্ষের শুনানি শেষে আদালত অব্যাহতির আবেদন খারিজ করে অভিযোগ গঠনের মাধ্যমে বিচার শুরুর আদেশ দেন। একইসঙ্গে সাক্ষ্যগ্রহণের জন্য আগামী ২২ ফেব্রুয়ারি দিন ধার্য করেন আদালত।"

similarity = bn*doc2vec.get*document_similarity(

  model_path,

  article_1,

  article_2
)

print(similarity)

```

### Train doc2vec vector with custom text files
===============================================

```py

from bnlp import BengaliDoc2vec

bn_doc2vec = BengaliDoc2vec()

text_files = "path/myfiles"

checkpoint_path = "msc/logs"

bn*doc2vec.train*doc2vec(

  text_files,

  checkpoint*path=checkpoint*path,

  vector_size=100,

  min_count=2,

  epochs=10
)

it will train doc2vec with your text files and save the train model in checkpoint_path
======================================================================================

```

# Bengali POS Tagging
=====================

## Bengali CRF POS Tagging
==========================

### Find Pos Tag Using Pretrained Model
=======================================

```py

from bnlp import POS

bn_pos = POS()

model*path = "model/bn*pos.pkl"

text = "আমি ভাত খাই।" # or you can pass ['আমি', 'ভাত', 'খাই', '।']

res = bn*pos.tag(model*path, text)

print(res)
[('আমি', 'PPR'), ('ভাত', 'NC'), ('খাই', 'VM'), ('।', 'PU')]
===========================================================

```

### Train POS Tag Model
=======================

```py

from bnlp import POS

bn_pos = POS()

model*name = "pos*model.pkl"

train_data = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা',  'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

test_data = [[('রপ্তানি', 'JJ'), ('দ্রব্য', 'NC'), ('-', 'PU'), ('তাজা', 'JJ'), ('ও', 'CCD'), ('শুকনা', 'JJ'), ('ফল', 'NC'), (',', 'PU'), ('আফিম', 'NC'), (',', 'PU'), ('পশুচর্ম', 'NC'), ('ও', 'CCD'), ('পশম', 'NC'), ('এবং', 'CCD'),('কার্পেট', 'NC'), ('৷', 'PU')], [('মাটি', 'NC'), ('থেকে', 'PP'), ('বড়জোর', 'JQ'), ('চার', 'JQ'), ('পাঁচ', 'JQ'), ('ফুট', 'CCL'), ('উঁচু', 'JJ'), ('হবে', 'VM'), ('৷', 'PU')]]

bn*pos.train(model*name, train*data, test*data)

```

# Bengali NER
=============

## Bengali CRF NER
==================

### Find NER Tag Using Pretrained Model
=======================================

```py

from bnlp import NER

bn_ner = NER()

model*path = "model/bn*ner.pkl"

text = "সে ঢাকায় থাকে।" # or you can pass ['সে', 'ঢাকায়', 'থাকে', '।']

result = bn*ner.tag(model*path, text)

print(result)
[('সে', 'O'), ('ঢাকায়', 'S-LOC'), ('থাকে', 'O')]
================================================

```

### Train NER Tag Model
=======================

```py

from bnlp import NER

bn_ner = NER()

model*name = "ner*model.pkl"

train_data = [[('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')]]

test_data = [[('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')], [('ত্রাণ', 'O'),('ও', 'O'),('সমাজকল্যাণ', 'O'),('সম্পাদক', 'S-PER'),('সুজিত', 'B-PER'),('রায়', 'I-PER'),('নন্দী', 'E-PER'),('প্রমুখ', 'O'),('সংবাদ', 'O'),('সম্মেলনে', 'O'),('উপস্থিত', 'O'),('ছিলেন', 'O')]]

bn*ner.train(model*name, train*data, test*data)

```


# Bengali Corpus Class
======================

## Stopwords and Punctuations
=============================

```py

from bnlp.corpus import stopwords, punctuations, letters, digits

print(stopwords)

print(punctuations)

print(letters)

print(digits)

```

## Remove stopwords from Text
=============================

```py

from bnlp.corpus import stopwords

from bnlp.corpus.util import remove_stopwords

raw_text = 'আমি ভাত খাই।'

result = remove*stopwords(raw*text, stopwords)

print(result)
['ভাত', 'খাই', '।']
===================

```

# Text Cleaning
===============

We adopted different text cleaning formula, codes from `clean-text <https://github.com/jfilter/clean-text>`_ and modified for Bangla. Now you can normalize and clean your text using the following methods.

```py

from bnlp import CleanText

clean_text = CleanText(

	fix\_unicode=True,

	unicode\_norm=True,

	unicode\_norm\_form="NFKC",

	remove\_url=False,

	remove\_email=False,

	remove\_emoji=False,

	remove\_number=False, # not implement yet

	remove\_digits=False, # not implement yet

	remove\_punct=False  # not implement yet

)

input_text = "আমার সোনার বাংলা।"

clean*text = clean*text(input_text)

print(clean_text)

```

# Contributor Guide
===================

Check `CONTRIBUTING.md <https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md>`_ page for details.


# Thanks To
===========

* `Semantics Lab <https://www.facebook.com/lab.semantics/>`_

