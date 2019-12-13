Bengali Natural Language Processing(BNLP)
=========================================

[![Build Status](https://travis-ci.org/sagorbrur/bnlp.svg?branch=master)](https://travis-ci.org/sagorbrur/bnlp)

[![PyPI version](https://img.shields.io/pypi/v/bnlp_toolkit)](https://pypi.org/project/bnlp-toolkit/)

[![release version](https://img.shields.io/github/v/release/sagorbrur/bnlp)](https://github.com/sagorbrur/bnlp/releases/tag/1.1.0)

[![Support Python Version](https://img.shields.io/badge/python-3.6%7C3.7-brightgreen)](https://pypi.org/project/bnlp-toolkit/)

[![pypi Downloads](https://img.shields.io/pypi/dw/bnlp_toolkit?color=green)](https://pypi.org/project/bnlp-toolkit/)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **construct neural model** for Bengali NLP purposes.


Contents
========
- `Current Features <#current-features>`_
- `Installation <#installation>`_
- `Pretrained Model <#pretrained-model>`_
- `Tokenization <#tokenization>`_
- `Embedding <#word-embedding>`_
- `Issue <#issue>`_
- `Contributor Guide <#contributor-guide>`_
- `Contributor List <#contributor-list>`_


Current Features
==================

* `Bengali Tokenization <#tokenization>`_

  - SentencePiece Tokenizer

  - Basic Tokenizer

  - NLTK Tokenizer
* `Bengali Word Embedding <#word-embedding>`_

  - Bengali Word2Vec

  - Bengali Fasttext

  - Bengali GloVe


Installation
==============

* pypi package installer(python 3.6, 3.7 tested okay)

  ```pip install bnlp_toolkit```
  
* Local

  ```

  $git clone https://github.com/sagorbrur/bnlp.git

  $cd bnlp

  $python setup.py install

  ```



Pretrained Model
==================

Download Link
================

* `Bengali SentencePiece <https://github.com/sagorbrur/bnlp/tree/master/model>`_

* `Bengali Word2Vec <https://drive.google.com/open?id=1DxR8Vw61zRxuUm17jzFnOX97j7QtNW7U>`_

* `Bengali FastText <https://drive.google.com/open?id=1CFA-SluRyz3s5gmGScsFUcs7AjLfscm2>`_

* `Bengali GloVe Wordvectors <https://github.com/sagorbrur/GloVe-Bengali>`_

Training Details
===================

* All three model trained with **Bengali Wikipedia Dump Dataset**

  - `Bengali Wiki Dump <https://dumps.wikimedia.org/bnwiki/latest/>`_
* SentencePiece Training Vocab Size=50000

* Fasttext trained with total words = 20M, vocab size = 1171011, epoch=50, embedding dimension = 300 and the training loss = 0.318668,

* Word2Vec word embedding dimension = 300

* To Know Bengali GloVe Wordvector and training process follow `this <https://github.com/sagorbrur/GloVe-Bengali>`_ repository


Tokenization
==============

* **Bengali SentencePiece Tokenization**

  - tokenization using trained model

	```py

	from bnlp.sentencepiece\_tokenizer import SP\_Tokenizer

	bsp = SP\_Tokenizer()

	model\_path = "./model/bn\_spm.model"

	input\_text = "আমি ভাত খাই। সে বাজারে যায়।"

	tokens = bsp.tokenize(model\_path, input\_text)

	print(tokens)

	```

  - Training SentencePiece

	```py

	from bnlp.sentencepiece\_tokenizer import SP\_Tokenizer
	
	bsp = SP\_Tokenizer(is\_train=True)

	data = "test.txt"

	model\_prefix = "test"

	vocab\_size = 5

	bsp.train\_bsp(data, model\_prefix, vocab\_size) 

	```

* **Basic Tokenizer**

 

  ```py

  from bnlp.basic_tokenizer import BasicTokenizer

  basic_t = BasicTokenizer(False)

  raw_text = "আমি বাংলায় গান গাই।"

  tokens = basic*t.tokenize(raw*text)

  print(tokens)
  
  # output: ["আমি", "বাংলায়", "গান", "গাই", "।"]

  ```

* **NLTK Tokenization**

  ```py

  from bnlp.nltk*tokenizer import NLTK*Tokenizer

  text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"

  bnltk = NLTK_Tokenizer(text)

  word*tokens = bnltk.word*tokenize()

  sentence*tokens = bnltk.sentence*tokenize()

  print(word_tokens)

  print(sentence_tokens)
  
  # output

  # word_token: ["আমি", "ভাত", "খাই", "।", "সে", "বাজারে", "যায়", "।", "তিনি", "কি", "সত্যিই", "ভালো", "মানুষ", "?"]

  # sentence_token: ["আমি ভাত খাই।", "সে বাজারে যায়।", "তিনি কি সত্যিই ভালো মানুষ?"]

  ```


Word Embedding
================

* **Bengali Word2Vec**

  - Generate Vector using pretrain model

	```py

	from bnlp.bengali\_word2vec import Bengali\_Word2Vec

	bwv = Bengali\_Word2Vec()

	model\_path = "model/bengali\_word2vec.model"

	word = 'আমার'

	vector = bwv.generate\_word\_vector(model\_path, word)

	print(vector.shape)

	print(vector)

	```

  - Find Most Similar Word Using Pretrained Model

	```py

	from bnlp.bengali\_word2vec import Bengali\_Word2Vec

	bwv = Bengali\_Word2Vec()

	model\_path = "model/bengali\_word2vec.model"

	word = 'আমার'

	similar = bwv.most\_similar(model\_path, word)

	print(similar)

	```

  - Train Bengali Word2Vec with your own data

	```py

	from bnlp.bengali\_word2vec import Bengali\_Word2Vec

	bwv = Bengali\_Word2Vec(is\_train=True)

	data\_file = "test.txt"

	model\_name = "test\_model.model"

	vector\_name = "test\_vector.vector"

	bwv.train\_word2vec(data\_file, model\_name, vector\_name)


	```
	
 * **Bengali FastText**
 

	\- Generate Vector Using Pretrained Model
	  

	  ```py

	  from bnlp.bengali\_fasttext import Bengali\_Fasttext

	  bft = Bengali\_Fasttext()

	  word = "গ্রাম"

	  model\_path = "model/bengali\_fasttext.bin"

	  word\_vector = bft.generate\_word\_vector(model\_path, word)

	  print(word\_vector.shape)

	  print(word\_vector)


	  ```

	\- Train Bengali FastText Model

	  ```py

	  from bnlp.bengali\_fasttext import Bengali\_Fasttext

	  bft = Bengali\_Fasttext(is\_train=True)

	  data = "data.txt"

	  model\_name = "saved\_model.bin"

	  bft.train\_fasttext(data, model\_name)

	  ```

* **Bengali GloVe Word Vectors**

  We trained glove model with bengali data(wiki+news articles) and published bengali glove word vectors</br>

  You can download and use it on your different machine learning purposes.

  ```py

  from bnlp.glove*wordvector import BN*Glove

  glove*path = "bn*glove.39M.100d.txt"

  word = "গ্রাম"

  bng = BN_Glove()

  res = bng.closest*word(glove*path, word)

  print(res)

  vec = bng.word2vec(glove_path, word)

  print(vec)

  ```

Issue
=======

* if `ModuleNotFoundError: No module named 'fasttext'` problem arise please do the next line

```pip install fasttext```

* if `nltk` issue arise please do the following line before importing `bnlp`

```py

import nltk

nltk.download("punkt")

```


Contributor Guide
===================

Check `CONTRIBUTING.md <https://github.com/sagorbrur/bnlp/blob/master/CONTRIBUTING.md>`_ page for details.


Thanks To
===========

* `Semantics Lab <http://semanticslab.net/>`_

Contributor List
==================

* `Sagor Sarker <https://github.com/sagorbrur>`_

* `Faruk Ahmad <https://github.com/faruk-ahmad>`_

* `Mehadi Hasan Menon <https://github.com/menon92>`_

* `Kazal Chandra Barman <https://github.com/kazalbrur>`_

* `Md Ibrahim <https://github.com/iriad11>`_

