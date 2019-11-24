# Bengali Natural Language Processing(BNLP)

BNLP is a natural language processing toolkit for Bengali Language. This tool will help you to **tokenize Bengali text**, **Embedding Bengali words**, **construct neural model** for Bengali NLP purposes.

## Installation

* local install

  ```
  git clone https://github.com/sagorbrur/bnlp.git
  cd bnlp
  python setup.py install

  ```

* pypi package installer

  ```python -m pip install bnlp-tool```


## Tokenization

* **Bengali SentencePiece Tokenization**

  - tokenization using trained model
    ```py
    from bnlp.tokenizer.sentencepiece_tokenizer import SP_Tokenizer

    bsp = SP_Tokenizer()
    model_path = "./model/bn_spm.model"
    input_text = "আমি ভাত খাই। সে বাজারে যায়।"
    tokens = bsp.tokenize(model_path, input_text)
    print(tokens)

    ```
  - Training SentencePiece
    ```py
    from bnlp.tokenizer.sentencepiece_tokenizer import SP_Tokenizer
    
    bsp = SP_Tokenizer(is_train=True)
    data = "test.txt"
    model_prefix = "test"
    vocab_size = 5
    bsp.train_bsp(data, model_prefix, vocab_size) 

    ```

* **NLTK Tokenization**

```py
from bnlp.tokenizer.nltk_tokenizer import NLTK_Tokenizer

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
    from bnlp.embedding.bengali_word2vec import Bengali_Word2Vec

    bwv = Bengali_Word2Vec()
    model_path = "model/wiki.bn.text.model"
    word = 'আমার'
    vector = bwv.generate_word_vector(model_path, word)
    print(vector.shape)
    print(vector)

    ```

  - Find Most Similar Word Using Pretrained Model

    ```py
    from bnlp.embedding.bengali_word2vec import Bengali_Word2Vec

    bwv = Bengali_Word2Vec()
    model_path = "model/wiki.bn.text.model"
    word = 'আমার'
    similar = bwv.most_similar(model_path, word)
    print(similar)

    ```
  - Train Bengali Word2Vec with your own data

    ```py
    from bnlp.embedding.bengali_word2vec import Bengali_Word2Vec

    data_file = "test.txt"
    model_name = "test_model.model"
    vector_name = "test_vector.vector"
    bwv.train_word2vec(data_file, model_name, vector_name)


    ```
    
 * **Bengali FastText**
 

    - Download Bengali FastText Pretrained Model From [Here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.bin.gz)

    - Generate Vector Using Pretrained Model
      

      ```py
      from bnlp.embedding.bengali_fasttext import Bengali_Fasttext

      bft = Bengali_Fasttext()
      word = "গ্রাম"
      model_path = "cc.bn.300.bin"
      word_vector = bf.generate_word_vector(model_path, word)
      print(word_vector.shape)
      print(word_vector)


      ```
    - Train Bengali FastText Model

      ```py
      from bnlp.embedding.bengali_fasttext import Bengali_Fasttext

      bft = Bengali_Fasttext(is_train=True)
      data = "data.txt"
      model_name = "saved_model.bin"
      bf.train_fasttext(data, model_name)

      ```

## Issue
* if `ModuleNotFoundError: No module named 'fasttext'` problem arise please do the next line

```pip install fasttext```

## Developer Guide

* `Fork`
* `add` or `modify`
* send `pull request` for merging
* we will verify and include your name in `Contributor List`

## Contributor List

* [Sagor Sarker](https://github.com/sagorbrur)
* [Faruk Ahmad](https://github.com/faruk-ahmad)
