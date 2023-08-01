#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import multiprocessing
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from bnlp.tokenizer.nltk import NLTKTokenizer

from bnlp.utils.downloader import download_model
from bnlp.utils.config import ModelTypeEnum

from typing import List, Tuple


class BengaliWord2Vec:
    def __init__(self, model_path: str = ""):
        if not model_path:
            model_path = download_model(ModelTypeEnum.WORD2VEC)
        self.model = Word2Vec.load(model_path)
    
    def get_word_vector(self, word: str) -> np.ndarray:
        vector = self.model.wv[word]
        return vector

    def get_most_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        
        similar_word = self.model.wv.most_similar(word, topn=topn)
        return similar_word


class MyCorpus:
    """An iterator that yields sentences (lists of str).
    We used NLTKTokenizer from bnlp to tokenize sentence words
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.bnltk = NLTKTokenizer()

    def __iter__(self):
        for line in open(self.data_path):
            sentences = self.bnltk.sentence_tokenize(line)
            for sentence in sentences:
                tokens = self.bnltk.word_tokenize(sentence)
                yield tokens


class Word2VecTraining:
    def train(
        self,
        data_path,
        model_name,
        vector_name,
        vector_size=100,
        alpha=0.025,
        min_alpha=0.0001,
        sg=0,
        hs=0,
        negative=5,
        ns_exponent=0.75,
        window=5,
        min_count=5,
        max_vocab_size=None,
        workers=3,
        epochs=5,
        sample=1e-3,
        cbow_mean=1,
        compute_loss=True,
        callbacks=(),
    ):
        """train bengali word2vec

        Args:
            data_path (str/list): raw text data path as string with extension or
                                  sentence token list. example: [[], []]
            model_name (str): output model name ex: mymodel.model
            vector_name (str): output vector name ex: myvector.txt
            vector_size (int, optional): vector dimension. Defaults to 100.
            alpha (float, optional): initial learning rate. Defaults to 0.025.
            min_alpha (float, optional): minimum learning rate. Defaults to 0.0001.
            sg (int, optional): skip-gram model or cbow model. if 1 then skip-gram. Defaults to 0.
            hs (int, optional): hierarchical softmax. Defaults to 0.
            negative (int, optional): negative sampling. Defaults to 5.
            ns_exponent (float, optional): The exponent used to shape the
                        negative sampling distribution. Defaults to 0.75.
            window (int, optional): window size. Defaults to 5.
            min_count (int, optional): minimum word count to ignore. Defaults to 5.
            max_vocab_size ([type], optional): maximum vocab size. Defaults to None.
            workers (int, optional): worker number. Defaults to 3.
            epochs (int, optional): number of training iteration. Defaults to 5.
            sample ([type], optional): sampling rate. Defaults to 1e-3.
            cbow_mean (int, optional): cbow_mean or cbow_sum. Defaults to 1.
            compute_loss (bool, optional): compute training loss. Defaults to True.
            callbacks (tuple, optional): callback sequence. Defaults to ().
        """
        if isinstance(data_path, list):
            sentences = data_path
        else:
            sentences = MyCorpus(data_path)

        print("training started.......")
        print(
            "please wait.....it will take time according to your data size and computation capability"
        )
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            alpha=alpha,
            min_alpha=min_alpha,
            sg=sg,
            hs=hs,
            negative=negative,
            ns_exponent=ns_exponent,
            sample=sample,
            cbow_mean=cbow_mean,
            window=window,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            workers=workers,
            epochs=epochs,
            compute_loss=compute_loss,
            callbacks=callbacks,
        )
        # getting the training loss value
        training_loss = model.get_latest_training_loss()

        print("train completed successfully")
        print(f"trianing loss: {training_loss}")
        print("model and vector saving...")
        model.save(model_name)
        model.wv.save_word2vec_format(vector_name, binary=False)
        print(f"model and vector saved as {model_name} and {vector_name}")

    def pretrain(
        self, model_path, new_sentences, output_model_name, output_vector_name, epochs=5
    ):
        """resume training from saved word2vec model

        Args:
            model_path (bin): path of trained word2vec model
            new_sentences (list): list of new sentences
            output_model_name (str): output model name
            output_vector_name (str): output vector name
            epoch(int): number of training iteration
        """
        if isinstance(new_sentences, str):
            new_sentences = MyCorpus(new_sentences)
        print("model loading ....")
        model = Word2Vec.load(model_path)
        print("vocab building with new sentences")
        model.build_vocab(new_sentences, update=True)
        print("pre-training started.......")
        print(
            "please wait.....it will take time according to your data size and computation capability"
        )
        model.train(new_sentences, total_examples=model.corpus_count, epochs=epochs)
        # getting the training loss value
        training_loss = model.get_latest_training_loss()

        print("pre-train completed successfully")
        print(f"pre-trianing loss: {training_loss}")
        print("model and vector saving...")
        model.save(output_model_name)
        model.wv.save_word2vec_format(output_vector_name, binary=False)
        print(
            f"model and vector saved as {output_model_name} and {output_vector_name}"
        )