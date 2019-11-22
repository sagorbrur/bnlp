#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence




class Bengali_Word2Vec(object):
    def __init__(self, is_train=False):
        self.is_train = is_train

        """
        :is_train: boolean value to choose training option

        """

    def train_word2vec(self, data_file, model_name, vector_name):

        """
        :data_file: (str) input text data file with name and extension
        :model_name: (str) model path with file name and extension
        :vector_name: (str) vector path with file name and extension

        """
        
        if self.is_train:
            model = Word2Vec(LineSentence(data_file), size=300, window=5, min_count=1,
                        workers=multiprocessing.cpu_count()) # size = 200

            model.save(model_name)
            model.wv.save_word2vec_format(vector_name, binary=False)
            print("%s and %s saved in your current directory."%(model_name, vector_name))

    def generate_word_vector(self, model_path, input_word):
        """
        :model_name: (str) model path with file name and extension
        :input_word: (str) word to generate vector

        """
        model = Word2Vec.load(model_path)
        vector = model.wv[input_word]
        return vector

    def most_similar(self, model_path, word):
        """
        :model_name: (str) model path with file name and extension
        :word: (str) word to find similar word

        """
        model = Word2Vec.load(model_path)
        similar_word = model.most_similar(word)
        return similar_word

    def generate_text2vector(self, model_path, input_text):
        pass
