#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class BengaliWord2Vec:
    def train(self, data_file, model_name, vector_name, vector_size=300, 
    window=1, min_count=1, max_vocab_size=None, workers=3, epochs=5, 
    compute_loss=True):

        """
        :data_file: (str) input text data file with name and extension
        :model_name: (str) model path with file name and extension
        :vector_name: (str) vector path with file name and extension

        """
        model = Word2Vec(
            LineSentence(data_file), 
            size=vector_size, # vector_size = 200
            window=window, 
            min_count=min_count, 
            max_vocab_size=max_vocab_size, 
            workers=workers,
            iter=epochs, 
            compute_loss=compute_loss
        ) 

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


