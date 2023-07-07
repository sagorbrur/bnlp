__version__ = "4.0.0"


import os
from bnlp.pos import BengaliPOS
from bnlp.ner import BengaliNER
from bnlp.token_classification_trainer import CRFTaggerTrainer
from bnlp.tokenizer.nltk import NLTKTokenizer
from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.tokenizer.sentencepiece import (
    SentencepieceTokenizer, 
    SentencepieceTrainer,
)
from bnlp.embedding.word2vec import (
    BengaliWord2Vec,
    Word2VecTraining,
)
from bnlp.embedding.glove import BengaliGlove
from bnlp.embedding.doc2vec import (
    BengaliDoc2vec, 
    BengaliDoc2vecTrainer,
)
from bnlp.cleantext.clean import CleanText