__version__ = "3.3.2"

import os
from bnlp.pos import POS
from bnlp.ner import NER
from bnlp.tokenizer.nltk import NLTKTokenizer
from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.tokenizer.sentencepiece import SentencepieceTokenizer
from bnlp.embedding.word2vec import BengaliWord2Vec
from bnlp.embedding.glove import BengaliGlove
from bnlp.embedding.doc2vec import BengaliDoc2vec
from bnlp.cleantext.clean import CleanText

import logging

logging.warning(f"""The current version ({__version__}) of bnlp will not be compatible with the upcoming release.
If you are using version <={__version__} please specify bnlp_toolkit with exact version, otherwise it will raises error in the upcoming version. 
To migrate feel free to checkout the newer version (4.0.0). It will release soon as beta.""")