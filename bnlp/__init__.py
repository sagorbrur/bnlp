__version__="1.1.0"


import os
import logging
logger = logging.getLogger(__name__)

from bnlp.sentencepiece_tokenizer import SP_Tokenizer
from bnlp.nltk_tokenizer import NLTK_Tokenizer
from bnlp.basic_tokenizer import BasicTokenizer
from bnlp.bengali_word2vec import Bengali_Word2Vec
from bnlp.bengali_fasttext import Bengali_Fasttext


from .utils import is_torch_available


if is_torch_available():
    from bnlp.sentiment_analysis import Sequences
    from bnlp.sentiment_analysis import RNN
    from bnlp.sentiment_analysis import BN_Sentiment



if not is_torch_available():
    logger.warning("For sentiment analysis please install torch using", 
    "https://github.com/sagorbrur/bnlp/blob/master/README.md instruction.")

