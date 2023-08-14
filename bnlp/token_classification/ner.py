"""
tool: We used sklearn crf_suite for bengali name entity recognition
https://sklearn-crfsuite.readthedocs.io/en/latest/

"""

import string
from sklearn_crfsuite import CRF
from typing import Union, List, Tuple, Callable

from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.utils.utils import load_pickle_model
from bnlp.utils.utils import features
from bnlp.utils.downloader import download_model

class BengaliNER:
    def __init__(self, model_path: str = "", tokenizer: Callable = None):
        if not model_path:
            model_path = download_model("NER")
        self.model = load_pickle_model(model_path)
        self.tokenizer = tokenizer if tokenizer else BasicTokenizer()

    def tag(self, text: str) -> List[Tuple[str, str]]:
        punctuations = string.punctuation + "।"

        tokens = self.tokenizer(text)
        # remove punctuation from tokens
        tokens = [x for x in tokens if x not in punctuations]

        sentence_features = [
            features(tokens, index) for index in range(len(tokens))
        ]
        result = list(zip(tokens, self.model.predict([sentence_features])[0]))

        return result
