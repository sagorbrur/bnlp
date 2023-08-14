"""
tool: We used sklearn crf_suite for bengali pos tagging
https://sklearn-crfsuite.readthedocs.io/en/latest/
"""
from sklearn_crfsuite import CRF
from typing import Union, List, Tuple, Callable

from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.utils.utils import load_pickle_model
from bnlp.utils.utils import features
from bnlp.utils.downloader import download_model

class BengaliPOS:
    def __init__(self, model_path: str = "", tokenizer: Callable = None):
        if not model_path:
            model_path = download_model("POS")
        self.model = load_pickle_model(model_path)
        self.tokenizer = tokenizer if tokenizer else BasicTokenizer()


    def tag(self, text: str) -> List[Tuple[str, str]]:
        tokens = self.tokenizer(text)
        # remove punctuation from tokens
        sentence_features = [
            features(tokens, index) for index in range(len(tokens))
        ]
        result = list(zip(tokens, self.model.predict([sentence_features])[0]))

        return result
