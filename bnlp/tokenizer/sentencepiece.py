import os
from typing import List
import sentencepiece as bsp


from bnlp.utils.downloader import download_model
from bnlp.utils.config import ModelTypeEnum

class SentencepieceTokenizer:
    def __init__(self, model_path: str = ""):
        if not model_path:
            model_path = download_model(ModelTypeEnum.SENTENCEPIECE)
        self.model = bsp.SentencePieceProcessor()
        self.model.Load(model_path)

    def tokenize(self, text: str) -> List[str]:
        tokens = self.model.EncodeAsPieces(text)

        return tokens

    def text2id(self, text: str) -> List[int]:
        ids = self.model.EncodeAsIds(text)
        return ids

    def id2text(self, ids: List[int]) -> str:
        text = self.model.DecodeIds(ids)
        return text

class SentencepieceTrainer:
    def __init__(self, data, vocab_size, model_prefix):
        self.data = data
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix

    def train(self):
        train_args = (
            "--model_prefix="
            + self.model_prefix
            + " --input="
            + self.data
            + " --vocab_size="
            + str(self.vocab_size)
        )
        bsp.SentencePieceTrainer.train(train_args)
        print(
            "%s.model and %s.vocab is saved on your current directory"
            % (self.model_prefix, self.model_prefix)
        )