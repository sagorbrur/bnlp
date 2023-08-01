import warnings
warnings.filterwarnings("ignore")

import os
import glob
import gensim
import numpy as np
from tqdm import tqdm
from scipy import spatial
from typing import Callable, List
from gensim.models.doc2vec import Doc2Vec

from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.utils.downloader import download_model
from bnlp.utils.config import ModelTypeEnum

default_tokenizer = BasicTokenizer()


def _read_corpus(files: List[str], tokenizer=None):
    for i, file in tqdm(enumerate(files)):
        with open(file) as f:
            text = f.read()
            if tokenizer:
                tokens = tokenizer(text)
            else:
                tokens = default_tokenizer.tokenize(text)
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


class BengaliDoc2vec:
    def __init__(
        self,
        model_path: str = "",
        tokenizer: Callable = None
        ):
        if model_path == "" or model_path == ModelTypeEnum.NEWS_DOC2VEC:
            model_path = download_model(ModelTypeEnum.NEWS_DOC2VEC)
        if model_path == ModelTypeEnum.WIKI_DOC2VEC:
            model_path = download_model(ModelTypeEnum.WIKI_DOC2VEC)
        self.tokenizer = tokenizer
        self.model = Doc2Vec.load(model_path)

    def get_document_vector(self, document: str) -> np.ndarray:
        """Get document vector using trained doc2vec model

        Args:
            document (str): input documents

        Returns:
            ndarray: generated vector
        """
        
        if self.tokenizer:
            tokens = self.tokenizer(document)
        else:
            tokens = default_tokenizer.tokenize(document)

        vector = self.model.infer_vector(tokens)

        return vector

    def get_document_similarity(self, document_1: str, document_2: str) -> float:
        """Get document similarity score from input two document using pretrained doc2vec model

        Args:
            document_1 (str): input document
            document_2 (str): input document

        Returns:
            float: output similarity score
        """
        if self.tokenizer:
            document_1_tokens = self.tokenizer(document_1)
            document_2_tokens = self.tokenizer(document_2)
        else:
            document_1_tokens = default_tokenizer.tokenize(document_1)
            document_2_tokens = default_tokenizer.tokenize(document_2)

        document_1_vector = self.model.infer_vector(document_1_tokens)
        document_2_vector = self.model.infer_vector(document_2_tokens)

        similarity = round(
            1 - spatial.distance.cosine(document_1_vector, document_2_vector), 2
        )

        return similarity

class BengaliDoc2vecTrainer:
    def __init__(self, tokenizer: Callable = None):
        self.tokenizer = tokenizer

    def train(
        self,
        text_files,
        checkpoint_path="ckpt",
        vector_size=100,
        min_count=2,
        epochs=10,
    ):
        """Train doc2vec with custom text files

        Args:
            text_files (str): path contains the text files with extension .txt
            checkpoint_path (str, optional): checkpoint save path. Defaults to 'ckpt'.
            vector_size (int, optional): size of the vector. Defaults to 100.
            min_count (int, optional): minimum word count. Defaults to 2.
            epochs (int, optional): training iteration number. Defaults to 10.
        """
        text_files = glob.glob(text_files + "/*.txt")
        if self.tokenizer:
            train_corpus = list(_read_corpus(text_files, self.tokenizer))
        else:
            train_corpus = list(_read_corpus(text_files))

        model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
        model.build_vocab(train_corpus)
        model.train(
            train_corpus, total_examples=model.corpus_count, epochs=model.epochs
        )

        os.makedirs(checkpoint_path, exist_ok=True)
        output_model_name = os.path.join(checkpoint_path, "custom_doc2vec_model.model")
        model.save(output_model_name)