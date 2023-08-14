import scipy
import numpy as np
from typing import List
from scipy import spatial

from bnlp.utils.downloader import download_model
from bnlp.utils.config import ModelTypeEnum

class BengaliGlove:
    def __init__(self, glove_vector_path: str = ""):
        if not glove_vector_path:
            glove_vector_path = download_model(ModelTypeEnum.GLOVE)
        self.embedding_dict = self._get_embedding_dict(glove_vector_path)

    def get_word_vector(self, word: str) -> np.ndarray:
        word_vector = self.embedding_dict[word]
        return word_vector

    def get_closest_word(self, word: str) -> List[str]:
        def find_closest_embeddings(embedding):
            return sorted(
                self.embedding_dict.keys(),
                key=lambda word: spatial.distance.euclidean(
                    self.embedding_dict[word], embedding
                ),
            )

        result = find_closest_embeddings(self.embedding_dict[word])[:10]
        return result

    def _get_embedding_dict(self, glove_vector_path: str):
        embeddings_dict = {}
        with open(glove_vector_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        return embeddings_dict
