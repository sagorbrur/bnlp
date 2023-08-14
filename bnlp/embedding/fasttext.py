import multiprocessing
import numpy as np

from bnlp.utils.downloader import download_model
from bnlp.utils.config import ModelTypeEnum

try:
    import fasttext
except ImportError:
    print("fasttext not installed. Install it by 'pip install fasttext'")

class BengaliFasttext:
    def __init__(self, model_path: str = ""):
        if not model_path:
            model_path = download_model(ModelTypeEnum.FASTTEXT)
        self.model = fasttext.load_model(model_path)

    def get_word_vector(self, word: str) -> np.ndarray:
        """generate word vector from given input word

        Args:
            word (str): input word or token

        Returns:
            str: word or token vector
        """
        word_vector = self.model[word]

        return word_vector

    def bin2vec(self, vector_name: str):
        """Generate vector text file from fasttext binary model

        Args:
            vector_name (str): name of the output vector with extension
        """
        output_vector = open(vector_name, "w")

        words = self.model.get_words()
        vocab_len = str(len(words))
        dimension = str(self.model.get_dimension())
        output_vector.write(vocab_len + " " + dimension + "\n")
        for w in words:
            v = self.model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            output_vector.write(w + vstr + "\n")
        output_vector.close()

class FasttextTrainer:
    def train(
        self,
        data,
        model_name,
        epoch,
        lr=0.05,
        dim=300,
        ws=5,
        minCount=5,
        minn=3,
        maxn=6,
        neg=5,
        wordNgrams=1,
        loss="ns",
        bucket=2000000,
        thread=multiprocessing.cpu_count() - 1,
    ):
        """train fasttext with raw text data

        Args:
            data (str): raw text data path
            model_name (str): name of output trained model with extension
            epoch (int): number of training iteration
            lr (float, optional): learning rate. Defaults to 0.05.
            dim (int, optional): vector size or dimension. Defaults to 300.
            ws (int, optional): window size. Defaults to 5.
            minCount (int, optional): minimum word count to ignore training. Defaults to 5.
            minn (int, optional): [description]. Defaults to 3.
            maxn (int, optional): [description]. Defaults to 6.
            neg (int, optional): negative sampling. Defaults to 5.
            wordNgrams (int, optional): [description]. Defaults to 1.
            loss (str, optional): loss type . Defaults to "ns".
            bucket (int, optional): [description]. Defaults to 2000000.
            thread ([type], optional): [description]. Defaults to multiprocessing.cpu_count()-1.
        """
        print("training started.....")
        model = fasttext.train_unsupervised(
            data,
            model="skipgram",
            epoch=epoch,
            lr=lr,
            dim=dim,
            ws=ws,
            minCount=minCount,
            minn=minn,
            maxn=maxn,
            neg=neg,
            wordNgrams=wordNgrams,
            loss=loss,
            bucket=bucket,
            thread=thread,
        )
        print(f"training done! saving as {model_name}")
        model.save_model(model_name)
