import os
import glob
import gensim
from tqdm import tqdm
from scipy import spatial
from gensim.models.doc2vec import Doc2Vec
from bnlp.tokenizer.basic import BasicTokenizer

default_tokenizer = BasicTokenizer()

def read_corpus(files, tokenizer=None):
    for i, file in tqdm(enumerate(files)):
      with open(file) as f:
        text = f.read()
        if tokenizer:
            tokens = tokenizer(text)
        else:
            tokens = default_tokenizer.tokenize(text)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
          

class BengaliDoc2vec:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def get_document_vector(self, model_path, document):
        """Get document vector using trained doc2vec model

        Args:
            model_path (bin): trained doc2vec model path
            document (str): input documents

        Returns:
            ndarray: generated vector 
        """
        model = Doc2Vec.load(model_path)
        if self.tokenizer:
            tokens = self.tokenizer(document)
        else:
            tokens = default_tokenizer.tokenize(document)

        vector = model.infer_vector(tokens)

        return vector

    def get_document_similarity(self, model_path, document_1, document_2):
        """Get document similarity score from input two document using pretrained doc2vec model

        Args:
            model_path (bin): pretrained doc2vec
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

        model = Doc2Vec.load(model_path)

        document_1_vector = model.infer_vector(document_1_tokens)
        document_2_vector = model.infer_vector(document_2_tokens)

        similarity = round(1 - spatial.distance.cosine(document_1_vector, document_2_vector), 2)

        return similarity

    def train_doc2vec(self, text_files, checkpoint_path='ckpt', vector_size=100, min_count=2, epochs=10):
        text_files = glob.glob(text_files + '/*.txt')
        if self.tokenizer:
            train_corpus = list(read_corpus(text_files, self.tokenizer))
        else:
            train_corpus = list(read_corpus(text_files))

        model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        
        os.makedirs(checkpoint_path, exist_ok=True)
        output_model_name = os.path.join(checkpoint_path, 'custom_doc2vec_model.model')
        model.save(output_model_name)
