from scipy import spatial
from gensim.models.doc2vec import Doc2Vec
from bnlp.tokenizer.basic import BasicTokenizer

default_tokenizer = BasicTokenizer()

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
