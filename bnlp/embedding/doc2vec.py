from scipy import spatial
from gensim.models.doc2vec import Doc2Vec
from bnlp.tokenizer.basic import BasicTokenizer


class BengaliDoc2vec:
    def __init__(self, tokenizer=None):
        if not tokenizer:
            self.tokenizer = BasicTokenizer()
        
    def news_article_similarity(self, model_path, article_1, article_2):
        article_1_tokens = self.tokenizer.tokenize(article_1)
        article_2_tokens = self.tokenizer.tokenize(article_2)
        model = Doc2Vec.load(model_path)

        article_1_vector = model.infer_vector(article_1_tokens)
        article_2_vector = model.infer_vector(article_2_tokens)

        similarity = 1 - spatial.distance.cosine(article_1_vector, article_2_vector)

        return similarity
