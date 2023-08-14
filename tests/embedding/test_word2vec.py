import unittest
from bnlp import BengaliWord2Vec

class TestBengaliWord2Vec(unittest.TestCase):
    def setUp(self):
        self.word2vec = BengaliWord2Vec()

    def test_get_word_vector(self):
        word = "আমি"
        vector = self.word2vec.get_word_vector(word)
        self.assertEqual(vector.shape, (100,))

    def test_get_most_similar_words(self):
        word = "আমি"
        topn = 5
        similar_words = self.word2vec.get_most_similar_words(word, topn=topn)
        self.assertEqual(len(similar_words), topn)
        self.assertTrue(all(isinstance(word, str) and isinstance(similarity, float) for word, similarity in similar_words))


if __name__ == '__main__':
    unittest.main()
