import unittest
from bnlp import BengaliGlove

class TestBengaliGlove(unittest.TestCase):
    def setUp(self):
        self.glove = BengaliGlove()

    def test_get_word_vector(self):
        word = "আমি"
        vector = self.glove.get_word_vector(word)
        self.assertEqual(vector.shape, (100,))

if __name__ == '__main__':
    unittest.main()
