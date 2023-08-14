import unittest
import numpy as np
from bnlp import BengaliFasttext

class TestBengaliFasttext(unittest.TestCase):
    def setUp(self):
        self.fasttext = BengaliFasttext()

    def test_generate_word_vector(self):
        word = "আমি"
        vector = self.fasttext.generate_word_vector(word)
        self.assertEqual(vector.shape, (300,))

if __name__ == '__main__':
    unittest.main()
