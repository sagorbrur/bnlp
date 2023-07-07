import unittest
from bnlp.ner import BengaliNER

class TestBengaliNER(unittest.TestCase):
    def setUp(self):
        model_path = "model/bn_ner.pkl"
        self.ner = BengaliNER(model_path)

    def test_tag(self):
        text = "সে ঢাকায় থাকে।"
        tags = self.ner.tag(text)
        self.assertEqual(tags, [("সে", "O"), ("ঢাকায়", "S-LOC"), ("থাকে", "O")])

if __name__ == '__main__':
    unittest.main()