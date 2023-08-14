import unittest
from bnlp.token_classification.ner import BengaliNER

class TestBengaliNER(unittest.TestCase):
    def setUp(self):
        self.ner = BengaliNER()

    def test_tag(self):
        text = "সে ঢাকায় থাকে।"
        tags = self.ner.tag(text)
        self.assertEqual(tags, [("সে", "O"), ("ঢাকায়", "S-LOC"), ("থাকে", "O")])

if __name__ == '__main__':
    unittest.main()