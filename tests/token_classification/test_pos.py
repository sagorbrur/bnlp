import unittest
from bnlp.token_classification.pos import BengaliPOS

class TestBengaliNER(unittest.TestCase):
    def setUp(self):
        self.ner = BengaliPOS()

    def test_tag(self):
        text = "আমি ভাত খাই।"
        tags = self.ner.tag(text)
        self.assertEqual(tags, [("আমি", "PPR"), ("ভাত", "NC"), ("খাই", "VM"), ("।", "PU")])

if __name__ == '__main__':
    unittest.main()