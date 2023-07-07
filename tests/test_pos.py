import unittest
from bnlp.pos import BengaliPOS

class TestBengaliNER(unittest.TestCase):
    def setUp(self):
        model_path = "model/bn_pos.pkl"
        self.ner = BengaliPOS(model_path)

    def test_tag(self):
        text = "আমি ভাত খাই।"
        tags = self.ner.tag(text)
        self.assertEqual(tags, [("আমি", "PPR"), ("ভাত", "NC"), ("খাই", "VM"), ("।", "PU")])

if __name__ == '__main__':
    unittest.main()