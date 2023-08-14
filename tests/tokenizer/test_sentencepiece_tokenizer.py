import unittest
from bnlp import SentencepieceTokenizer

class TestSentencepieceTokenizer(unittest.TestCase):
    def setUp(self):
        self.bsp = SentencepieceTokenizer()
        self.input_text = "সে বাজারে যায়।"
        self.input_text_gt_tokens = ['▁সে', '▁বাজারে', '▁যায়', '।']

    def test_sentencepiece_tokenizer_with_input_bangla_text_and_trained_model(self):
        tokens = self.bsp.tokenize(self.input_text)
        self.assertEqual(tokens, self.input_text_gt_tokens)

if __name__ == "__main__":
    unittest.main()
