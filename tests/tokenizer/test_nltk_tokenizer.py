import unittest
from bnlp import NLTKTokenizer

class TestBasicTokenizer(unittest.TestCase):
    def setUp(self):
        self.nltk_tokenizer = NLTKTokenizer()
    
    def test_nltk_word_tokenizer_with_sample_bangla_text(self):
        text = "আমি ভাত খাই।"
        tokens = self.nltk_tokenizer.word_tokenize(text)
        self.assertEqual(tokens, ["আমি", "ভাত", "খাই", "।"])

    def test_nltk_word_tokenizer_with_long_bangla_text(self):
        text = """
        ভারত থেকে অনুপ্রবেশ ঠেকাতে বর্ডার গার্ড বাংলাদেশের (বিজিবি)
         সঙ্গে রাজশাহীর চরখানপুর সীমান্ত পাহারা দিচ্ছেন গ্রামবাসী।
         সীমান্তে নজরদারি জোরদার করার জন্য     চরখানপুর গ্রামের প্রায় আড়াই শ
         বাসিন্দা রাত জেগে পালাক্রমে এই কাজ করছেন গত ২৮ নভেম্বর থেকে।
        """
        tokens = self.nltk_tokenizer.word_tokenize(text)
        gt_tokens = [
            "ভারত",
            "থেকে",
            "অনুপ্রবেশ",
            "ঠেকাতে",
            "বর্ডার",
            "গার্ড",
            "বাংলাদেশের",
            "(",
            "বিজিবি",
            ")",
            "সঙ্গে",
            "রাজশাহীর",
            "চরখানপুর",
            "সীমান্ত",
            "পাহারা",
            "দিচ্ছেন",
            "গ্রামবাসী",
            "।",
            "সীমান্তে",
            "নজরদারি",
            "জোরদার",
            "করার",
            "জন্য",
            "চরখানপুর",
            "গ্রামের",
            "প্রায়",
            "আড়াই",
            "শ",
            "বাসিন্দা",
            "রাত",
            "জেগে",
            "পালাক্রমে",
            "এই",
            "কাজ",
            "করছেন",
            "গত",
            "২৮",
            "নভেম্বর",
            "থেকে",
            "।",
        ]
        self.assertEqual(tokens, gt_tokens)

    def test_nltk_word_tokenizer_with_dot_in_bangla_text(self):
        text = "মো. রহিম বাজারে গিয়েছেন।"
        tokens = self.nltk_tokenizer.word_tokenize(text)
        gt_tokens = ['মো.', 'রহিম', 'বাজারে', 'গিয়েছেন', '।']
        self.assertEqual(tokens, gt_tokens)

    def test_nltk_sentence_tokenizer(self):
        text = "আমি ভাত খাই। সে বাজারে যায়। কী বলছো এসব?"
        gt_tokens = ["আমি ভাত খাই।", "সে বাজারে যায়।", "কী বলছো এসব?"]
        sentence_tokens = self.nltk_tokenizer.sentence_tokenize(text)
        self.assertEqual(sentence_tokens, gt_tokens)

if __name__ == "__main__":
    unittest.main()