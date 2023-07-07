import unittest
from bnlp import BasicTokenizer

class TestBasicTokenizer(unittest.TestCase):
    def setUp(self):
        self.basic_tokenizer = BasicTokenizer()
    
    def test_basic_tokenizer_with_sample_bangla_text(self):
        text = "আমি ভাত খাই।"
        tokens = self.basic_tokenizer(text)
        self.assertEqual(tokens, ["আমি", "ভাত", "খাই", "।"])

    def test_basic_tokenizer_with_long_bangla_text(self):
        text = """
        ভারত থেকে অনুপ্রবেশ ঠেকাতে বর্ডার গার্ড বাংলাদেশের (বিজিবি)
         সঙ্গে রাজশাহীর চরখানপুর সীমান্ত পাহারা দিচ্ছেন গ্রামবাসী।
         সীমান্তে নজরদারি জোরদার করার জন্য     চরখানপুর গ্রামের প্রায় আড়াই শ
         বাসিন্দা রাত জেগে পালাক্রমে এই কাজ করছেন গত ২৮ নভেম্বর থেকে।
        """
        tokens = self.basic_tokenizer(text)
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

    def test_basic_tokenizer_with_dot_in_bangla_text(self):
        text = "মো. রহিম বাজারে গিয়েছেন।"
        tokens = self.basic_tokenizer(text)
        gt_tokens = ['মো.', 'রহিম', 'বাজারে', 'গিয়েছেন', '।']
        self.assertEqual(tokens, gt_tokens)

if __name__ == "__main__":
    unittest.main()