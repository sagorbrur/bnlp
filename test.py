import unittest
from bnlp import POS
from bnlp import NER
from bnlp import NLTKTokenizer
from bnlp import BasicTokenizer
from bnlp import SentencepieceTokenizer
from bnlp import BengaliWord2Vec
from bnlp import BengaliFasttext

class TestBNLP(unittest.TestCase):

    def test_BT(self):
        bt = BasicTokenizer()
        text = "আমি ভাত খাই।"
        tokens = bt.tokenize(text)
        self.assertEqual(tokens, ["আমি", "ভাত", "খাই", "।"])

        text1 = """
        ভারত থেকে অনুপ্রবেশ ঠেকাতে বর্ডার গার্ড বাংলাদেশের (বিজিবি)
         সঙ্গে রাজশাহীর চরখানপুর সীমান্ত পাহারা দিচ্ছেন গ্রামবাসী।
         সীমান্তে নজরদারি জোরদার করার জন্য     চরখানপুর গ্রামের প্রায় আড়াই শ 
         বাসিন্দা রাত জেগে পালাক্রমে এই কাজ করছেন গত ২৮ নভেম্বর থেকে।
        """
        tokens1 = bt.tokenize(text1)
        output_1 = ["ভারত", "থেকে", "অনুপ্রবেশ", "ঠেকাতে", "বর্ডার", "গার্ড", "বাংলাদেশের", "(", "বিজিবি", ")",
         "সঙ্গে", "রাজশাহীর", "চরখানপুর", "সীমান্ত", "পাহারা", "দিচ্ছেন", "গ্রামবাসী", "।",
         "সীমান্তে", "নজরদারি", "জোরদার", "করার", "জন্য", "চরখানপুর", "গ্রামের", "প্রায়", "আড়াই", "শ", 
         "বাসিন্দা", "রাত", "জেগে", "পালাক্রমে", "এই", "কাজ", "করছেন", "গত", "২৮", "নভেম্বর", "থেকে", "।"]
        self.assertEqual(tokens1, output_1)

    def test_NLTK(self):
        text1 = """
        ভারত থেকে অনুপ্রবেশ ঠেকাতে বর্ডার গার্ড বাংলাদেশের (বিজিবি)
         সঙ্গে রাজশাহীর চরখানপুর সীমান্ত পাহারা দিচ্ছেন গ্রামবাসী।
         সীমান্তে নজরদারি জোরদার করার জন্য     চরখানপুর গ্রামের প্রায় আড়াই শ 
         বাসিন্দা রাত জেগে পালাক্রমে এই কাজ করছেন গত ২৮ নভেম্বর থেকে।
        """
        text2 = "আমি ভাত খাই। সে বাজারে যায়। কী বলছো এসব?"

        gt_word_tokens = ["ভারত", "থেকে", "অনুপ্রবেশ", "ঠেকাতে", "বর্ডার", "গার্ড", "বাংলাদেশের", "(", "বিজিবি", ")",
         "সঙ্গে", "রাজশাহীর", "চরখানপুর", "সীমান্ত", "পাহারা", "দিচ্ছেন", "গ্রামবাসী", "।",
         "সীমান্তে", "নজরদারি", "জোরদার", "করার", "জন্য", "চরখানপুর", "গ্রামের", "প্রায়", "আড়াই", "শ", 
         "বাসিন্দা", "রাত", "জেগে", "পালাক্রমে", "এই", "কাজ", "করছেন", "গত", "২৮", "নভেম্বর", "থেকে", "।"]

        gt_sen_tokens = ["আমি ভাত খাই।", "সে বাজারে যায়।", "কী বলছো এসব?"]
        nl = NLTKTokenizer()
        out_word_tokens = nl.word_tokenize(text1)
        self.assertEqual(out_word_tokens, gt_word_tokens)

        nl2 = NLTKTokenizer()
        out_sen_tokens = nl2.sentence_tokenize(text2)
        self.assertEqual(out_sen_tokens, gt_sen_tokens)
    
    def test_POS(self):
        bn_pos = POS()
        model_path = "model/bn_pos.pkl"
        text = "আমি ভাত খাই।"
        res = bn_pos.tag(model_path, text)
        self.assertEqual(res, [('আমি', 'PPR'), ('ভাত', 'NC'), ('খাই', 'VM'), ('।', 'PU')])    

    def test_NER(self):
        bn_ner = NER()
        model_path = "model/bn_ner.pkl"
        text = "সে ঢাকায় থাকে।"
        res = bn_ner.tag(model_path, text)
        self.assertEqual(res, [('সে', 'O'), ('ঢাকায়', 'S-LOC'), ('থাকে', 'O')])    

if __name__ == '__main__':
    unittest.main()
