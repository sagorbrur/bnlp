import unittest
from bnlp.sentencepiece_tokenizer import SP_Tokenizer
from bnlp.nltk_tokenizer import NLTK_Tokenizer
from bnlp.basic_tokenizer import BasicTokenizer
from bnlp.bengali_word2vec import Bengali_Word2Vec
from bnlp.bengali_fasttext import Bengali_Fasttext
from bnlp.pos import POS
from bnlp.ner import NER

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
        nl = NLTK_Tokenizer()
        out_word_tokens = nl.word_tokenize(text1)
        self.assertEqual(out_word_tokens, gt_word_tokens)

        nl2 = NLTK_Tokenizer()
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
