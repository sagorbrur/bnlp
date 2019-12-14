#! /usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from bnlp.sentencepiece_tokenizer import SP_Tokenizer
from bnlp.nltk_tokenizer import NLTK_Tokenizer
from bnlp.basic_tokenizer import BasicTokenizer
from bnlp.bengali_word2vec import Bengali_Word2Vec
from bnlp.bengali_fasttext import Bengali_Fasttext

class TestBNLP(unittest.TestCase):

    def test_BT(self):
        bt = BasicTokenizer(False)
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
        nl = NLTK_Tokenizer(text1)
        out_word_tokens = nl.word_tokenize()
        self.assertEqual(out_word_tokens, gt_word_tokens)

        nl2 = NLTK_Tokenizer(text2)
        out_sen_tokens = nl2.sentence_tokenize()
        self.assertEqual(out_sen_tokens, gt_sen_tokens)



    

if __name__ == '__main__':
    unittest.main()
