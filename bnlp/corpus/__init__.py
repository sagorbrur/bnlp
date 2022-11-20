# BNLP Corpus Reader
# Author: Sagor Sarker

"""
The module in th is package will provide you function that can used to read corpus.

Available Corpus:
- Bengali Stopwords
    Collected from: https://github.com/stopwords-iso/stopwords-bn

- Bengali letters and vowel mark
    collected from https://github.com/MinhasKamal/BengaliDictionary/blob/master/BengaliCharacterCombinations.txt

"""
import json

# return list of bengali stopwords
stopwords = json.load(open("./stopwords.json", "r"))

# return list of bengali punctuation
punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~।ঃ'

# return bangla letters
letters = 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁ'

# return bangla digits
digits = '০১২৩৪৫৬৭৮৯'

# bengali vower mark
vower_mark = 'া ি ী ু ৃ ে ৈ ো ৌ'



