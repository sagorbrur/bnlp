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

from typing import List
from bnlp.tokenizer.basic import BasicTokenizer
from ._stopwords import bengali_stopwords

class BengaliCorpus:
    punctuations: str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~।ঃ"
    letters: str = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁ"
    digits: str = "০১২৩৪৫৬৭৮৯"
    vowels: str = "া ি ী ু ৃ ে ৈ ো ৌ"
    stopwords: List[str] = bengali_stopwords

