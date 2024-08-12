import nltk
from typing import List

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("punkt not found. downloading...")
    nltk.download("punkt_tab")

DUMMYTOKEN = "XTEMPTOKEN"

DUMMYTOKEN = "XTEMPDOT"

class NLTKTokenizer:
    def word_tokenize(self, text: str) -> List[str]:
        text = text.replace(".", DUMMYTOKEN)  # to deal with abbreviations
        text = text.replace("।", ".")
        tokens = nltk.word_tokenize(text)
        new_tokens = []
        for token in tokens:
            token = token.replace(".", "।")  # do operation in reverse order
            token = token.replace(DUMMYTOKEN, ".")
            new_tokens.append(token)

        return new_tokens

    def sentence_tokenize(self, text: str) -> List[str]:
        text = text.replace(".", DUMMYTOKEN)  # to deal with abbreviations
        text = text.replace("।", ".")
        tokens = nltk.tokenize.sent_tokenize(text)
        new_tokens = []
        for token in tokens:
            token = token.replace(".", "।")  # do operation in reverse order
            token = token.replace(DUMMYTOKEN, ".")
            new_tokens.append(token)
        return new_tokens
