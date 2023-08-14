"""
Basic Tokenization
Basic tokenization tokenize sentence using white spaces, punctuation mark
Code shamelessly copied from BERT tokenization
To check Original Code: https://github.com/google-research/bert/blob/master/tokenization.py
"""
import six
import unicodedata
from typing import List

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def whitespace_tokenize(text: str) -> List[str]:
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


DUMMYTOKEN = 'XTEMPDOT'

class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        # handle (.) in bangla text
        text = text.replace('.', DUMMYTOKEN)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # get (.) back in output tokens
        output_tokens = [token.replace(DUMMYTOKEN, '.') for token in output_tokens]
        return output_tokens
        
    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]
