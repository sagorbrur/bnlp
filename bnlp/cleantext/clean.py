"""
This cleantext scripts functions solely depends on clean-text library.
Most of the functions are copied from clean-text.
"""
import re
from bnlp.cleantext import constants
from bnlp.corpus.corpus import BengaliCorpus as corpus

from ftfy import fix_text
from unicodedata import category, normalize
from emoji import UNICODE_EMOJI, demojize, emojize

def fix_bad_unicode(text, normalization="NFC"):
    return fix_text(text, normalization=normalization)

def fix_strange_quotes(text):
    """
    Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
    """
    text = constants.SINGLE_QUOTE_REGEX.sub("'", text)
    text = constants.DOUBLE_QUOTE_REGEX.sub('"', text)
    return text

def replace_urls(text, replace_with=""):
    """
    Replace all URLs in ``text`` str with ``replace_with`` str.
    """
    return constants.URL_REGEX.sub(replace_with, text)

def replace_emails(text, replace_with=""):
    """
    Replace all emails in ``text`` str with ``replace_with`` str.
    """
    return constants.EMAIL_REGEX.sub(replace_with, text)

def remove_substrings(text, to_replace, replace_with=""):
    """
    Remove (or replace) substrings from a text.
    Args:
        text (str): raw text to preprocess
        to_replace (iterable or str): substrings to remove/replace
        replace_with (str): defaults to an empty string but
            you replace substrings with a token.
    """
    if isinstance(to_replace, str):
        to_replace = [to_replace]

    result = text
    for x in to_replace:
        result = result.replace(x, replace_with)
    return result

def remove_emoji(text):
    return remove_substrings(text, UNICODE_EMOJI["en"])

def remove_number_or_digit(text, replace_with=""):
    return re.sub(constants.BANGLA_DIGIT_REGEX, replace_with, text)

def remove_punctuations(text, replace_with=""):
    for punc in corpus.punctuations:
        text = text.replace(punc, replace_with)
    
    return text

class CleanText(object):
    def __init__(
        self,
        fix_unicode=True,
        unicode_norm=True,
        unicode_norm_form="NFKC",
        remove_url=False,
        remove_email=False,
        remove_number=False,
        remove_digits=False,
        remove_emoji=False,
        remove_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_number="<NUMBER>",
        replace_with_digit="<DIGIT>",
        replace_with_punct = "<PUNC>"
        ):
        self.fix_unicode = fix_unicode
        self.unicode_norm = unicode_norm
        self.unicode_norm_form = unicode_norm_form
        self.remove_url = remove_url
        self.remove_email = remove_email
        self.remove_number = remove_number
        self.remove_digits = remove_digits
        self.remove_emoji = remove_emoji
        self.remove_punct = remove_punct
        
        self.replace_with_url = replace_with_url
        self.replace_with_email = replace_with_email
        self.replace_with_number = replace_with_number
        self.replace_with_digit = replace_with_digit
        self.replace_with_punct = replace_with_punct

    def __call__(self, text: str) -> str:
        if text is None:
            text = ""
        text = str(text)
        text = fix_strange_quotes(text)

        if self.fix_unicode:
            text = fix_bad_unicode(text)
        if self.unicode_norm:
            text = normalize(self.unicode_norm_form, text)
        if self.remove_punct:
            text = remove_punctuations(text, replace_with=self.replace_with_punct)
        if self.remove_url:
            text = replace_urls(text, replace_with=self.replace_with_url)
        if self.remove_email:
            text = replace_emails(text, replace_with=self.replace_with_email)
        if self.remove_emoji:
            text = remove_emoji(text)
        if self.remove_digits:
            text = remove_number_or_digit(text, replace_with=self.replace_with_digit)
        if self.remove_number:
            text = remove_number_or_digit(text, replace_with=self.replace_with_number)

        return text

