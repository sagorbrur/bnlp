import nltk


class NLTK_Tokenizer(object):
    def __init__(self, text):
        self.text = text
        """
        :text: (str) input text for tokenization

        """

    def word_tokenize(self):
        tokens = nltk.word_tokenize(self.text)
        return tokens
    
    def sentence_tokenize(self):
        text = self.text.replace("।", ".")
        tokens = nltk.tokenize.sent_tokenize(text)
        new_tokens = []
        for a in tokens:
            if a[-1] == ".":
                a = a[:-2] + a[-2:].replace(".","।")
            new_tokens.append(a)
        return tokens





