import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("punkt not found. downloading...")
    nltk.download('punkt')


class NLTK_Tokenizer(object):
    def __init__(self, text):
        self.text = text
        """
        :text: (str) input text for tokenization

        """

    def word_tokenize(self):
        tokens = nltk.word_tokenize(self.text)
        new_tokens = []
        for a in tokens:
            if a[-1] == "।":
                a1 = a[:-1]
                a2 = a[-1]
                new_tokens.append(a1)
                new_tokens.append(a2)
            else:
              new_tokens.append(a)
        return new_tokens
    
    def sentence_tokenize(self):
        text = self.text.replace("।", ".")
        tokens = nltk.tokenize.sent_tokenize(text)
        new_tokens = []
        for a in tokens:
            if a[-1] == ".":
                a = a[:-2] + a[-2:].replace(".","।")
            new_tokens.append(a)
        return new_tokens





