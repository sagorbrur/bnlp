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
        for token in tokens:
            if token[-1] == "।" and len(token)>1:
                token_1 = token[:-1]
                token_2 = token[-1]
                new_tokens.append(token_1)
                new_tokens.append(token_2)
            else:
              new_tokens.append(token)
        return new_tokens
    
    def sentence_tokenize(self):
        text = self.text.replace("।", ".")
        tokens = nltk.tokenize.sent_tokenize(text)
        new_tokens = []
        for token in tokens:
            if token[-1] == ".":
                token = token[:-2] + token[-2:].replace(".","।")
            new_tokens.append(token)
        return new_tokens





