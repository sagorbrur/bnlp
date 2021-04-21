import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("punkt not found. downloading...")
    nltk.download('punkt')


class NLTKTokenizer:
    def word_tokenize(self, text):
        tokens = nltk.word_tokenize(text)
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
    
    def sentence_tokenize(self, text):
        text = text.replace(".", "<dummy_bangla_token>") # to deal with abbreviations
        text = text.replace("।", ".")
        tokens = nltk.tokenize.sent_tokenize(text)
        new_tokens = []
        for token in tokens:
            token = token.replace(".","।") # do operation in reverse order
            token = token.replace("<dummy_bangla_token>",".")
            new_tokens.append(token)
        return new_tokens




