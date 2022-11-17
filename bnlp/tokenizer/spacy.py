try:
    import spacy
    from spacy.lang.bn import Bengali
except ImportError:
    print("spacy not installed. install it by pip install bnlp_toolkit[spacy]")


def spacy_tokenizer(text):
    nlp = Bengali()
    tokenizer = nlp.tokenizer
    tokens = list(tokenizer(text))
    return tokens