from bnlp.tokenizer.basic import BasicTokenizer


def remove_stopwords(text, stopwords):
  """
  This function remove stopwords from text
  parameters:
    text: str
    stopwords: list
  return: tokens of word without stopwords

  """
  tokenizer = BasicTokenizer()
  words = tokenizer.tokenize(text)
  filtered_words = [w for w in words if not w in stopwords]
  return filtered_words




