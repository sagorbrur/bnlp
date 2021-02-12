from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.corpus import punctuations


def remove_stopwords(text, stopwords):
  """This function remove stopwords from Bengali text

  Args:
      text (str): input text
      stopwords (list): stopword list

  Returns:
      list: list of words without stopwords
  """
  tokenizer = BasicTokenizer()
  words = tokenizer.tokenize(text)
  filtered_words = [w for w in words if not w in stopwords]
  return filtered_words

def remove_foreign_words(text):
  """This function removes foreign words from Bengali text
     formula copied from this stackoverflow answers
     https://stackoverflow.com/questions/64433299/how-can-i-remove-foreign-word-from-bengali-text-in-python

  Args:
      text (str): input text

  Returns:
      str: output text
  """
  text = "".join(i for i in text if i in punctuations or 2432 <= ord(i) <= 2559 or ord(i)== 32)
  tokens = text.split()
  # removing extra punctions from tokens
  tokens = [token for token in tokens if token not in punctuations]
  result_text = " ".join(tokens)
  return result_text


