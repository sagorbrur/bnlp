import numpy as np
import scipy
from scipy import spatial
#print(np.__version__) #1.17.4
#print(scipy.__version__) #1.3.3



class BN_Glove(object):
  
  def word2vec(self, glove_path, test_word):
      embeddings_dict = {}
      with open(glove_path, 'r') as f:
          for line in f:
              values = line.split()
              word = values[0]
              vector = np.asarray(values[1:], "float32")
              embeddings_dict[word] = vector
      result_vec = embeddings_dict[test_word]
      return result_vec

  def closest_word(self,glove_path, test_word):

    def find_closest_embeddings(embedding):
      return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
    
    
    embeddings_dict = {}
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    result = find_closest_embeddings(embeddings_dict[test_word])[:10]
    return result


if __name__=="__main__":
    glove_path = "bn_glove.39M.100d.txt"
    word = "গ্রাম"
    bng = BN_Glove()
    res = bng.closest_word(glove_path, word)
    print(res)
    vec = bng.word2vec(glove_path, word)
    print(vec)
