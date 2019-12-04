
from bnlp.sentencepiece_tokenizer import SP_Tokenizer
from bnlp.nltk_tokenizer import NLTK_Tokenizer
from bnlp.bengali_word2vec import Bengali_Word2Vec
from bnlp.basic_tokenizer import BasicTokenizer




if __name__ == "__main__":

    # # nltk testing
    # text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"
    # nlt = NLTK_Tokenizer(text)
    # nl_tokens = nlt.word_tokenize()
    # nl_sen = nlt.sentence_tokenize()
    # print(nl_tokens)
    # print(nl_sen)

    # # sentence testing
    # spp = SP_Tokenizer()
    # path = "./model/bn_spm.model"
    # token = spp.tokenize(path, "আমি বাংলায়।")
    # print(token)
    # tr = SP_Tokenizer(is_train=True)
    # data = "test.txt"
    # tr.train_bsp(data, "test", 5) 

    # word2vec testing
    # bwv = Bengali_Word2Vec(is_train=True)
    # data_file = "test.txt"
    # model_name = "test_model.model"
    # vector_name = "test_vector.vector"
    # bwv.train_word2vec(data_file, model_name, vector_name)

    # bwv = Bengali_Word2Vec()
    # model_path = "bnlp/embedding/wiki.bn.text.model"
    # word = 'আমার'
    # similar = bwv.most_similar(model_path, word)
    # print(similar)
    # vector = bwv.generate_word_vector(model_path, word)
    # print(vector.shape)
    # print(vector)
    # basic_t = BasicTokenizer(False)
    # raw_text = "আমি বাংলায় গান গাই।"
    # tokens = basic_t.tokenize(raw_text)
    # print(tokens)


    
