
from bnlp.sentencepiece_tokenizer import SP_Tokenizer
from bnlp.nltk_tokenizer import NLTK_Tokenizer
from bnlp.bengali_word2vec import Bengali_Word2Vec
from bnlp.basic_tokenizer import BasicTokenizer
from bnlp.glove_wordvector import BN_Glove
from bnlp.bengali_pos import BN_CRF_POS




if __name__ == "__main__":

    # # nltk testing
    # text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"
    # nlt = NLTK_Tokenizer()
    # nl_tokens = nlt.word_tokenize(text)
    # nl_sen = nlt.sentence_tokenize(text)
    # print(nl_tokens)
    # print(nl_sen)

    # # sentence testing
    # spp = SP_Tokenizer()
    # path = "./model/bn_spm.model"
    # token = spp.tokenize(path, "আমি বাংলায়।")
    # print(token)
    # tr = SP_Tokenizer()
    # data = "test.txt"
    # tr.train_bsp(data, "test", 41) 

    # word2vec testing
    # bwv = Bengali_Word2Vec()
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
    # basic_t = BasicTokenizer()
    # raw_text = "আমি বাংলায় গান গাই।"
    # tokens = basic_t.tokenize(raw_text)
    # print(tokens)
    
#     glove_path = "bn_glove.39M.100d.txt"
#     word = "গ্রাম"
#     bng = BN_Glove()
#     res = bng.closest_word(glove_path, word)
#     print(res)
#     vec = bng.word2vec(glove_path, word)
#     print(vec)
#     spm = SP_Tokenizer()
#     model_path = "model/bn_spm.model"
#     tokens = spm.tokenize(model_path, "আমি ভাত খাই।")
#     print(tokens)
#     ids = spm.text2id(model_path, "আমি ভাত খাই।")
#     print(ids)
#     text = spm.id2text(model_path, ids)
#     print(text)
    # bn_pos = BN_CRF_POS()
    # model_path = "bn_pos_model.pkl"
    # text = "আমি ভাত খাই।"
    # res = bn_pos.pos_tag(model_path, text)
    # print(res)

    
