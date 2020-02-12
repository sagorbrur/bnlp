import os
import sentencepiece as bsp


class SP_Tokenizer(object):
    
    def train_bsp(self, data, model_prefix, vocab_size):
        """
        :data: (str) data path with extension
        :model_prefix: (str) model name prefix
        :vocab_size: (int) size of train vocabulary

        """
        train_args = "--model_prefix="+model_prefix+" --input="+data+" --vocab_size="+str(vocab_size)
        bsp.SentencePieceTrainer.train(train_args)
        print("%s.model and %s.vocab is saved on your current directory"%(model_prefix, model_prefix))

    def tokenize(self, model_path, text):
        """
        :model_path: (str) path of the model with extension
        :text: (str) input text for tokenization

        """
        model = bsp.SentencePieceProcessor()
        model.Load(model_path)
        tokens = model.EncodeAsPieces(text)

        return tokens
    
    def text2id(self, model_path, text):
        model = bsp.SentencePieceProcessor()
        model.Load(model_path)
        ids = model.EncodeAsIds(text)
        return ids
    def id2text(self, model_path, ids):
        model = bsp.SentencePieceProcessor()
        model.Load(model_path)
        text = model.DecodeIds(ids)
        return text




