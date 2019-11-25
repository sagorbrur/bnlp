import os
import sentencepiece as bsp


class SP_Tokenizer(object):
    def __init__(self, is_train=False):
        self.is_train = is_train
        """
        :is_train: boolean value to choose training option

        """


    def train_bsp(self, data, model_prefix, vocab_size):
        """
        :data: (str) data path with extension
        :model_prefix: (str) model name prefix
        :vocab_size: (int) size of train vocabulary

        """
        if self.is_train:
            train_args = "--model_prefix="+model_prefix+" --input="+data+" --vocab_size="+str(vocab_size)
            # print(train_args)
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



