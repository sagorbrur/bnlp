import os
import sentencepiece as bsp
import urllib.request
import sys
import time

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
    
    
def check_model(model_path):
    pretrained_model_url = "https://github.com/iriad11/bnlp/raw/master/model/bn_spm.model"
    pretrained_vocab_url = "https://github.com/iriad11/bnlp/raw/master/model/bn_spm.vocab"
    if model_path[0] == "/":
        model_path = model_path[:2].replace("/","./") + model_path[2:]

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        print('Downloading pretrained model...')
        urllib.request.urlretrieve(pretrained_model_url, model_path, reporthook)
        print('\nDownload completed.\n')
        
    mod = os.path.dirname(model_path)
    vocab_path = "." + mod + "/bn_spm.vocab"
    
    if not os.path.exists(vocab_path):
        os.makedirs(os.path.dirname(vocab_path),exist_ok=True)
        print('Downloading pretrained vocab...')
        urllib.request.urlretrieve(pretrained_vocab_url, vocab_path, reporthook)
        print('\nDownload completed.\n')
        
    return model_path
    

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
        model_path = check_model(model_path)
        model.Load(model_path)
        tokens = model.EncodeAsPieces(text)

        return tokens


