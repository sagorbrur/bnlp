import fasttext


class Bengali_Fasttext(object):
    def __init__(self, is_train=False):
        self.is_train = is_train

    def train_fasttext(self, data, model_name):
        if self.is_train:
            model = fasttext.train_unsupervised(data, model='skipgram', minCount=1)
            model.save_model(model_name)

    def generate_word_vector(self, model_path, word):
        model = fasttext.load_model(model_path)
        word_vector = model[word]

        return word_vector