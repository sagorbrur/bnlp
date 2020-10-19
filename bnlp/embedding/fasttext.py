try:
    import fasttext
except ImportError:
    print("fasttext not installed. Install it by pip install fasttext")

class BengaliFasttext:
    
    def train(self, data, model_name, epoch):
        model = fasttext.train_unsupervised(data, model='skipgram', minCount=1, epoch=epoch)
        model.save_model(model_name)

    def generate_word_vector(self, model_path, word):
        model = fasttext.load_model(model_path)
        word_vector = model[word]

        return word_vector
    
    
