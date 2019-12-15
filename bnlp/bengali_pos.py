"""
tool: We used sklearn crf_suite for bengali pos tagging
https://sklearn-crfsuite.readthedocs.io/en/latest/

Contributor: 
* Sagor Sarker
* Md. Ibrahim 

"""

import pickle
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from nltk.tag.util import untag
from bnlp.basic_tokenizer import BasicTokenizer

def features(sentence, index):
        """ sentence: [w1, w2, ...], index: the index of the word """
        return {
            'word': sentence[index],
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            'prefix-1': sentence[index][0],
            'prefix-2': sentence[index][:2],
            'prefix-3': sentence[index][:3],
            'suffix-1': sentence[index][-1],
            'suffix-2': sentence[index][-2:],
            'suffix-3': sentence[index][-3:],
            'prev_word': '' if index == 0 else sentence[index - 1],
            'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }

def transform_to_dataset(tagged_sentences):
    X, y = [], []
     
    for tagged in tagged_sentences:
        try:
            X.append([features(untag(tagged), index) for index in range(len(tagged))])
            y.append([tag for _, tag in tagged])
        except Exception as e:
            print(e)
 
    return X, y

class BN_CRF_POS(object):
    def __init__(self, is_training=False):
        self.is_training = is_training

    def pos_tag(self, model_path, text):
        model = pickle.load(open(model_path, 'rb'))
        basic_t = BasicTokenizer(False)
        tokens = basic_t.tokenize(text)
        sentence_features = [features(tokens, index) for index in range(len(tokens))]
        result = list(zip(tokens, model.predict([sentence_features])[0]))
        return result

    def training(self, model_name, tagged_sentences):
        # Split the dataset for training and testing
        cutoff = int(.75 * len(tagged_sentences))
        training_sentences = tagged_sentences[:cutoff]
        test_sentences = tagged_sentences[cutoff:]

        X_train, y_train = transform_to_dataset(training_sentences)
        X_test, y_test = transform_to_dataset(test_sentences)
        print(len(X_train))
        print(len(X_test))


        print("Training Start........")
        model = CRF()
        model.fit(X_train, y_train)
        print("Training Finished!")
        
        print("Evaluating with Test Data...")
        y_pred = model.predict(X_test)
        print("Accuracy is: ")
        print(metrics.flat_accuracy_score(y_test, y_pred))
        
        pickle.dump(model, open(model_name, 'wb'))
        print("Model Saved!")
