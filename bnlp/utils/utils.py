import pickle
from sklearn_crfsuite import CRF
from nltk.tag.util import untag

def features(sentence, index):
    """sentence: [w1, w2, ...], index: the index of the word"""
    return {
        "word": sentence[index],
        "is_first": index == 0,
        "is_last": index == len(sentence) - 1,
        "is_capitalized": sentence[index][0].upper() == sentence[index][0],
        "is_all_caps": sentence[index].upper() == sentence[index],
        "is_all_lower": sentence[index].lower() == sentence[index],
        "prefix-1": sentence[index][0],
        "prefix-2": sentence[index][:2],
        "prefix-3": sentence[index][:3],
        "suffix-1": sentence[index][-1],
        "suffix-2": sentence[index][-2:],
        "suffix-3": sentence[index][-3:],
        "prev_word": "" if index == 0 else sentence[index - 1],
        "next_word": "" if index == len(sentence) - 1 else sentence[index + 1],
        "has_hyphen": "-" in sentence[index],
        "is_numeric": sentence[index].isdigit(),
        "capitals_inside": sentence[index][1:].lower() != sentence[index][1:],
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

def load_pickle_model(model_path: str) -> CRF:
    with open(model_path, "rb") as pkl_model:
        model = pickle.load(pkl_model)
        
    return model
