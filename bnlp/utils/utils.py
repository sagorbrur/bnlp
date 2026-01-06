"""Utility functions for BNLP."""

import logging
import pickle
from typing import List, Dict, Tuple, Any

from sklearn_crfsuite import CRF
from nltk.tag.util import untag

logger = logging.getLogger(__name__)


def features(sentence: List[str], index: int) -> Dict[str, Any]:
    """Extract features for a word in a sentence.

    Args:
        sentence: List of words [w1, w2, ...]
        index: Index of the word to extract features for

    Returns:
        Dictionary of features for the word
    """
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


def transform_to_dataset(
    tagged_sentences: List[List[Tuple[str, str]]]
) -> Tuple[List[List[Dict[str, Any]]], List[List[str]]]:
    """Transform tagged sentences to CRF dataset format.

    Args:
        tagged_sentences: List of tagged sentences, each containing (word, tag) tuples

    Returns:
        Tuple of (features_list, tags_list)
    """
    X, y = [], []

    for tagged in tagged_sentences:
        try:
            X.append([features(untag(tagged), index) for index in range(len(tagged))])
            y.append([tag for _, tag in tagged])
        except (IndexError, ValueError) as e:
            logger.warning(f"Error processing tagged sentence: {e}")
            continue

    return X, y


def load_pickle_model(model_path: str) -> CRF:
    """Load a pickled CRF model from file.

    Args:
        model_path: Path to the pickle file

    Returns:
        Loaded CRF model
    """
    with open(model_path, "rb") as pkl_model:
        model = pickle.load(pkl_model)

    return model
