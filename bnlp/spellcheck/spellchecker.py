"""
Bengali Spell Checker

This module provides spell checking and correction for Bengali text
using the SymSpell algorithm for fast approximate string matching.

The SymSpell algorithm is 1000x faster than traditional spell checkers
while maintaining high accuracy.

References:
- SymSpell: https://github.com/wolfgarbe/SymSpell
- symspellpy: https://github.com/mammothb/symspellpy
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from pathlib import Path

try:
    from symspellpy import SymSpell, Verbosity
except ImportError:
    raise ImportError(
        "symspellpy is required for spell checking. "
        "Install it with: pip install symspellpy"
    )

from bnlp.tokenizer.basic import BasicTokenizer
from bnlp.corpus.corpus import BengaliCorpus


@dataclass
class SpellingError:
    """Represents a spelling error in text.

    Attributes:
        word: The misspelled word
        position: Starting position in the original text
        suggestions: List of suggested corrections with scores
        best_correction: The most likely correction
    """
    word: str
    position: int
    suggestions: List[Tuple[str, float]] = field(default_factory=list)
    best_correction: Optional[str] = None

    def __repr__(self) -> str:
        return f"SpellingError(word='{self.word}', best_correction='{self.best_correction}')"


class BengaliSpellChecker:
    """Bengali spell checker using SymSpell algorithm.

    This class provides fast and accurate spell checking for Bengali text.
    It uses the SymSpell algorithm which is 1000x faster than traditional
    spell checkers based on Levenshtein distance.

    Example:
        >>> from bnlp.spellcheck import BengaliSpellChecker
        >>>
        >>> checker = BengaliSpellChecker()
        >>>
        >>> # Check for errors
        >>> errors = checker.check("আমি বাংলায় গন গাই।")
        >>> for error in errors:
        ...     print(f"{error.word} -> {error.best_correction}")
        >>>
        >>> # Correct text
        >>> corrected = checker.correct("আমি বাংলায় গন গাই।")
        >>> print(corrected)
        >>>
        >>> # Get suggestions for a word
        >>> suggestions = checker.suggestions("গন")
        >>> print(suggestions)

    Attributes:
        max_edit_distance: Maximum edit distance for corrections (default: 2)
        prefix_length: Length of word prefix for indexing (default: 7)
    """

    # Common Bengali word frequency dictionary
    # This is a curated list of common Bengali words with frequencies
    _DEFAULT_WORDS: Dict[str, int] = {
        # Pronouns
        "আমি": 100000, "আমরা": 90000, "তুমি": 85000, "তোমরা": 80000,
        "আপনি": 95000, "আপনারা": 75000, "সে": 98000, "তারা": 88000,
        "তিনি": 92000, "এটা": 70000, "এটি": 72000, "ওটা": 60000,
        "যে": 96000, "যা": 94000, "কে": 93000, "কি": 97000,
        "কী": 91000, "কেউ": 65000, "কিছু": 78000, "সব": 82000,
        "নিজে": 55000, "নিজের": 56000,

        # Verbs (common forms)
        "করা": 99000, "করি": 95000, "করে": 96000, "করো": 80000,
        "করেন": 85000, "করছি": 75000, "করছে": 76000, "করব": 70000,
        "করবে": 71000, "করেছি": 72000, "করেছে": 73000, "করতে": 74000,
        "হওয়া": 98000, "হয়": 97000, "হই": 80000, "হও": 75000,
        "হন": 82000, "হচ্ছে": 78000, "হবে": 85000, "হয়েছে": 83000,
        "হতে": 86000, "হলে": 84000, "হলো": 81000,
        "যাওয়া": 92000, "যাই": 88000, "যায়": 90000, "যাও": 82000,
        "যান": 83000, "যাচ্ছি": 75000, "যাচ্ছে": 76000, "যাব": 78000,
        "যাবে": 79000, "গেছি": 70000, "গেছে": 71000, "গেল": 72000,
        "আসা": 91000, "আসি": 85000, "আসে": 87000, "আসো": 78000,
        "আসেন": 80000, "আসছি": 72000, "আসছে": 73000, "আসব": 74000,
        "আসবে": 75000, "এসেছি": 68000, "এসেছে": 69000, "এলো": 70000,
        "দেখা": 89000, "দেখি": 84000, "দেখে": 86000, "দেখো": 77000,
        "দেখেন": 79000, "দেখছি": 71000, "দেখছে": 72000, "দেখব": 73000,
        "দেখবে": 74000, "দেখেছি": 67000, "দেখেছে": 68000, "দেখলে": 69000,
        "বলা": 90000, "বলি": 85000, "বলে": 88000, "বলো": 78000,
        "বলেন": 80000, "বলছি": 72000, "বলছে": 73000, "বলব": 74000,
        "বলবে": 75000, "বলেছি": 68000, "বলেছে": 69000, "বলল": 70000,
        "খাওয়া": 85000, "খাই": 80000, "খায়": 82000, "খাও": 75000,
        "খান": 76000, "খাচ্ছি": 68000, "খাচ্ছে": 69000, "খাব": 70000,
        "খাবে": 71000, "খেয়েছি": 64000, "খেয়েছে": 65000, "খেলে": 66000,
        "পড়া": 84000, "পড়ি": 79000, "পড়ে": 81000, "পড়ো": 74000,
        "পড়েন": 75000, "পড়ছি": 67000, "পড়ছে": 68000, "পড়ব": 69000,
        "পড়বে": 70000, "পড়েছি": 63000, "পড়েছে": 64000, "পড়লে": 65000,
        "লেখা": 82000, "লিখি": 77000, "লেখে": 79000, "লেখো": 72000,
        "লেখেন": 73000, "লিখছি": 65000, "লিখছে": 66000, "লিখব": 67000,
        "লিখবে": 68000, "লিখেছি": 61000, "লিখেছে": 62000,
        "থাকা": 88000, "থাকি": 83000, "থাকে": 85000, "থাকো": 76000,
        "থাকেন": 78000, "থাকছি": 70000, "থাকছে": 71000, "থাকব": 72000,
        "থাকবে": 73000, "ছিলাম": 66000, "ছিল": 80000, "আছি": 87000,
        "আছে": 92000, "আছো": 75000, "আছেন": 82000,
        "নেওয়া": 80000, "নিই": 75000, "নেয়": 77000, "নাও": 70000,
        "নেন": 72000, "নিচ্ছি": 64000, "নিচ্ছে": 65000, "নেব": 66000,
        "নেবে": 67000, "নিয়েছি": 60000, "নিয়েছে": 61000,
        "দেওয়া": 81000, "দিই": 76000, "দেয়": 78000, "দাও": 71000,
        "দেন": 73000, "দিচ্ছি": 65000, "দিচ্ছে": 66000, "দেব": 67000,
        "দেবে": 68000, "দিয়েছি": 61000, "দিয়েছে": 62000,
        "চাওয়া": 78000, "চাই": 85000, "চায়": 80000, "চাও": 72000,
        "চান": 74000, "চাইছি": 66000, "চাইছে": 67000, "চাইব": 68000,
        "চাইবে": 69000, "চেয়েছি": 62000, "চেয়েছে": 63000,
        "পাওয়া": 83000, "পাই": 78000, "পায়": 80000, "পাও": 73000,
        "পান": 75000, "পাচ্ছি": 67000, "পাচ্ছে": 68000, "পাব": 69000,
        "পাবে": 70000, "পেয়েছি": 63000, "পেয়েছে": 64000, "পেলে": 65000,
        "জানা": 86000, "জানি": 82000, "জানে": 84000, "জানো": 77000,
        "জানেন": 79000, "জানছি": 71000, "জানছে": 72000, "জানব": 73000,
        "জানবে": 74000, "জেনেছি": 67000, "জেনেছে": 68000,
        "ভাবা": 75000, "ভাবি": 70000, "ভাবে": 72000, "ভাবো": 65000,
        "ভাবেন": 67000, "ভাবছি": 59000, "ভাবছে": 60000, "ভাবব": 61000,
        "ভাববে": 62000, "ভেবেছি": 55000, "ভেবেছে": 56000,
        "শোনা": 77000, "শুনি": 72000, "শোনে": 74000, "শোনো": 67000,
        "শোনেন": 69000, "শুনছি": 61000, "শুনছে": 62000, "শুনব": 63000,
        "শুনবে": 64000, "শুনেছি": 57000, "শুনেছে": 58000,
        "বোঝা": 74000, "বুঝি": 69000, "বোঝে": 71000, "বোঝো": 64000,
        "বোঝেন": 66000, "বুঝছি": 58000, "বুঝছে": 59000, "বুঝব": 60000,
        "বুঝবে": 61000, "বুঝেছি": 54000, "বুঝেছে": 55000,
        "রাখা": 79000, "রাখি": 74000, "রাখে": 76000, "রাখো": 69000,
        "রাখেন": 71000, "রাখছি": 63000, "রাখছে": 64000, "রাখব": 65000,
        "রাখবে": 66000, "রেখেছি": 59000, "রেখেছে": 60000,
        "মনে": 88000, "মানা": 72000, "মানি": 67000, "মানে": 85000,

        # Nouns
        "মানুষ": 95000, "লোক": 85000, "জন": 90000, "বছর": 88000,
        "দিন": 92000, "রাত": 80000, "সময়": 87000, "কাজ": 91000,
        "কথা": 89000, "জীবন": 86000, "দেশ": 94000, "বাংলাদেশ": 93000,
        "ভারত": 82000, "পৃথিবী": 78000, "বিশ্ব": 80000, "সংসার": 70000,
        "ঘর": 85000, "বাড়ি": 84000, "রাস্তা": 75000, "পথ": 77000,
        "গ্রাম": 79000, "শহর": 81000, "নগর": 65000, "দেশ": 83000,
        "জল": 76000, "পানি": 82000, "আগুন": 68000, "বাতাস": 67000,
        "মাটি": 72000, "আকাশ": 74000, "সূর্য": 71000, "চাঁদ": 70000,
        "তারা": 69000, "বৃষ্টি": 66000, "মেঘ": 64000,
        "মা": 91000, "বাবা": 90000, "ভাই": 87000, "বোন": 86000,
        "দাদা": 78000, "দিদি": 77000, "ছেলে": 85000, "মেয়ে": 84000,
        "বাচ্চা": 76000, "শিশু": 75000, "পরিবার": 80000, "বন্ধু": 82000,
        "বই": 83000, "পড়া": 79000, "লেখা": 78000, "শিক্ষা": 81000,
        "স্কুল": 80000, "কলেজ": 74000, "বিশ্ববিদ্যালয়": 72000,
        "শিক্ষক": 76000, "ছাত্র": 77000, "ছাত্রী": 75000,
        "খাবার": 82000, "ভাত": 80000, "রুটি": 70000, "মাছ": 78000,
        "মাংস": 75000, "ফল": 77000, "শাকসবজি": 68000,
        "টাকা": 88000, "পয়সা": 75000, "দাম": 79000, "বাজার": 81000,
        "দোকান": 77000, "ব্যবসা": 73000,
        "গান": 84000, "নাচ": 70000, "খেলা": 83000, "সিনেমা": 75000,
        "নাটক": 72000, "গল্প": 80000, "কবিতা": 76000, "উপন্যাস": 71000,
        "সংবাদ": 78000, "খবর": 82000, "পত্রিকা": 74000,
        "সরকার": 85000, "রাজনীতি": 79000, "নেতা": 77000, "মন্ত্রী": 75000,
        "প্রধানমন্ত্রী": 80000, "রাষ্ট্রপতি": 73000,
        "ধর্ম": 81000, "মসজিদ": 72000, "মন্দির": 71000, "গির্জা": 65000,
        "নামাজ": 74000, "পূজা": 73000, "প্রার্থনা": 70000,
        "স্বাস্থ্য": 78000, "রোগ": 76000, "ডাক্তার": 79000, "হাসপাতাল": 77000,
        "ওষুধ": 75000, "চিকিৎসা": 74000,

        # Adjectives
        "ভালো": 95000, "ভাল": 90000, "খারাপ": 85000, "মন্দ": 70000,
        "বড়": 92000, "ছোট": 91000, "ছোটো": 88000, "লম্বা": 75000,
        "উঁচু": 72000, "নিচু": 70000, "গভীর": 68000,
        "সুন্দর": 88000, "কুৎসিত": 55000, "সাদা": 78000, "কালো": 77000,
        "লাল": 76000, "নীল": 75000, "সবুজ": 74000, "হলুদ": 72000,
        "নতুন": 90000, "পুরাতন": 80000, "পুরানো": 82000, "পুরোনো": 81000,
        "গরম": 79000, "ঠান্ডা": 78000, "শীতল": 70000,
        "সহজ": 82000, "কঠিন": 80000, "সরল": 72000, "জটিল": 70000,
        "দ্রুত": 77000, "ধীর": 72000, "দ্রুতগতি": 65000,
        "প্রথম": 86000, "দ্বিতীয়": 80000, "তৃতীয়": 75000, "শেষ": 84000,
        "সব": 93000, "সকল": 88000, "সমস্ত": 82000, "পুরো": 85000,
        "অনেক": 91000, "বেশি": 89000, "কম": 87000, "অল্প": 78000,
        "একটু": 85000, "একটুখানি": 70000,
        "সত্য": 83000, "মিথ্যা": 78000, "সঠিক": 80000, "ভুল": 82000,
        "সম্ভব": 79000, "অসম্ভব": 72000,
        "খুশি": 84000, "দুঃখী": 75000, "আনন্দিত": 72000, "বিরক্ত": 68000,
        "রাগী": 65000, "ভয়": 80000, "ভয়ানক": 70000,

        # Adverbs
        "আজ": 92000, "কাল": 88000, "আগামীকাল": 75000, "গতকাল": 78000,
        "এখন": 94000, "তখন": 85000, "পরে": 88000, "আগে": 87000,
        "সবসময়": 80000, "সারাদিন": 75000, "সারারাত": 70000,
        "এখানে": 90000, "সেখানে": 85000, "কোথায়": 88000, "যেখানে": 82000,
        "কেন": 91000, "কীভাবে": 85000, "কিভাবে": 84000, "কতটা": 78000,
        "খুব": 93000, "অনেক": 91000, "বেশি": 89000, "কম": 87000,
        "একটু": 85000, "আবার": 88000, "আর": 95000, "এবং": 94000,
        "তবে": 86000, "কিন্তু": 90000, "যদি": 89000, "তাহলে": 84000,
        "তাই": 88000, "সুতরাং": 75000, "অতএব": 70000,
        "হয়তো": 82000, "সম্ভবত": 78000, "অবশ্যই": 85000, "নিশ্চয়ই": 80000,
        "শুধু": 87000, "শুধুমাত্র": 80000, "কেবল": 78000, "মাত্র": 82000,
        "প্রায়": 80000, "মোটামুটি": 72000, "সম্পূর্ণ": 78000,
        "সত্যি": 82000, "আসলে": 85000, "বাস্তবে": 75000,
        "ধীরে": 75000, "দ্রুত": 77000, "তাড়াতাড়ি": 78000,
        "একসাথে": 80000, "আলাদা": 75000, "পৃথক": 68000,

        # Postpositions
        "জন্য": 92000, "থেকে": 94000, "পর্যন্ত": 85000, "দিকে": 82000,
        "মধ্যে": 88000, "ভেতরে": 78000, "বাইরে": 80000, "উপরে": 82000,
        "নিচে": 80000, "পাশে": 78000, "কাছে": 85000, "দূরে": 75000,
        "সামনে": 82000, "পেছনে": 78000, "পিছনে": 77000,
        "সাথে": 90000, "সঙ্গে": 88000, "বিনা": 72000, "ছাড়া": 82000,
        "মতো": 85000, "মত": 84000, "মতন": 75000,
        "দ্বারা": 78000, "দিয়ে": 88000, "নিয়ে": 87000,
        "সম্পর্কে": 80000, "বিষয়ে": 78000, "ব্যাপারে": 75000,

        # Conjunctions
        "এবং": 94000, "ও": 95000, "আর": 95000, "কিন্তু": 90000,
        "তবে": 86000, "যদি": 89000, "তাহলে": 84000, "তাই": 88000,
        "কারণ": 85000, "যেহেতু": 75000, "যখন": 87000, "তখন": 85000,
        "যেমন": 82000, "তেমন": 78000, "যত": 80000, "তত": 78000,
        "অথবা": 85000, "কিংবা": 80000, "নাকি": 78000, "না": 96000,
        "হ্যাঁ": 92000, "হাঁ": 85000, "নাহ": 75000, "নাহলে": 78000,

        # Numbers
        "এক": 95000, "দুই": 94000, "তিন": 93000, "চার": 92000,
        "পাঁচ": 91000, "ছয়": 90000, "সাত": 89000, "আট": 88000,
        "নয়": 87000, "দশ": 86000, "শত": 80000, "হাজার": 82000,
        "লাখ": 78000, "কোটি": 75000,
        "প্রথম": 86000, "দ্বিতীয়": 80000, "তৃতীয়": 75000,

        # Common phrases/words
        "ধন্যবাদ": 85000, "দয়া": 78000, "অনুগ্রহ": 72000, "দুঃখিত": 75000,
        "স্বাগতম": 80000, "বিদায়": 72000, "শুভ": 82000, "মঙ্গল": 78000,
        "সকাল": 85000, "দুপুর": 80000, "বিকাল": 78000, "সন্ধ্যা": 77000,
        "নমস্কার": 75000, "সালাম": 78000, "আসসালামুআলাইকুম": 72000,
    }

    def __init__(
        self,
        max_edit_distance: int = 2,
        prefix_length: int = 7,
        dictionary_path: Optional[str] = None,
        custom_words: Optional[Dict[str, int]] = None,
    ):
        """Initialize the Bengali spell checker.

        Args:
            max_edit_distance: Maximum edit distance for corrections (default: 2)
            prefix_length: Length of word prefix for indexing (default: 7)
            dictionary_path: Path to custom dictionary file (one word per line,
                           optionally with frequency: "word frequency")
            custom_words: Dictionary of custom words with frequencies
        """
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self._tokenizer = BasicTokenizer()

        # Initialize SymSpell
        self._sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=prefix_length,
        )

        # Load default Bengali words
        self._load_default_dictionary()

        # Load custom dictionary if provided
        if dictionary_path:
            self.load_dictionary(dictionary_path)

        # Add custom words if provided
        if custom_words:
            for word, freq in custom_words.items():
                self.add_word(word, freq)

    def _load_default_dictionary(self) -> None:
        """Load the default Bengali word dictionary."""
        for word, freq in self._DEFAULT_WORDS.items():
            self._sym_spell.create_dictionary_entry(word, freq)

        # Add stopwords from corpus (common words should be in dictionary)
        for word in BengaliCorpus.stopwords:
            if word not in self._DEFAULT_WORDS:
                self._sym_spell.create_dictionary_entry(word, 50000)

    def load_dictionary(self, path: str) -> int:
        """Load a dictionary from a file.

        The file should have one word per line, optionally with frequency:
        "word" or "word frequency"

        Args:
            path: Path to the dictionary file

        Returns:
            Number of words loaded
        """
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    word, freq = parts[0], int(parts[1])
                else:
                    word, freq = parts[0], 1

                self._sym_spell.create_dictionary_entry(word, freq)
                count += 1

        return count

    def add_word(self, word: str, frequency: int = 1) -> None:
        """Add a word to the dictionary.

        Args:
            word: Word to add
            frequency: Word frequency (higher = more common)
        """
        self._sym_spell.create_dictionary_entry(word, frequency)

    def add_words(self, words: List[str], default_frequency: int = 1) -> None:
        """Add multiple words to the dictionary.

        Args:
            words: List of words to add
            default_frequency: Default frequency for all words
        """
        for word in words:
            self.add_word(word, default_frequency)

    def is_correct(self, word: str) -> bool:
        """Check if a word is spelled correctly.

        Args:
            word: Word to check

        Returns:
            True if the word is in the dictionary
        """
        # Check if word exists in dictionary (edit distance 0)
        suggestions = self._sym_spell.lookup(
            word,
            Verbosity.TOP,
            max_edit_distance=0,
        )
        return len(suggestions) > 0

    def check(self, text: str) -> List[SpellingError]:
        """Check text for spelling errors.

        Args:
            text: Text to check

        Returns:
            List of SpellingError objects for each misspelled word
        """
        errors = []
        tokens = self._tokenizer.tokenize(text)

        # Track position in original text
        current_pos = 0

        for token in tokens:
            # Skip punctuation
            if token in BengaliCorpus.punctuations or len(token) < 2:
                current_pos = text.find(token, current_pos) + len(token)
                continue

            # Skip if token contains non-Bengali characters (likely English/numbers)
            if not self._is_bengali_word(token):
                current_pos = text.find(token, current_pos) + len(token)
                continue

            # Check if word is correct
            if not self.is_correct(token):
                # Get suggestions
                suggestions = self._sym_spell.lookup(
                    token,
                    Verbosity.CLOSEST,
                    max_edit_distance=self.max_edit_distance,
                )

                suggestion_list = [
                    (s.term, s.distance) for s in suggestions[:5]
                ]

                best = suggestions[0].term if suggestions else None
                position = text.find(token, current_pos)

                errors.append(SpellingError(
                    word=token,
                    position=position,
                    suggestions=suggestion_list,
                    best_correction=best,
                ))

            current_pos = text.find(token, current_pos) + len(token)

        return errors

    def correct(self, text: str) -> str:
        """Correct spelling errors in text.

        Args:
            text: Text to correct

        Returns:
            Corrected text
        """
        errors = self.check(text)

        if not errors:
            return text

        # Sort errors by position in reverse order to replace from end
        errors.sort(key=lambda e: e.position, reverse=True)

        corrected = text
        for error in errors:
            if error.best_correction:
                # Replace the misspelled word with the best correction
                start = error.position
                end = start + len(error.word)
                corrected = corrected[:start] + error.best_correction + corrected[end:]

        return corrected

    def suggestions(
        self,
        word: str,
        max_suggestions: int = 5,
    ) -> List[Tuple[str, int]]:
        """Get spelling suggestions for a word.

        Args:
            word: Word to get suggestions for
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (suggestion, edit_distance) tuples
        """
        results = self._sym_spell.lookup(
            word,
            Verbosity.ALL,
            max_edit_distance=self.max_edit_distance,
        )

        return [(s.term, s.distance) for s in results[:max_suggestions]]

    def word_probability(self, word: str) -> float:
        """Get the probability/frequency of a word.

        Args:
            word: Word to check

        Returns:
            Word frequency (0 if not in dictionary)
        """
        results = self._sym_spell.lookup(
            word,
            Verbosity.TOP,
            max_edit_distance=0,
        )

        if results:
            return results[0].count
        return 0

    def _is_bengali_word(self, word: str) -> bool:
        """Check if a word contains Bengali characters.

        Args:
            word: Word to check

        Returns:
            True if the word contains mostly Bengali characters
        """
        bengali_chars = 0
        total_chars = len(word)

        for char in word:
            # Bengali Unicode range: U+0980 to U+09FF
            if '\u0980' <= char <= '\u09FF':
                bengali_chars += 1

        # Consider it Bengali if more than 50% characters are Bengali
        return bengali_chars > total_chars * 0.5

    def __call__(self, text: str) -> str:
        """Callable interface for correction.

        Args:
            text: Text to correct

        Returns:
            Corrected text
        """
        return self.correct(text)
