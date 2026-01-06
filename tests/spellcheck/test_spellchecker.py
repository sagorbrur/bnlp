import unittest
import tempfile
import os
from bnlp import BengaliSpellChecker, SpellingError


class TestBengaliSpellChecker(unittest.TestCase):
    def setUp(self):
        self.checker = BengaliSpellChecker()

    def test_spellchecker_creation(self):
        """Test creating a BengaliSpellChecker."""
        checker = BengaliSpellChecker()
        self.assertIsNotNone(checker)

    def test_spellchecker_with_custom_distance(self):
        """Test creating checker with custom edit distance."""
        checker = BengaliSpellChecker(max_edit_distance=1)
        self.assertEqual(checker.max_edit_distance, 1)

    def test_is_correct_valid_word(self):
        """Test is_correct with valid Bengali words."""
        self.assertTrue(self.checker.is_correct("আমি"))
        self.assertTrue(self.checker.is_correct("বাংলাদেশ"))
        self.assertTrue(self.checker.is_correct("করি"))

    def test_is_correct_invalid_word(self):
        """Test is_correct with misspelled words."""
        # These are intentionally misspelled
        self.assertFalse(self.checker.is_correct("আমর"))  # আমি -> আমর
        self.assertFalse(self.checker.is_correct("বাংলাদশ"))  # বাংলাদেশ -> বাংলাদশ

    def test_check_no_errors(self):
        """Test check with correct text."""
        text = "আমি বাংলায় গান গাই।"
        errors = self.checker.check(text)
        # Most common words should be in dictionary
        self.assertIsInstance(errors, list)

    def test_check_with_errors(self):
        """Test check with misspelled text."""
        text = "আমর দেশ বাংলাদশ"  # আমি -> আমর, বাংলাদেশ -> বাংলাদশ
        errors = self.checker.check(text)
        self.assertIsInstance(errors, list)
        # Should find at least one error
        self.assertGreater(len(errors), 0)

    def test_check_returns_spelling_errors(self):
        """Test that check returns SpellingError objects."""
        text = "আমর ভালো"  # আমর is misspelled
        errors = self.checker.check(text)
        if errors:
            self.assertIsInstance(errors[0], SpellingError)
            self.assertEqual(errors[0].word, "আমর")
            self.assertIsNotNone(errors[0].suggestions)

    def test_correct_text(self):
        """Test correcting misspelled text."""
        text = "আমর ভালো আছি"  # আমর -> আমি
        corrected = self.checker.correct(text)
        self.assertIsInstance(corrected, str)
        # Should attempt correction
        self.assertNotEqual(corrected, "")

    def test_correct_no_changes_needed(self):
        """Test correct with already correct text."""
        text = "আমি ভালো আছি"
        corrected = self.checker.correct(text)
        # Should return same or similar text
        self.assertIsInstance(corrected, str)

    def test_suggestions(self):
        """Test getting suggestions for a word."""
        suggestions = self.checker.suggestions("আমর")
        self.assertIsInstance(suggestions, list)
        # Should return suggestions as (word, distance) tuples
        if suggestions:
            self.assertIsInstance(suggestions[0], tuple)
            self.assertEqual(len(suggestions[0]), 2)

    def test_suggestions_max_count(self):
        """Test limiting suggestion count."""
        suggestions = self.checker.suggestions("আমর", max_suggestions=3)
        self.assertLessEqual(len(suggestions), 3)

    def test_add_word(self):
        """Test adding a custom word."""
        custom_word = "কাস্টমশব্দ"
        self.checker.add_word(custom_word, frequency=1000)
        self.assertTrue(self.checker.is_correct(custom_word))

    def test_add_words(self):
        """Test adding multiple custom words."""
        custom_words = ["শব্দএক", "শব্দদুই", "শব্দতিন"]
        self.checker.add_words(custom_words, default_frequency=500)
        for word in custom_words:
            self.assertTrue(self.checker.is_correct(word))

    def test_word_probability(self):
        """Test getting word probability."""
        prob = self.checker.word_probability("আমি")
        self.assertGreater(prob, 0)

    def test_word_probability_unknown(self):
        """Test word probability for unknown word."""
        prob = self.checker.word_probability("অজানাশব্দ")
        self.assertEqual(prob, 0)

    def test_callable(self):
        """Test using checker as callable."""
        text = "আমর ভালো"
        result = self.checker(text)
        self.assertIsInstance(result, str)

    def test_load_dictionary_from_file(self):
        """Test loading dictionary from file."""
        # Create temporary dictionary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                          delete=False, encoding='utf-8') as f:
            f.write("টেস্টশব্দ 1000\n")
            f.write("আরেকটি 500\n")
            temp_path = f.name

        try:
            checker = BengaliSpellChecker()
            count = checker.load_dictionary(temp_path)
            self.assertEqual(count, 2)
            self.assertTrue(checker.is_correct("টেস্টশব্দ"))
            self.assertTrue(checker.is_correct("আরেকটি"))
        finally:
            os.unlink(temp_path)

    def test_custom_words_in_constructor(self):
        """Test passing custom words in constructor."""
        custom = {"নতুনশব্দ": 1000, "আরেকটিশব্দ": 500}
        checker = BengaliSpellChecker(custom_words=custom)
        self.assertTrue(checker.is_correct("নতুনশব্দ"))
        self.assertTrue(checker.is_correct("আরেকটিশব্দ"))

    def test_skip_punctuation(self):
        """Test that punctuation is skipped during check."""
        text = "আমি। তুমি!"
        errors = self.checker.check(text)
        # Punctuation should not be flagged as errors
        for error in errors:
            self.assertNotIn(error.word, ["।", "!"])

    def test_skip_short_words(self):
        """Test that very short words are skipped."""
        text = "আ ই উ"
        errors = self.checker.check(text)
        # Single characters should not be checked
        for error in errors:
            self.assertGreater(len(error.word), 1)


class TestSpellingError(unittest.TestCase):
    def test_spelling_error_creation(self):
        """Test creating a SpellingError."""
        error = SpellingError(
            word="আমর",
            position=0,
            suggestions=[("আমি", 1), ("আমার", 2)],
            best_correction="আমি"
        )
        self.assertEqual(error.word, "আমর")
        self.assertEqual(error.position, 0)
        self.assertEqual(error.best_correction, "আমি")
        self.assertEqual(len(error.suggestions), 2)

    def test_spelling_error_repr(self):
        """Test SpellingError string representation."""
        error = SpellingError(
            word="আমর",
            position=0,
            best_correction="আমি"
        )
        repr_str = repr(error)
        self.assertIn("আমর", repr_str)
        self.assertIn("আমি", repr_str)

    def test_spelling_error_defaults(self):
        """Test SpellingError default values."""
        error = SpellingError(word="test", position=0)
        self.assertEqual(error.suggestions, [])
        self.assertIsNone(error.best_correction)


if __name__ == "__main__":
    unittest.main()
