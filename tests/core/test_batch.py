import unittest
from bnlp import BasicTokenizer, CleanText
from bnlp.core import (
    BatchProcessor,
    tokenize_batch,
    clean_batch,
)


class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BasicTokenizer()

    def test_batch_processor_creation(self):
        """Test creating a BatchProcessor."""
        batch = BatchProcessor(self.tokenizer.tokenize)
        self.assertIsNotNone(batch)

    def test_batch_processor_process(self):
        """Test batch processing multiple texts."""
        batch = BatchProcessor(self.tokenizer.tokenize)
        texts = ["আমি বাংলায় গান গাই।", "তুমি কোথায় যাও?"]
        results = batch.process(texts)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], list)
        self.assertIsInstance(results[1], list)

    def test_batch_processor_callable(self):
        """Test BatchProcessor as callable."""
        batch = BatchProcessor(self.tokenizer.tokenize)
        texts = ["আমি বাংলায় গান গাই।", "তুমি কোথায় যাও?"]
        results = batch(texts)

        self.assertEqual(len(results), 2)

    def test_batch_processor_empty_input(self):
        """Test batch processing with empty input."""
        batch = BatchProcessor(self.tokenizer.tokenize)
        results = batch.process([])
        self.assertEqual(results, [])

    def test_batch_processor_single_item(self):
        """Test batch processing with single item."""
        batch = BatchProcessor(self.tokenizer.tokenize)
        texts = ["আমি বাংলায় গান গাই।"]
        results = batch.process(texts)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], ["আমি", "বাংলায়", "গান", "গাই", "।"])

    def test_batch_processor_with_max_workers(self):
        """Test batch processing with custom max_workers."""
        batch = BatchProcessor(self.tokenizer.tokenize, max_workers=2)
        texts = ["আমি বাংলায় গান গাই।"] * 10
        results = batch.process(texts)

        self.assertEqual(len(results), 10)


class TestTokenizeBatch(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BasicTokenizer()

    def test_tokenize_batch(self):
        """Test tokenize_batch function."""
        texts = ["আমি বাংলায় গান গাই।", "তুমি কোথায় যাও?", "সে বই পড়ে।"]
        results = tokenize_batch(self.tokenizer.tokenize, texts)

        self.assertEqual(len(results), 3)
        self.assertIn("আমি", results[0])
        self.assertIn("তুমি", results[1])
        self.assertIn("সে", results[2])

    def test_tokenize_batch_empty(self):
        """Test tokenize_batch with empty input."""
        results = tokenize_batch(self.tokenizer.tokenize, [])
        self.assertEqual(results, [])

    def test_tokenize_batch_single(self):
        """Test tokenize_batch with single text."""
        texts = ["আমি বাংলায় গান গাই।"]
        results = tokenize_batch(self.tokenizer.tokenize, texts)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], ["আমি", "বাংলায়", "গান", "গাই", "।"])


class TestCleanBatch(unittest.TestCase):
    def setUp(self):
        self.cleaner = CleanText(remove_punct=True)

    def test_clean_batch(self):
        """Test clean_batch function."""
        texts = ["আমি বাংলায়!", "তুমি কোথায়?"]
        results = clean_batch(self.cleaner, texts)

        self.assertEqual(len(results), 2)
        # Punctuation should be replaced
        self.assertIn("<PUNC>", results[0])

    def test_clean_batch_empty(self):
        """Test clean_batch with empty input."""
        results = clean_batch(self.cleaner, [])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
