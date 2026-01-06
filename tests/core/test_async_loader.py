import unittest
import time
from bnlp import BasicTokenizer
from bnlp.core import (
    AsyncModelLoader,
    LazyModelLoader,
    LoadingStatus,
    LoadingProgress,
    load_model_async,
)


class TestAsyncModelLoader(unittest.TestCase):
    def test_async_loader_creation(self):
        """Test creating an AsyncModelLoader."""
        loader = AsyncModelLoader(BasicTokenizer)
        self.assertEqual(loader.status, LoadingStatus.PENDING)
        self.assertFalse(loader.is_ready)
        self.assertFalse(loader.is_loading)

    def test_async_loader_start_loading(self):
        """Test starting async loading."""
        loader = AsyncModelLoader(BasicTokenizer)
        loader.start_loading()

        # Wait a bit for loading to start
        time.sleep(0.1)

        # Should be loading or completed
        self.assertIn(loader.status, [LoadingStatus.LOADING, LoadingStatus.COMPLETED])

    def test_async_loader_get_model(self):
        """Test getting the loaded model."""
        loader = AsyncModelLoader(BasicTokenizer)
        loader.start_loading()
        model = loader.get_model(timeout=5)

        self.assertIsInstance(model, BasicTokenizer)
        self.assertTrue(loader.is_ready)
        self.assertEqual(loader.status, LoadingStatus.COMPLETED)

    def test_async_loader_auto_start(self):
        """Test that get_model auto-starts loading if not started."""
        loader = AsyncModelLoader(BasicTokenizer)
        model = loader.get_model(timeout=5)

        self.assertIsInstance(model, BasicTokenizer)

    def test_async_loader_wait(self):
        """Test waiting for model to load."""
        loader = AsyncModelLoader(BasicTokenizer)
        loader.start_loading()
        success = loader.wait(timeout=5)

        self.assertTrue(success)
        self.assertTrue(loader.is_ready)

    def test_async_loader_with_callbacks(self):
        """Test async loader with callbacks."""
        progress_called = []
        complete_called = []

        def on_progress(progress):
            progress_called.append(progress)

        def on_complete(model):
            complete_called.append(model)

        loader = AsyncModelLoader(
            BasicTokenizer,
            on_progress=on_progress,
            on_complete=on_complete,
        )
        loader.start_loading()
        loader.wait(timeout=5)

        self.assertTrue(len(progress_called) > 0)
        self.assertEqual(len(complete_called), 1)
        self.assertIsInstance(complete_called[0], BasicTokenizer)

    def test_async_loader_context_manager(self):
        """Test async loader as context manager."""
        with AsyncModelLoader(BasicTokenizer) as loader:
            model = loader.get_model(timeout=5)
            self.assertIsInstance(model, BasicTokenizer)

    def test_async_loader_method_chaining(self):
        """Test method chaining with start_loading."""
        loader = AsyncModelLoader(BasicTokenizer).start_loading()
        self.assertIsInstance(loader, AsyncModelLoader)


class TestLazyModelLoader(unittest.TestCase):
    def test_lazy_loader_creation(self):
        """Test creating a LazyModelLoader."""
        lazy = LazyModelLoader(BasicTokenizer)
        self.assertFalse(lazy.is_loaded)

    def test_lazy_loader_get(self):
        """Test getting model from lazy loader."""
        lazy = LazyModelLoader(BasicTokenizer)
        model = lazy.get()

        self.assertIsInstance(model, BasicTokenizer)
        self.assertTrue(lazy.is_loaded)

    def test_lazy_loader_caches_model(self):
        """Test that lazy loader caches the model."""
        lazy = LazyModelLoader(BasicTokenizer)
        model1 = lazy.get()
        model2 = lazy.get()

        self.assertIs(model1, model2)

    def test_lazy_loader_preload(self):
        """Test preloading the model."""
        lazy = LazyModelLoader(BasicTokenizer)
        result = lazy.preload()

        self.assertIs(result, lazy)
        self.assertTrue(lazy.is_loaded)


class TestLoadModelAsync(unittest.TestCase):
    def test_load_model_async(self):
        """Test load_model_async convenience function."""
        loader = load_model_async(BasicTokenizer)

        self.assertIsInstance(loader, AsyncModelLoader)
        # Should already be started
        model = loader.get_model(timeout=5)
        self.assertIsInstance(model, BasicTokenizer)

    def test_load_model_async_with_callback(self):
        """Test load_model_async with callbacks."""
        completed = []

        loader = load_model_async(
            BasicTokenizer,
            on_complete=lambda m: completed.append(m)
        )
        loader.wait(timeout=5)

        self.assertEqual(len(completed), 1)


class TestLoadingProgress(unittest.TestCase):
    def test_loading_progress_creation(self):
        """Test creating LoadingProgress."""
        progress = LoadingProgress(
            status=LoadingStatus.LOADING,
            progress=0.5,
            message="Loading...",
        )

        self.assertEqual(progress.status, LoadingStatus.LOADING)
        self.assertEqual(progress.progress, 0.5)
        self.assertEqual(progress.message, "Loading...")
        self.assertIsNone(progress.error)

    def test_loading_progress_with_error(self):
        """Test LoadingProgress with error."""
        error = Exception("Test error")
        progress = LoadingProgress(
            status=LoadingStatus.FAILED,
            progress=0.0,
            message="Failed",
            error=error,
        )

        self.assertEqual(progress.status, LoadingStatus.FAILED)
        self.assertEqual(progress.error, error)


class TestLoadingStatus(unittest.TestCase):
    def test_loading_status_values(self):
        """Test LoadingStatus enum values."""
        self.assertEqual(LoadingStatus.PENDING.value, "pending")
        self.assertEqual(LoadingStatus.LOADING.value, "loading")
        self.assertEqual(LoadingStatus.COMPLETED.value, "completed")
        self.assertEqual(LoadingStatus.FAILED.value, "failed")


if __name__ == "__main__":
    unittest.main()
