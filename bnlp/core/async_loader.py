"""
BNLP Async Model Loading

This module provides async/background model loading capabilities.
"""

import os
import threading
from typing import Optional, Callable, Any, TypeVar, Generic
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum


class LoadingStatus(Enum):
    """Status of model loading."""
    PENDING = "pending"
    LOADING = "loading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LoadingProgress:
    """Progress information for model loading."""
    status: LoadingStatus
    progress: float  # 0.0 to 1.0
    message: str
    error: Optional[Exception] = None


T = TypeVar("T")


class AsyncModelLoader(Generic[T]):
    """Async model loader for loading models in the background.

    This class allows loading large models without blocking the main thread.

    Example:
        >>> from bnlp.core import AsyncModelLoader
        >>> from bnlp import BengaliWord2Vec
        >>>
        >>> # Create async loader
        >>> loader = AsyncModelLoader(BengaliWord2Vec)
        >>>
        >>> # Start loading in background
        >>> loader.start_loading()
        >>>
        >>> # Do other work while loading...
        >>> print("Model loading in background...")
        >>>
        >>> # Wait for model when needed
        >>> model = loader.get_model()  # Blocks until ready
        >>> vector = model.get_word_vector("বাংলা")
    """

    def __init__(
        self,
        model_class: type,
        *args,
        on_progress: Optional[Callable[[LoadingProgress], None]] = None,
        on_complete: Optional[Callable[[T], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        **kwargs,
    ):
        """Initialize AsyncModelLoader.

        Args:
            model_class: The model class to instantiate
            *args: Positional arguments for model constructor
            on_progress: Callback for progress updates
            on_complete: Callback when loading completes
            on_error: Callback when loading fails
            **kwargs: Keyword arguments for model constructor
        """
        self.model_class = model_class
        self.args = args
        self.kwargs = kwargs
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error

        self._model: Optional[T] = None
        self._future: Optional[Future] = None
        self._status = LoadingStatus.PENDING
        self._error: Optional[Exception] = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def status(self) -> LoadingStatus:
        """Get current loading status."""
        return self._status

    @property
    def is_ready(self) -> bool:
        """Check if model is ready to use."""
        return self._status == LoadingStatus.COMPLETED and self._model is not None

    @property
    def is_loading(self) -> bool:
        """Check if model is currently loading."""
        return self._status == LoadingStatus.LOADING

    def _update_progress(self, progress: float, message: str) -> None:
        """Update progress and call callback."""
        if self.on_progress:
            self.on_progress(LoadingProgress(
                status=self._status,
                progress=progress,
                message=message,
            ))

    def _load_model(self) -> T:
        """Internal method to load the model."""
        with self._lock:
            self._status = LoadingStatus.LOADING

        self._update_progress(0.1, "Initializing model loading...")

        try:
            self._update_progress(0.3, "Loading model...")
            model = self.model_class(*self.args, **self.kwargs)

            self._update_progress(0.9, "Finalizing...")

            with self._lock:
                self._model = model
                self._status = LoadingStatus.COMPLETED

            self._update_progress(1.0, "Model loaded successfully!")

            if self.on_complete:
                self.on_complete(model)

            return model

        except Exception as e:
            with self._lock:
                self._status = LoadingStatus.FAILED
                self._error = e

            if self.on_progress:
                self.on_progress(LoadingProgress(
                    status=LoadingStatus.FAILED,
                    progress=0.0,
                    message=f"Loading failed: {str(e)}",
                    error=e,
                ))

            if self.on_error:
                self.on_error(e)

            raise

    def start_loading(self) -> "AsyncModelLoader[T]":
        """Start loading the model in the background.

        Returns:
            self for method chaining
        """
        if self._status != LoadingStatus.PENDING:
            return self

        self._future = self._executor.submit(self._load_model)
        return self

    def get_model(self, timeout: Optional[float] = None) -> T:
        """Get the loaded model, waiting if necessary.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            The loaded model

        Raises:
            TimeoutError: If timeout is reached before model is ready
            Exception: If model loading failed
        """
        if self._model is not None:
            return self._model

        if self._future is None:
            self.start_loading()

        try:
            return self._future.result(timeout=timeout)
        except Exception as e:
            if self._error:
                raise self._error
            raise

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for model to finish loading.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.get_model(timeout=timeout)
            return True
        except Exception:
            return False

    def __enter__(self) -> "AsyncModelLoader[T]":
        """Context manager entry."""
        self.start_loading()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self._executor.shutdown(wait=False)

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)


def load_model_async(
    model_class: type,
    *args,
    on_progress: Optional[Callable[[LoadingProgress], None]] = None,
    on_complete: Optional[Callable[[Any], None]] = None,
    **kwargs,
) -> AsyncModelLoader:
    """Convenience function to create and start an async model loader.

    Args:
        model_class: The model class to instantiate
        *args: Positional arguments for model constructor
        on_progress: Callback for progress updates
        on_complete: Callback when loading completes
        **kwargs: Keyword arguments for model constructor

    Returns:
        AsyncModelLoader instance (already started)

    Example:
        >>> from bnlp.core import load_model_async
        >>> from bnlp import BengaliWord2Vec
        >>>
        >>> loader = load_model_async(
        ...     BengaliWord2Vec,
        ...     on_complete=lambda m: print("Model ready!")
        ... )
        >>>
        >>> # Do other work...
        >>>
        >>> model = loader.get_model()
    """
    loader = AsyncModelLoader(
        model_class,
        *args,
        on_progress=on_progress,
        on_complete=on_complete,
        **kwargs,
    )
    loader.start_loading()
    return loader


class LazyModelLoader(Generic[T]):
    """Lazy model loader that loads model on first access.

    Example:
        >>> from bnlp.core import LazyModelLoader
        >>> from bnlp import BengaliWord2Vec
        >>>
        >>> # Model not loaded yet
        >>> lazy_model = LazyModelLoader(BengaliWord2Vec)
        >>>
        >>> # Model loads on first access
        >>> vector = lazy_model.get().get_word_vector("বাংলা")
    """

    def __init__(self, model_class: type, *args, **kwargs):
        """Initialize LazyModelLoader.

        Args:
            model_class: The model class to instantiate
            *args: Positional arguments for model constructor
            **kwargs: Keyword arguments for model constructor
        """
        self.model_class = model_class
        self.args = args
        self.kwargs = kwargs
        self._model: Optional[T] = None
        self._lock = threading.Lock()

    def get(self) -> T:
        """Get the model, loading it if necessary.

        Returns:
            The loaded model
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self.model_class(*self.args, **self.kwargs)
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if model is already loaded."""
        return self._model is not None

    def preload(self) -> "LazyModelLoader[T]":
        """Preload the model.

        Returns:
            self for method chaining
        """
        self.get()
        return self
