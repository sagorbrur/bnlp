"""
BNLP Custom Exceptions

This module provides custom exception classes for better error handling.
"""


class BNLPException(Exception):
    """Base exception for all BNLP errors."""

    def __init__(self, message: str, details: str = ""):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ModelNotFoundError(BNLPException):
    """Raised when a model file is not found."""

    def __init__(self, model_name: str, model_path: str = ""):
        message = f"Model '{model_name}' not found."
        details = ""
        if model_path:
            details = f"Expected path: {model_path}"
        details += "\nTry downloading the model using: bnlp download <model_name>"
        super().__init__(message, details)
        self.model_name = model_name
        self.model_path = model_path


class ModelLoadError(BNLPException):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, reason: str = ""):
        message = f"Failed to load model '{model_name}'."
        details = reason if reason else "The model file may be corrupted or incompatible."
        super().__init__(message, details)
        self.model_name = model_name


class TokenizationError(BNLPException):
    """Raised when tokenization fails."""

    def __init__(self, text: str = "", reason: str = ""):
        message = "Tokenization failed."
        details = reason
        if text:
            preview = text[:50] + "..." if len(text) > 50 else text
            details = f"Input text: '{preview}'\n{reason}"
        super().__init__(message, details)


class EmbeddingError(BNLPException):
    """Raised when word/document embedding fails."""

    def __init__(self, word: str = "", reason: str = ""):
        message = "Embedding generation failed."
        details = reason
        if word:
            details = f"Word/text: '{word}'\n{reason}"
        super().__init__(message, details)
        self.word = word


class TaggingError(BNLPException):
    """Raised when POS tagging or NER fails."""

    def __init__(self, text: str = "", tag_type: str = "tagging", reason: str = ""):
        message = f"{tag_type.upper()} tagging failed."
        details = reason
        if text:
            preview = text[:50] + "..." if len(text) > 50 else text
            details = f"Input text: '{preview}'\n{reason}"
        super().__init__(message, details)


class DownloadError(BNLPException):
    """Raised when model download fails."""

    def __init__(self, model_name: str, url: str = "", reason: str = ""):
        message = f"Failed to download model '{model_name}'."
        details = reason
        if url:
            details = f"URL: {url}\n{reason}"
        details += "\nPlease check your internet connection and try again."
        super().__init__(message, details)
        self.model_name = model_name
        self.url = url


class PipelineError(BNLPException):
    """Raised when pipeline execution fails."""

    def __init__(self, step_name: str = "", reason: str = ""):
        message = "Pipeline execution failed."
        details = reason
        if step_name:
            details = f"Failed at step: '{step_name}'\n{reason}"
        super().__init__(message, details)
        self.step_name = step_name


class InvalidInputError(BNLPException):
    """Raised when input validation fails."""

    def __init__(self, param_name: str, expected: str, received: str = ""):
        message = f"Invalid input for parameter '{param_name}'."
        details = f"Expected: {expected}"
        if received:
            details += f"\nReceived: {received}"
        super().__init__(message, details)
        self.param_name = param_name
