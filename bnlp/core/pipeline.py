"""
BNLP Pipeline API

This module provides a Pipeline class for chaining NLP operations.
"""

from typing import List, Any, Callable, Optional, Union, Dict
from dataclasses import dataclass, field


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    name: str
    processor: Callable[[Any], Any]
    enabled: bool = True


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    output: Any
    steps_executed: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """A pipeline for chaining NLP operations.

    The Pipeline class allows you to chain multiple NLP operations together
    and execute them in sequence. Each step receives the output of the
    previous step as its input.

    Example:
        >>> from bnlp import CleanText, BasicTokenizer
        >>> from bnlp.core import Pipeline
        >>>
        >>> # Create pipeline
        >>> pipeline = Pipeline([
        ...     CleanText(remove_punct=True),
        ...     BasicTokenizer(),
        ... ])
        >>>
        >>> # Process text
        >>> result = pipeline("আমি বাংলায় গান গাই।")
        >>> print(result)
        ['আমি', 'বাংলায়', 'গান', 'গাই']

    Example with named steps:
        >>> pipeline = Pipeline()
        >>> pipeline.add_step("clean", CleanText(remove_url=True))
        >>> pipeline.add_step("tokenize", BasicTokenizer())
        >>>
        >>> # Get detailed result
        >>> result = pipeline.run("Check https://example.com আমি বাংলায়", return_details=True)
        >>> print(result.intermediate_results)
    """

    def __init__(
        self,
        steps: Optional[List[Union[Callable, PipelineStep]]] = None,
        name: str = "pipeline",
    ):
        """Initialize Pipeline.

        Args:
            steps: List of processor functions or PipelineStep objects
            name: Name of the pipeline
        """
        self.name = name
        self._steps: List[PipelineStep] = []

        if steps:
            for i, step in enumerate(steps):
                if isinstance(step, PipelineStep):
                    self._steps.append(step)
                else:
                    step_name = getattr(step, "__class__.__name__", f"step_{i}")
                    if hasattr(step, "__class__"):
                        step_name = step.__class__.__name__
                    self._steps.append(PipelineStep(
                        name=step_name,
                        processor=step,
                    ))

    def add_step(
        self,
        name: str,
        processor: Callable[[Any], Any],
        enabled: bool = True,
    ) -> "Pipeline":
        """Add a step to the pipeline.

        Args:
            name: Name of the step
            processor: Processing function
            enabled: Whether the step is enabled

        Returns:
            self for method chaining
        """
        self._steps.append(PipelineStep(
            name=name,
            processor=processor,
            enabled=enabled,
        ))
        return self

    def remove_step(self, name: str) -> "Pipeline":
        """Remove a step from the pipeline by name.

        Args:
            name: Name of the step to remove

        Returns:
            self for method chaining
        """
        self._steps = [s for s in self._steps if s.name != name]
        return self

    def enable_step(self, name: str) -> "Pipeline":
        """Enable a step by name.

        Args:
            name: Name of the step to enable

        Returns:
            self for method chaining
        """
        for step in self._steps:
            if step.name == name:
                step.enabled = True
                break
        return self

    def disable_step(self, name: str) -> "Pipeline":
        """Disable a step by name.

        Args:
            name: Name of the step to disable

        Returns:
            self for method chaining
        """
        for step in self._steps:
            if step.name == name:
                step.enabled = False
                break
        return self

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name.

        Args:
            name: Name of the step

        Returns:
            PipelineStep or None if not found
        """
        for step in self._steps:
            if step.name == name:
                return step
        return None

    @property
    def steps(self) -> List[str]:
        """Get list of step names."""
        return [s.name for s in self._steps]

    @property
    def enabled_steps(self) -> List[str]:
        """Get list of enabled step names."""
        return [s.name for s in self._steps if s.enabled]

    def run(
        self,
        input_data: Any,
        return_details: bool = False,
        stop_on_error: bool = True,
    ) -> Union[Any, PipelineResult]:
        """Run the pipeline on input data.

        Args:
            input_data: Input data for the pipeline
            return_details: Return PipelineResult with details
            stop_on_error: Stop execution if a step fails

        Returns:
            Output data (or PipelineResult if return_details=True)

        Raises:
            Exception: If a step fails and stop_on_error=True
        """
        current = input_data
        steps_executed = []
        intermediate_results = {}

        for step in self._steps:
            if not step.enabled:
                continue

            try:
                current = step.processor(current)
                steps_executed.append(step.name)
                intermediate_results[step.name] = current
            except Exception as e:
                if stop_on_error:
                    from bnlp.core.exceptions import PipelineError
                    raise PipelineError(
                        step_name=step.name,
                        reason=str(e),
                    ) from e
                # Continue with previous value if not stopping on error
                intermediate_results[step.name] = f"Error: {e}"

        if return_details:
            return PipelineResult(
                output=current,
                steps_executed=steps_executed,
                intermediate_results=intermediate_results,
            )

        return current

    def __call__(self, input_data: Any) -> Any:
        """Callable interface for pipeline execution.

        Args:
            input_data: Input data for the pipeline

        Returns:
            Output data
        """
        return self.run(input_data)

    def __len__(self) -> int:
        """Get number of steps in pipeline."""
        return len(self._steps)

    def __repr__(self) -> str:
        """String representation of pipeline."""
        steps_str = " -> ".join(self.steps)
        return f"Pipeline({self.name}): {steps_str}"

    def clone(self) -> "Pipeline":
        """Create a copy of the pipeline.

        Returns:
            New Pipeline instance with same steps
        """
        new_pipeline = Pipeline(name=f"{self.name}_copy")
        for step in self._steps:
            new_pipeline._steps.append(PipelineStep(
                name=step.name,
                processor=step.processor,
                enabled=step.enabled,
            ))
        return new_pipeline

    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Combine two pipelines.

        Args:
            other: Another pipeline to append

        Returns:
            New combined Pipeline
        """
        combined = self.clone()
        combined.name = f"{self.name}+{other.name}"
        for step in other._steps:
            combined._steps.append(PipelineStep(
                name=step.name,
                processor=step.processor,
                enabled=step.enabled,
            ))
        return combined


# Pre-built pipeline factories
def create_tokenization_pipeline(
    clean: bool = True,
    tokenizer_type: str = "basic",
) -> Pipeline:
    """Create a tokenization pipeline.

    Args:
        clean: Whether to include text cleaning
        tokenizer_type: Type of tokenizer (basic, nltk, sentencepiece)

    Returns:
        Configured Pipeline
    """
    from bnlp import CleanText, BasicTokenizer, NLTKTokenizer

    steps = []

    if clean:
        steps.append(CleanText())

    if tokenizer_type == "basic":
        steps.append(BasicTokenizer())
    elif tokenizer_type == "nltk":
        tokenizer = NLTKTokenizer()
        steps.append(lambda x: tokenizer.word_tokenize(x))
    else:
        from bnlp import SentencepieceTokenizer
        steps.append(SentencepieceTokenizer())

    return Pipeline(steps, name=f"tokenization_{tokenizer_type}")


def create_ner_pipeline(
    clean: bool = True,
    tokenizer_type: str = "basic",
) -> Pipeline:
    """Create an NER pipeline.

    Args:
        clean: Whether to include text cleaning
        tokenizer_type: Type of tokenizer for cleaning step

    Returns:
        Configured Pipeline
    """
    from bnlp import CleanText, BengaliNER

    steps = []

    if clean:
        steps.append(CleanText())

    steps.append(BengaliNER())

    return Pipeline(steps, name="ner_pipeline")


def create_pos_pipeline(
    clean: bool = True,
) -> Pipeline:
    """Create a POS tagging pipeline.

    Args:
        clean: Whether to include text cleaning

    Returns:
        Configured Pipeline
    """
    from bnlp import CleanText, BengaliPOS

    steps = []

    if clean:
        steps.append(CleanText())

    steps.append(BengaliPOS())

    return Pipeline(steps, name="pos_pipeline")
