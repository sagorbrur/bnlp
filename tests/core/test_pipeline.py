import unittest
from bnlp import Pipeline, CleanText, BasicTokenizer
from bnlp.core import PipelineStep, PipelineResult, PipelineError


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BasicTokenizer()
        self.cleaner = CleanText()

    def test_pipeline_creation_with_list(self):
        """Test creating pipeline with list of processors."""
        pipeline = Pipeline([self.cleaner, self.tokenizer])
        self.assertEqual(len(pipeline), 2)

    def test_pipeline_creation_empty(self):
        """Test creating empty pipeline."""
        pipeline = Pipeline()
        self.assertEqual(len(pipeline), 0)

    def test_pipeline_add_step(self):
        """Test adding steps to pipeline."""
        pipeline = Pipeline()
        pipeline.add_step("clean", self.cleaner)
        pipeline.add_step("tokenize", self.tokenizer)
        self.assertEqual(len(pipeline), 2)
        self.assertIn("clean", pipeline.steps)
        self.assertIn("tokenize", pipeline.steps)

    def test_pipeline_remove_step(self):
        """Test removing steps from pipeline."""
        pipeline = Pipeline()
        pipeline.add_step("clean", self.cleaner)
        pipeline.add_step("tokenize", self.tokenizer)
        pipeline.remove_step("clean")
        self.assertEqual(len(pipeline), 1)
        self.assertNotIn("clean", pipeline.steps)

    def test_pipeline_execution(self):
        """Test basic pipeline execution."""
        pipeline = Pipeline([self.cleaner, self.tokenizer])
        result = pipeline("আমি বাংলায় গান গাই।")
        self.assertIsInstance(result, list)
        self.assertIn("আমি", result)

    def test_pipeline_callable(self):
        """Test pipeline as callable."""
        pipeline = Pipeline([self.tokenizer])
        result = pipeline("আমি বাংলায় গান গাই।")
        self.assertEqual(result, ["আমি", "বাংলায়", "গান", "গাই", "।"])

    def test_pipeline_run_with_details(self):
        """Test pipeline run with detailed results."""
        pipeline = Pipeline()
        pipeline.add_step("tokenize", self.tokenizer)
        result = pipeline.run("আমি বাংলায় গান গাই।", return_details=True)

        self.assertIsInstance(result, PipelineResult)
        self.assertIn("tokenize", result.steps_executed)
        self.assertIn("tokenize", result.intermediate_results)

    def test_pipeline_disable_step(self):
        """Test disabling a step."""
        pipeline = Pipeline()
        pipeline.add_step("clean", self.cleaner)
        pipeline.add_step("tokenize", self.tokenizer)
        pipeline.disable_step("clean")

        self.assertNotIn("clean", pipeline.enabled_steps)
        self.assertIn("tokenize", pipeline.enabled_steps)

    def test_pipeline_enable_step(self):
        """Test enabling a disabled step."""
        pipeline = Pipeline()
        pipeline.add_step("clean", self.cleaner, enabled=False)
        pipeline.enable_step("clean")

        self.assertIn("clean", pipeline.enabled_steps)

    def test_pipeline_get_step(self):
        """Test getting a step by name."""
        pipeline = Pipeline()
        pipeline.add_step("tokenize", self.tokenizer)
        step = pipeline.get_step("tokenize")

        self.assertIsInstance(step, PipelineStep)
        self.assertEqual(step.name, "tokenize")

    def test_pipeline_get_nonexistent_step(self):
        """Test getting a non-existent step."""
        pipeline = Pipeline()
        step = pipeline.get_step("nonexistent")
        self.assertIsNone(step)

    def test_pipeline_clone(self):
        """Test cloning a pipeline."""
        pipeline = Pipeline([self.tokenizer], name="original")
        cloned = pipeline.clone()

        self.assertEqual(len(cloned), len(pipeline))
        self.assertNotEqual(id(cloned), id(pipeline))

    def test_pipeline_combine(self):
        """Test combining two pipelines."""
        pipeline1 = Pipeline([self.cleaner], name="p1")
        pipeline2 = Pipeline([self.tokenizer], name="p2")
        combined = pipeline1 + pipeline2

        self.assertEqual(len(combined), 2)

    def test_pipeline_repr(self):
        """Test string representation of pipeline."""
        pipeline = Pipeline([self.tokenizer], name="test")
        repr_str = repr(pipeline)
        self.assertIn("Pipeline", repr_str)
        self.assertIn("test", repr_str)

    def test_pipeline_with_empty_input(self):
        """Test pipeline with empty input."""
        pipeline = Pipeline([self.tokenizer])
        result = pipeline("")
        self.assertEqual(result, [])

    def test_pipeline_chaining(self):
        """Test method chaining."""
        pipeline = (
            Pipeline(name="chained")
            .add_step("clean", self.cleaner)
            .add_step("tokenize", self.tokenizer)
        )
        self.assertEqual(len(pipeline), 2)


class TestPipelineStep(unittest.TestCase):
    def test_pipeline_step_creation(self):
        """Test creating a PipelineStep."""
        step = PipelineStep(
            name="test",
            processor=lambda x: x,
            enabled=True
        )
        self.assertEqual(step.name, "test")
        self.assertTrue(step.enabled)

    def test_pipeline_step_disabled(self):
        """Test creating a disabled PipelineStep."""
        step = PipelineStep(
            name="test",
            processor=lambda x: x,
            enabled=False
        )
        self.assertFalse(step.enabled)


class TestPipelineResult(unittest.TestCase):
    def test_pipeline_result_creation(self):
        """Test creating a PipelineResult."""
        result = PipelineResult(
            output=["token1", "token2"],
            steps_executed=["step1"],
            intermediate_results={"step1": ["token1", "token2"]}
        )
        self.assertEqual(result.output, ["token1", "token2"])
        self.assertEqual(result.steps_executed, ["step1"])


if __name__ == "__main__":
    unittest.main()
