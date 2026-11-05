"""Weight edits and measurements for local language models.

The launcher in the repository root does not import this package. Tests under
lab/tests cover the arithmetic without torch. Loading Qwen imports torch and
transformers at call time.
"""

__version__ = "0.1.0"


class LabError(ValueError):
    """A refused edit, measurement, or model shape."""


from kilix_llm_lab.algebra import difference_of_means  # noqa: E402

__all__ = ["LabError", "difference_of_means", "__version__"]
