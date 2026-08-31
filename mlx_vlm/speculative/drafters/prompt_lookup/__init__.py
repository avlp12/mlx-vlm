from .config import PromptLookupConfig
from .lookup import Model, PromptLookupDraftModel
from .ngram import NgramIndex

ModelConfig = PromptLookupConfig

__all__ = [
    "Model",
    "ModelConfig",
    "NgramIndex",
    "PromptLookupConfig",
    "PromptLookupDraftModel",
]
