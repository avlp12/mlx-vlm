import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass
class PromptLookupConfig:
    """Config for the weight-free prompt-lookup drafter.

    There is no checkpoint, so every field has a default and the whole thing is
    constructible from the environment.
    """

    model_type: str = "prompt_lookup"
    n_min: int = 3
    n_max: int = 5
    block_size: int = 8
    keep: int = 2
    adaptive: bool = True
    # EMA horizon for the accept-rate gate, in rounds.
    adaptive_window: int = 16
    # Proposals to keep making even when the gate has collapsed, so the EMA can
    # recover on a workload that starts quoting again.
    explore_tokens: int = 1

    @classmethod
    def from_env(cls) -> "PromptLookupConfig":
        return cls(
            n_min=_env_int("MLX_VLM_LOOKUP_NMIN", 3),
            n_max=_env_int("MLX_VLM_LOOKUP_NMAX", 5),
            block_size=_env_int("MLX_VLM_LOOKUP_BLOCK", 8),
            keep=_env_int("MLX_VLM_LOOKUP_KEEP", 2),
            adaptive=_env_flag("MLX_VLM_LOOKUP_ADAPTIVE", True),
            adaptive_window=_env_int("MLX_VLM_LOOKUP_WINDOW", 16),
            explore_tokens=_env_int("MLX_VLM_LOOKUP_EXPLORE", 1),
        )

    def validate(self) -> None:
        if not 1 <= self.n_min <= self.n_max:
            raise ValueError("prompt-lookup requires 1 <= n_min <= n_max")
        if self.block_size < 2:
            raise ValueError("prompt-lookup requires block_size >= 2")
        if not 0 <= self.explore_tokens < self.block_size:
            raise ValueError("explore_tokens must be in [0, block_size)")


__all__ = ["PromptLookupConfig"]
