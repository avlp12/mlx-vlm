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
    # EMA horizon for the width gate, in *proposing* rounds.  Abstentions do
    # not inform it: whether a match exists is already known for free, so the
    # gate only has to answer "when I do match, how much of it survives?".
    adaptive_window: int = 16
    # Marginal cost of one extra proposed token, in units of a plain decode
    # step.  On GLM-5.3-Flash a verify at L=8 costs 2.23x a verify at L=1
    # (36.25 ms vs 80.7 ms; logs/vlm_m48_dflash2_p512_block8.json and the AIF
    # I798 round budget), i.e. ~0.175 per token.  The width gate proposes one
    # more token exactly while the chance it survives exceeds this.
    verify_cost_per_token: float = 0.175

    @classmethod
    def from_env(cls) -> "PromptLookupConfig":
        return cls(
            n_min=_env_int("MLX_VLM_LOOKUP_NMIN", 3),
            n_max=_env_int("MLX_VLM_LOOKUP_NMAX", 5),
            block_size=_env_int("MLX_VLM_LOOKUP_BLOCK", 8),
            keep=_env_int("MLX_VLM_LOOKUP_KEEP", 2),
            adaptive=_env_flag("MLX_VLM_LOOKUP_ADAPTIVE", True),
            adaptive_window=_env_int("MLX_VLM_LOOKUP_WINDOW", 16),
            verify_cost_per_token=float(
                os.environ.get("MLX_VLM_LOOKUP_VERIFY_COST", 0.175)
            ),
        )

    def validate(self) -> None:
        if not 1 <= self.n_min <= self.n_max:
            raise ValueError("prompt-lookup requires 1 <= n_min <= n_max")
        if self.block_size < 2:
            raise ValueError("prompt-lookup requires block_size >= 2")
        if not 0.0 < self.verify_cost_per_token < 1.0:
            raise ValueError("verify_cost_per_token must be in (0, 1)")


__all__ = ["PromptLookupConfig"]
