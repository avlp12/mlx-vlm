"""CPU-only contract tests for optional fused KDA prefix capture.

These cover the rollback selection logic without constructing a GLM model or
launching Metal.  They deliberately set the CPU device before creating arrays.
The separate Metal-kernel parity gate belongs to a later reviewed run.
"""

from types import SimpleNamespace
from unittest.mock import patch
from math import prod
from hashlib import sha256
import ast
import inspect
import textwrap

import mlx.core as mx
import pytest



@pytest.fixture(autouse=True)
def _cpu_default_device():
    """These selector/contract tests intend the CPU device; pin it per test and
    restore afterwards so importing this module never flips the process-wide
    default device for other collected modules (which compare Metal kernels
    against GPU eager references)."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.glm5_next import language as glm5_language
from mlx_vlm.models.glm5_next import fused_kda


H = D = 1
K = 2
C = 3 * H * D
_DEFAULT_BLOCK_SOURCE_SHA256 = "53ee875c5fe9fae0936db5ed13d1ed1842efb697c8b903ce929f0bef8eab5d9c"


def _array(shape, start=0, dtype=mx.float32):
    return mx.arange(start, start + prod(shape), dtype=dtype).reshape(shape)


def _entry(batch=1, steps=3, *, prefix_states=None, prefix_convs=None):
    final_state = _array((batch, H, D, D), 100)
    final_conv = _array((batch, K - 1, C), 200)
    q = _array((batch, steps, H, D), 1)
    legacy = (
        q,
        q + 10,
        q + 20,
        q + 30,
        _array((batch, steps, H), 40),
        mx.zeros((H, 1)),
        mx.zeros((H, D)),
        mx.zeros((batch, H, D, D)),
        mx.zeros((batch, K - 1 + steps, C)),
        K,
        -5.0,
    )
    if prefix_states is None:
        prefix_states = _array((batch, steps - 1, H, D, D), 300)
    if prefix_convs is None:
        prefix_convs = _array((batch, steps - 1, K - 1, C), 400)
    return legacy + (prefix_states, prefix_convs), final_state, final_conv


def _cache(state, conv):
    cache = ArraysCache(size=2)
    cache[0] = conv
    cache[1] = state
    return cache


def _tolist(value):
    mx.eval(value)
    return value.tolist()


def _owner():
    return SimpleNamespace(
        args=SimpleNamespace(index_kpool=2),
        supports_per_row_speculative_rollback=lambda _caches: True,
    )


def test_feature_is_default_off_and_needs_a_real_sink(monkeypatch):
    monkeypatch.setattr(glm5_language, "_FUSED_KDA_GDN_PREFIX_CAPTURE", False)
    assert not glm5_language._fused_kda_prefix_capture_enabled([], 2, 1)
    assert not glm5_language._fused_kda_prefix_capture_enabled(None, 2, 1)
    monkeypatch.setattr(glm5_language, "_FUSED_KDA_GDN_PREFIX_CAPTURE", True)
    assert glm5_language._fused_kda_prefix_capture_enabled([], 2, 1)
    assert glm5_language._fused_kda_prefix_capture_enabled([], 8, 2)
    assert not glm5_language._fused_kda_prefix_capture_enabled([], 1, 1)
    assert not glm5_language._fused_kda_prefix_capture_enabled([], 9, 1)
    assert not glm5_language._fused_kda_prefix_capture_enabled([], 2, 3)


def test_default_and_capture_block_abis_have_distinct_kernel_names_and_arities():
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return tuple(kwargs["output_shapes"])

    args = (
        mx.zeros((1, 2, 1)),
        mx.zeros((1, 2, 1)),
        mx.zeros((1, 2, 1)),
        mx.zeros((1, 1, 3)),
        mx.zeros((3, 2, 1)),
        mx.zeros((1, 2, 1)),
        mx.zeros((1, 2, 1)),
        mx.zeros((1,)),
        mx.zeros((1,)),
        mx.zeros((1, 1, 1, 1)),
        mx.zeros((1, 2, 1)),
        mx.zeros((1,)),
    )
    kwargs = dict(
        num_heads=1,
        head_dim=1,
        conv_kernel_size=2,
        lower_bound=-5.0,
        norm_eps=1e-5,
    )
    with patch.object(fused_kda, "_kernel", return_value=fake_kernel) as select:
        plain = fused_kda.fused_kda_verify_block(*args, **kwargs)
        captured = fused_kda.fused_kda_verify_block_capture(*args, **kwargs)
    assert [call.args[0] for call in select.call_args_list] == ["block", "block_capture"]
    assert len(plain) == 6
    assert len(captured) == 8
    assert calls[0]["output_shapes"] == calls[1]["output_shapes"][:6]


def test_default_block_source_and_six_output_abi_are_byte_preserved():
    # Hash is calculated from ab269706's literal _BLOCK_SOURCE.  The optional
    # capture program is built separately, so this catches an accidental edit to
    # the default-on Metal source as well as an ABI-arithmetic regression.
    assert sha256(fused_kda._BLOCK_SOURCE.encode()).hexdigest() == _DEFAULT_BLOCK_SOURCE_SHA256
    assert fused_kda._BLOCK_OUTPUT_NAMES == [
        "y",
        "state_out",
        "conv_state_out",
        "q_out",
        "k_out",
        "v_out",
    ]
    assert fused_kda._BLOCK_CAPTURE_OUTPUT_NAMES[:6] == fused_kda._BLOCK_OUTPUT_NAMES
    assert fused_kda._BLOCK_CAPTURE_OUTPUT_NAMES[6:] == ["prefix_state", "prefix_conv"]


def test_default_public_block_function_directly_dispatches_the_default_kernel():
    source = textwrap.dedent(inspect.getsource(fused_kda.fused_kda_verify_block))
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    assert "_fused_kda_verify_block" not in names
    kernel_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_kernel"
    ]
    assert len(kernel_calls) == 1
    assert isinstance(kernel_calls[0].args[0], ast.Constant)
    assert kernel_calls[0].args[0].value == "block"


def test_capture_kernel_is_lazily_built_only_when_its_abi_is_requested(monkeypatch):
    """Preparing ordinary kernels must not compile the experimental ABI."""
    calls = []

    def fake_metal_kernel(**kwargs):
        calls.append(kwargs["name"])
        return object()

    monkeypatch.setattr(fused_kda, "_KERNELS", {})
    monkeypatch.setattr(fused_kda, "_KERNEL_TRIED", False)
    monkeypatch.setattr(fused_kda.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(fused_kda.mx.fast, "metal_kernel", fake_metal_kernel)
    assert fused_kda._kernel("block") is not None
    assert "glm5_kda_verify_block_capture" not in calls
    assert fused_kda._kernel("block_capture") is not None
    assert calls.count("glm5_kda_verify_block_capture") == 1


def test_capture_probe_is_separate_and_exercises_the_s2_capture_launch_shape():
    source = inspect.getsource(fused_kda._probe_launch)
    assert 'steps = 2 if kind == "block_capture" else 1' in source
    assert "fused_kda_verify_block_capture(" in source
    probe_source = inspect.getsource(fused_kda.fused_kda_probe)
    # The key's leading discriminator prevents a base probe from blessing the
    # separate capture pipeline's threadgroup extent.
    assert "key = (\n        kind," in probe_source


def test_language_has_a_distinct_capture_probe_and_safe_legacy_fallback():
    source = inspect.getsource(glm5_language.Glm5NextLinearAttention._fused_kda_block)
    ready = inspect.getsource(
        glm5_language.Glm5NextLinearAttention._fused_kda_prefix_capture_ready
    )
    assert 'kind="block_capture"' in ready
    assert "_fused_kda_prefix_capture_ty" in source
    assert "capture_prefix = False" in source
    assert "fused_kda_verify_block(" in source


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_b1_prefix_selection_uses_prefix_or_existing_final(accepted):
    entry, final_state, final_conv = _entry()
    state, conv = glm5_language._captured_gdn_prefix_state(
        entry, final_state, final_conv, [accepted], 3
    )
    if accepted == 2:
        assert _tolist(state) == _tolist(final_state)
        assert _tolist(conv) == _tolist(final_conv)
    else:
        assert _tolist(state) == _tolist(entry[11][:, accepted])
        assert _tolist(conv) == _tolist(entry[12][:, accepted])


def test_batched_uniform_partial_and_full_use_one_receipt_shape():
    entry, final_state, final_conv = _entry(batch=2)
    partial_state, partial_conv = glm5_language._captured_gdn_prefix_state(
        entry, final_state, final_conv, [1, 1], 3
    )
    assert _tolist(partial_state) == _tolist(entry[11][:, 1])
    assert _tolist(partial_conv) == _tolist(entry[12][:, 1])
    full_state, full_conv = glm5_language._captured_gdn_prefix_state(
        entry, final_state, final_conv, [2, 2], 3
    )
    assert _tolist(full_state) == _tolist(final_state)
    assert _tolist(full_conv) == _tolist(final_conv)


def test_ragged_full_reject_selects_final_and_first_prefix_without_replay():
    entry, final_state, final_conv = _entry(batch=2)
    cache = _cache(final_state, final_conv)
    with patch.object(
        glm5_language,
        "gated_delta_update",
        side_effect=AssertionError("captured prefix must not replay gated_delta"),
    ):
        result = glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [entry], [2, 0], 3
        )
    assert result == 2
    assert _tolist(cache[1][0:1]) == _tolist(final_state[0:1])
    assert _tolist(cache[0][0:1]) == _tolist(final_conv[0:1])
    assert _tolist(cache[1][1:2]) == _tolist(entry[11][1:2, 0])
    assert _tolist(cache[0][1:2]) == _tolist(entry[12][1:2, 0])


def test_masked_row_uses_captured_prefix_value_without_recomputing_it():
    # Row one models a masked token whose post-token state equals its preceding
    # state.  Selection must keep that captured value, not invoke a recurrence.
    prefix_states = _array((2, 2, H, D, D), 300)
    prefix_states = mx.concatenate(
        [prefix_states[0:1], mx.full((1, 2, H, D, D), 777.0)], axis=0
    )
    entry, final_state, final_conv = _entry(batch=2, prefix_states=prefix_states)
    state, _conv = glm5_language._captured_gdn_prefix_state(
        entry, final_state, final_conv, [2, 0], 3
    )
    assert _tolist(state[1:2]) == _tolist(prefix_states[1:2, 0])


def test_legacy_eleven_field_entry_falls_back_to_existing_replay():
    entry, final_state, final_conv = _entry()
    legacy = entry[:11]
    cache = _cache(final_state, final_conv)
    called = []

    def replay(*args, **kwargs):
        called.append((args, kwargs))
        return mx.zeros((1, 2, H, D)), mx.full((1, H, D, D), 999.0)

    with patch.object(glm5_language, "gated_delta_update", side_effect=replay):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [legacy], 1, 3
        )
    assert len(called) == 1
    assert _tolist(cache[1]) == [[[[999.0]]]]


def test_legacy_none_entry_state_preserves_gated_delta_zero_init_contract():
    entry, final_state, final_conv = _entry()
    legacy = entry[:7] + (None,) + entry[8:11]
    cache = _cache(final_state, final_conv)
    called = []

    def replay(*args, **kwargs):
        called.append(kwargs)
        return mx.zeros((1, 2, H, D)), mx.full((1, H, D, D), 999.0)

    with patch.object(glm5_language, "gated_delta_update", side_effect=replay):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [legacy], 1, 3
        )
    assert called[0]["state"] is None


def test_present_but_malformed_prefix_receipt_fails_loudly_before_replay():
    entry, final_state, final_conv = _entry(
        prefix_states=mx.zeros((1, 3, H, D, D)),  # expected S - 1 == 2
    )
    cache = _cache(final_state, final_conv)
    with patch.object(
        glm5_language,
        "gated_delta_update",
        side_effect=AssertionError("malformed capture must not fall back"),
    ):
        with pytest.raises(ValueError, match="state-prefix shape"):
            glm5_language.LanguageModel.rollback_speculative_cache(
                _owner(), [cache], [entry], 1, 3
            )


@pytest.mark.parametrize(
    ("field", "bad_dtype", "message"),
    [
        ("state", mx.float16, "state-prefix dtype"),
        ("conv", mx.float16, "convolution-prefix dtype"),
    ],
)
def test_prefix_dtype_mismatch_fails_loudly_before_legacy_replay(field, bad_dtype, message):
    prefix_states = mx.zeros((1, 2, H, D, D), dtype=mx.float32)
    prefix_convs = mx.zeros((1, 2, K - 1, C), dtype=mx.float32)
    if field == "state":
        prefix_states = prefix_states.astype(bad_dtype)
    else:
        prefix_convs = prefix_convs.astype(bad_dtype)
    entry, final_state, final_conv = _entry(
        prefix_states=prefix_states,
        prefix_convs=prefix_convs,
    )
    cache = _cache(final_state, final_conv)
    with patch.object(
        glm5_language,
        "gated_delta_update",
        side_effect=AssertionError("dtype mismatch must not fall back"),
    ):
        with pytest.raises(ValueError, match=message):
            glm5_language.LanguageModel.rollback_speculative_cache(
                _owner(), [cache], [entry], 1, 3
            )


@pytest.mark.parametrize("receipts", [[], lambda entry: [entry, entry]])
def test_gdn_receipt_count_must_exactly_match_arrays_cache_count(receipts):
    entry, final_state, final_conv = _entry()
    cache = _cache(final_state, final_conv)
    values = receipts(entry) if callable(receipts) else receipts
    with pytest.raises(ValueError, match="count does not match"):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], values, 1, 3
        )


def test_late_malformed_gdn_receipt_leaves_all_kda_and_sparse_caches_untouched():
    """Preflight sees the later failure before the first legacy replay mutates."""
    entry1, state1, conv1 = _entry()
    entry2, state2, conv2 = _entry()
    malformed2 = entry2[:11] + (
        mx.zeros((1, 3, H, D, D)),  # S - 1 is 2, so this is invalid.
        entry2[12],
    )
    cache1, cache2 = _cache(state1, conv1), _cache(state2, conv2)
    before = tuple(_tolist(x) for x in (cache1[0], cache1[1], cache2[0], cache2[1]))
    sparse = SimpleNamespace(marker="must not be trimmed")
    with patch.object(
        glm5_language,
        "gated_delta_update",
        side_effect=AssertionError("preflight must run before legacy replay"),
    ), patch.object(
        glm5_language,
        "trim_sparse_cache",
        side_effect=AssertionError("preflight must run before sparse trim"),
    ):
        with pytest.raises(ValueError, match="state-prefix shape"):
            glm5_language.LanguageModel.rollback_speculative_cache(
                _owner(), [cache1, sparse, cache2], [entry1[:11], malformed2], 1, 3
            )
    after = tuple(_tolist(x) for x in (cache1[0], cache1[1], cache2[0], cache2[1]))
    assert after == before
    assert sparse.marker == "must not be trimmed"


def test_malformed_legacy_schema_refuses_before_any_replay_or_cache_mutation():
    entry, final_state, final_conv = _entry()
    malformed = (_array((1,)),) + entry[1:11]
    cache = _cache(final_state, final_conv)
    before = tuple(_tolist(x) for x in (cache[0], cache[1]))
    with patch.object(
        glm5_language,
        "gated_delta_update",
        side_effect=AssertionError("malformed legacy schema must not replay"),
    ):
        with pytest.raises(ValueError, match="block size does not match"):
            glm5_language.LanguageModel.rollback_speculative_cache(
                _owner(), [cache], [malformed], 1, 3
            )
    assert tuple(_tolist(x) for x in (cache[0], cache[1])) == before


def test_capture_gate_parameter_schema_is_preflighted_before_selection():
    entry, final_state, final_conv = _entry()
    malformed = entry[:5] + (mx.zeros((H + 1, 1)),) + entry[6:]
    cache = _cache(final_state, final_conv)
    with pytest.raises(ValueError, match="gate parameter shapes"):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [malformed], 1, 3
        )


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 8])
def test_capture_activation_dtype_schema_is_fail_loud(index):
    entry, final_state, final_conv = _entry()
    values = list(entry)
    values[index] = values[index].astype(mx.float16)
    cache = _cache(final_state, final_conv)
    with pytest.raises(ValueError, match="activation dtype"):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [tuple(values)], 1, 3
        )


def test_capture_final_convolution_dtype_is_the_activation_dtype_anchor():
    entry, final_state, final_conv = _entry()
    values = list(entry)
    values[12] = values[12].astype(mx.float16)
    cache = _cache(final_state, final_conv.astype(mx.float16))
    with pytest.raises(ValueError, match="activation dtype"):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [tuple(values)], 1, 3
        )


@pytest.mark.parametrize("index", [5, 6])
def test_capture_gate_parameter_dtype_schema_is_fail_loud(index):
    entry, final_state, final_conv = _entry()
    values = list(entry)
    values[index] = values[index].astype(mx.float16)
    cache = _cache(final_state, final_conv)
    with pytest.raises(ValueError, match="gate parameter dtype"):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [tuple(values)], 1, 3
        )


def test_capture_entry_state_dtype_schema_is_fail_loud():
    entry, final_state, final_conv = _entry()
    values = list(entry)
    values[7] = values[7].astype(mx.float16)
    cache = _cache(final_state, final_conv)
    with pytest.raises(ValueError, match="state dtype"):
        glm5_language.LanguageModel.rollback_speculative_cache(
            _owner(), [cache], [tuple(values)], 1, 3
        )
