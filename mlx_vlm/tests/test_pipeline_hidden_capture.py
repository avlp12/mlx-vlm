"""A6: the drafter's context, captured on two boxes and merged into one.

``speculative_hidden_capture`` refused every request with a hidden-reading
drafter attached, and the DEFAULT served config has one (DFlash2).  So after A5
had removed the vault block, this was the remaining reason a served request
could not use the peer -- and it is not a small one: DFlash2's target layers
``[5, 14, 24, 33, 42]`` STRADDLE the shipped split of 23, so neither box can
produce the capture by itself.

What is pinned here, in the order the argument runs:

1.  the trailing-window arithmetic in the tail process is the SAME arithmetic
    ``PrefillHiddenAccumulator`` performs, on random data, at every chunk/window
    ratio that matters (the tail cannot import the speculative package, so the
    code is duplicated and the duplication is measured);
2.  adopting a merged window and then appending the remainder forward's capture
    reproduces ``finish()`` -- arrays AND the ``target_hidden_offset`` the
    drafter's RoPE positions depend on -- bit for bit;
3.  over a real socket, with the real head and the real tail and a capture set
    that straddles the split, the merged window IS the single-box capture;
4.  through the served call site, the drafter's round-1 context and the prompt
    cache come out bit-equal to the single-box run, with the A5 vault collapse
    happening on the same request;
5.  a tail that does not return the window costs the request a single-box
    re-prefill and nothing else -- the answer is slower, never different;
6.  the refusals that remain have names, and the feature-off path is untouched.
"""

import hashlib
import socket
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm import pipeline_prefill as pp
from mlx_vlm import pipeline_runtime as pr
from mlx_vlm.generate.ar import PromptProcessingBatch, _left_pad_prompts
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel
from mlx_vlm.pipeline_prefill import CaptureSpec, CaptureUnsupported, TrailingHiddenWindow
from mlx_vlm.pipeline_runtime import PipelineHead, PipelineSettings
from mlx_vlm.speculative.utils import PrefillHiddenAccumulator

STEP = 8
PROMPT = list(range(3, 43))  # 40 tokens -> 4 pipelined chunks of 8, 8 left over
PIPELINED_CHUNKS = [STEP] * 4
DEPTH = sum(PIPELINED_CHUNKS)
SPLIT = 1  # the tiny model has two layers: KDA on the head, DSA on the tail
KEEP = 15

ON_GPU = mx.default_device() == mx.gpu


# A11.  Same clamp artifact as ``test_pipeline_batch_prefill.py`` (see the note
# there): at C = 8 the shipped ``TAIL_MIN`` of 1024 is clamped to the step, so
# L35(b)'s merge window covers this file's 40-token prompt.  Pin the served
# RATIO (1024/8192 = C/8, rounded up to 4 here so the window is still wide
# enough for the merge tests to aim at) instead of the absolute constant.
TAIL_MIN = 4


@pytest.fixture(autouse=True)
def _served_tail_min_ratio(monkeypatch):
    monkeypatch.setenv("MLX_VLM_GLM5_PREFILL_TAIL_MIN", str(TAIL_MIN))


@pytest.fixture(autouse=True)
def _clean_metrics():
    pr.METRICS.reset()
    pr.POOL.breaker.reset()
    yield
    pr.METRICS.reset()
    pr.POOL.breaker.reset()


def _tiny_text_config():
    return TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=0,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_head_dim=64,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        index_topk=6,
        index_head_dim=64,
        index_n_heads=2,
        index_kpool=3,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 64,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        eos_token_id=1,
    )


def _lm():
    mx.random.seed(0)
    lm = LanguageModel(_tiny_text_config())
    lm.eval()
    mx.eval(lm.parameters())
    return lm


class _StubDFlash:
    """The smallest object ``speculative/utils.py`` accepts as a dflash drafter.

    ``target_layer_ids`` is the knob under test: with the split at 1, ``[0, 1]``
    straddles it exactly as DFlash2's ``[5, 14, 24, 33, 42]`` straddles 23.
    """

    def __init__(self, layer_ids=(0, 1), keep=KEEP):
        self.config = SimpleNamespace(target_layer_ids=list(layer_ids))
        self._keep = keep
        self.adopted = []

    def prefill_context_keep(self):
        return self._keep

    def adopt_pretruncated_context(self, cache, skip):
        self.adopted.append(skip)


class _StubMTP:
    """MTP has no declared window of its own; ``mtp_prime_window()`` chooses one."""


def _cache_arrays(prompt_cache):
    out = []
    for entry in prompt_cache:
        stack = [entry.state]
        while stack:
            item = stack.pop()
            if isinstance(item, mx.array):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
    return out


def _digest(arrays):
    arrays = list(arrays)
    mx.eval(arrays)
    h = hashlib.sha256()
    for a in arrays:
        h.update(repr(tuple(a.shape)).encode())
        h.update(memoryview(np.asarray(a.astype(mx.float32))).tobytes())
    return h.hexdigest()


# ------------------------------------------------- 1. the window is the window


def _random_chunks(n_layers, widths, dim=7, seed=0):
    mx.random.seed(seed)
    return [
        [mx.random.normal((1, w, dim)).astype(mx.bfloat16) for _ in range(n_layers)]
        for w in widths
    ]


@pytest.mark.parametrize(
    "keep,widths",
    [
        (15, [8, 8, 8, 8]),   # the fixture's shape: window inside one chunk
        (2047, [2048] * 3),   # DFlash2 at the shipped chunk size
        (2048, [2048] * 3),   # MTP's window, exactly one chunk wide
        (40, [8, 8, 8, 8]),   # a window wider than everything captured
        (9, [4, 4, 4, 4, 4]), # a window that spans three chunk pieces
        (15, [8]),            # one chunk only
        # A11b: the LAST chunk may be wider than the ones before it (L35's tail
        # merge, up to C + tail_min - 1).  The window is width-agnostic by
        # construction -- it keeps per-piece widths -- and this measures it
        # rather than trusting the construction, at the fixture ratio and at the
        # served one.
        (15, [8, 8, 11]),
        (2047, [8192, 9215]),
    ],
)
def test_the_tails_window_is_the_accumulators_window(keep, widths):
    """Two implementations of one arithmetic, compared on random data.

    ``TrailingHiddenWindow`` runs in the tail process, which loads no drafter and
    must not import the speculative package to serve a prefill, so the pruning
    and trimming rules exist twice.  They are the same rules or the merged
    context is not the single-box context, which is what this measures instead
    of asserting.
    """
    chunks = _random_chunks(2, widths)
    window = TrailingHiddenWindow(keep)
    accumulator = PrefillHiddenAccumulator(keep=keep)
    for piece in chunks:
        window.append(piece)
        accumulator.append_layers(piece)
    got = window.window()
    want, offset = accumulator.finish()
    assert window.total_rows == accumulator.total_rows == sum(widths)
    assert window.dropped_rows == accumulator.dropped_rows
    assert len(got) == len(want) == 2
    for a, b in zip(got, want):
        assert a.shape == b.shape == (1, min(keep, sum(widths)), 7)
        assert mx.array_equal(a, b)
    # and it really is the trailing rows of the whole capture
    for layer in range(2):
        whole = mx.concatenate([c[layer] for c in chunks], axis=1)
        assert mx.array_equal(got[layer], whole[:, -min(keep, sum(widths)) :])
    # the offset the drafter is owed is the same count read off either one
    assert offset == max(0, sum(widths) - keep)


@pytest.mark.parametrize(
    "keep,widths,remainder",
    [
        (15, [8, 8, 8, 8], 8),
        (2047, [2048] * 3, 2048),
        (2047, [2048] * 3, 1),
        (2048, [2048] * 3, 512),
        (40, [8, 8, 8, 8], 8),
        (9, [4, 4, 4, 4], 4),
        # A11b: the peer runs the merged chunk and the head's remainder is the
        # single last token, which is the shape every ``2 <= r <= tail_min``
        # prompt now has.
        (15, [8, 8, 11], 1),
        (2047, [8192, 8192, 8192, 8715], 1),
    ],
)
def test_adopting_a_merged_window_reproduces_finish(keep, widths, remainder):
    """The load-bearing arithmetic of A6, stated as an equality.

    One box appends ``k`` chunk captures and then the remainder forward's.  Two
    boxes append only the remainder, on top of a window that stands for those
    ``k`` chunks.  ``finish()`` must not be able to tell the difference -- not in
    the arrays and not in the offset, which is the drafter's RoPE origin::

        single box   dropped = k*C - resident,  skip = resident + r - keep
        two box      dropped = k*C - W,         skip = W + r - keep
                     => dropped + skip = k*C + r - keep, both ways.
    """
    chunks = _random_chunks(3, widths, seed=1)
    tail = _random_chunks(3, [remainder], seed=2)[0]

    one_box = PrefillHiddenAccumulator(keep=keep)
    for piece in chunks:
        one_box.append_layers(piece)
    one_box.append_layers(tail)
    want, want_offset = one_box.finish()

    window = TrailingHiddenWindow(keep)
    for piece in chunks:
        window.append(piece)
    two_box = PrefillHiddenAccumulator(keep=keep)
    two_box.adopt_window(window.window(), rows_covered=sum(widths))
    two_box.append_layers(tail)
    got, got_offset = two_box.finish()

    assert got_offset == want_offset == max(0, sum(widths) + remainder - keep)
    assert len(got) == len(want) == 3
    for a, b in zip(got, want):
        assert a.shape == b.shape
        assert mx.array_equal(a, b), "the two-box context is not the one-box context"


def test_adopt_window_refuses_what_it_cannot_account_for():
    acc = PrefillHiddenAccumulator(keep=15)
    piece = _random_chunks(2, [15], seed=3)[0]
    with pytest.raises(RuntimeError, match="empty hidden window"):
        acc.adopt_window([], rows_covered=32)
    # a window narrower than the trim would have kept: the drafter would be
    # primed on fewer rows than the prompt can supply, silently
    with pytest.raises(RuntimeError, match="expected 20"):
        PrefillHiddenAccumulator(keep=20).adopt_window(piece, rows_covered=32)
    # ... and one that claims fewer rows than it carries
    with pytest.raises(RuntimeError, match="expected 10"):
        PrefillHiddenAccumulator(keep=15).adopt_window(piece, rows_covered=10)
    acc2 = PrefillHiddenAccumulator(keep=15)
    acc2.adopt_window(piece, rows_covered=32)
    with pytest.raises(RuntimeError, match="already holds chunks"):
        acc2.adopt_window(piece, rows_covered=32)
    ragged = [piece[0], piece[1][:, :3]]
    with pytest.raises(RuntimeError, match="disagree on length"):
        PrefillHiddenAccumulator(keep=15).adopt_window(ragged, rows_covered=32)


# --------------------------------------- 2. the spec both boxes have to agree on


@pytest.mark.parametrize(
    "obj,why",
    [
        ({"schema": 2, "kind": "layers", "layers": [0], "keep": 4}, "schema"),
        ({"schema": 1, "kind": "gdn", "layers": [0], "keep": 4}, "kind"),
        ({"schema": 1, "kind": "layers", "layers": [], "keep": 4}, "names no layers"),
        ({"schema": 1, "kind": "hidden", "layers": [0], "keep": 4}, "names layers"),
        ({"schema": 1, "kind": "layers", "layers": [1, 0], "keep": 4}, "ascending"),
        ({"schema": 1, "kind": "layers", "layers": [0, 0], "keep": 4}, "ascending"),
        ({"schema": 1, "kind": "layers", "layers": [2], "keep": 4}, "outside the stack"),
        ({"schema": 1, "kind": "layers", "layers": [0], "keep": 0}, "window"),
        ({"schema": 1, "kind": "layers", "layers": [0], "keep": 1 << 30}, "window"),
        ({"schema": 1, "kind": "layers", "layers": [0], "keep": 4, "x": 1}, "invalid"),
        ("layers", "invalid"),
    ],
)
def test_a_capture_the_peer_cannot_serve_is_refused_by_name(obj, why):
    with pytest.raises(CaptureUnsupported, match=why):
        CaptureSpec.parse(obj, n_layers=2)


def test_the_split_decides_which_box_owns_which_layer():
    spec = CaptureSpec.parse(
        {"schema": 1, "kind": "layers", "layers": [5, 14, 24, 33, 42], "keep": 2047},
        n_layers=45,
    )
    # DFlash2's own set at the shipped split: this is the case the whole design
    # exists for -- neither half can produce the capture alone.
    assert spec.head_layers(23) == [5, 14]
    assert spec.tail_layers(23) == [24, 33, 42]
    assert spec.tail_tensors(23) == 3
    mtp = CaptureSpec.parse(
        {"schema": 1, "kind": "hidden", "layers": [], "keep": 2048}, n_layers=45
    )
    assert mtp.head_layers(23) == [] and mtp.tail_tensors(23) == 1


def test_the_shipped_windows_are_the_sizes_the_receipts_claim():
    """The bytes one request puts on the wire, per drafter, at 4096 features."""
    mb = 4096 * 2 / 2 ** 20
    assert round(3 * 2047 * mb, 1) == 48.0  # DFlash2's three tail-side layers
    assert round(5 * 2047 * mb, 1) == 80.0  # all five, if the split moved
    assert round(1 * 2048 * mb, 1) == 16.0  # MTP's single hidden


# ----------------------------------------- 3. two real boxes, one merged window


def _bf16_lm():
    """The fixture model in the dtype a served model is.

    The boundary tensor is bf16 by contract (``prefill_chunk`` refuses anything
    else), so the two-box rails cannot run the float32 fixture the batch-path
    tests use.  The MoE layer goes with it: ``GatherMM`` is float32-only on CPU,
    and the routed expert MLP is not what a layer SPLIT is about -- what crosses
    the wire is the decoder layer's output, and both dense and sparse layers
    produce the same shape of it.
    """
    cfg = _tiny_text_config()
    cfg.mlp_layer_types = ["dense", "dense"]
    mx.random.seed(0)
    lm = LanguageModel(cfg)
    lm.eval()
    lm.set_dtype(mx.bfloat16)
    mx.eval(lm.parameters())
    return lm


def _single_box_capture(lm, ids, *, capture_kwargs, keep, chunks):
    """What one box's chunk loop captures over the same schedule."""
    cache = lm.make_cache()
    acc = PrefillHiddenAccumulator(keep=keep)
    start = 0
    for n in chunks:
        out = lm(ids[:, start : start + n], cache=cache, **capture_kwargs)
        acc.append(out)
        mx.eval(acc.pending() + [c.state for c in cache])
        start += n
    return acc, cache


def _run_two_box(monkeypatch, lm, ids, *, capture, chunks, drop_capture=False):
    """A real ``PipelineHead`` against a real tail, in one process, over TCP."""
    n_layers = 2
    tail_caches = lm.make_cache()
    for i in range(SPLIT):
        tail_caches[i] = None
    tail_model = SimpleNamespace(language_model=lm)
    monkeypatch.setattr(
        pp,
        "load_stage",
        lambda *a: (tail_model, tail_caches, list(range(SPLIT, n_layers)), n_layers, 0.0),
    )
    if drop_capture:
        # A tail that acknowledges the capture and then does not send it: the
        # one failure the head cannot detect from the ack alone.
        monkeypatch.setattr(pp, "capture_meta", lambda *a, **k: None)
        monkeypatch.setattr(pp, "capture_send", lambda sock, window: 0)
    reservation = socket.socket()
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
    reservation.close()
    args = SimpleNamespace(
        model="unused", split=SPLIT, layers=n_layers, prune=False, bind="127.0.0.1",
        port=port, model_sha256="a" * 64, source_revision="b" * 40,
        connect_timeout=5.0, io_timeout=10.0, depth=2, transport="socket", once=True,
    )
    errors = []

    def tail():
        try:
            pp.run_tail(args)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=tail, daemon=True)
    th.start()
    head = PipelineHead(
        PipelineSettings(
            ("127.0.0.1", port), None, str(SPLIT), 1, "unused", "socket",
            model_sha256="a" * 64, source_revision="b" * 40, io_timeout=10.0,
        ),
        SPLIT,
        n_layers,
    )
    cache = lm.make_cache()
    try:
        head.connect(timeout=5)
        head.begin(
            int(ids.shape[1]),
            chunks[0],
            input_ids=ids,
            capture=capture,
            # A11b: the plan, explicitly.  Equal to the envelope's own uniform
            # derivation on every schedule of C-wide chunks, and the only way to
            # say "the last one is the MERGED one" on the schedules that have one.
            chunks=list(chunks),
        )
        start = 0
        for n in chunks:
            head.prefill_chunk(lm, ids[:, start : start + n], None, cache)
            start += n
        stats = head.finalize(cache)
        hidden = head.take_hidden()
        head.close()
    finally:
        head.abort()
        th.join(5)
    assert not errors, errors
    return hidden, stats, cache


@pytest.mark.parametrize(
    "layer_ids,head_side,tail_side",
    [
        ((0, 1), 1, 1),  # STRADDLES the split: the case DFlash2 is
        ((1,), 0, 1),    # tail only
        ((0,), 1, 0),    # head only -- the tail still has to say "nothing"
    ],
)
def test_the_merged_window_is_the_single_box_capture(
    monkeypatch, layer_ids, head_side, tail_side
):
    """Two boxes, one socket, one capture: bit-identical to one box's.

    The head keeps its own layers' rows across chunks, the tail keeps its own
    and returns them once with ``done``, and the merge is the concatenation of
    the two ordered lists.  If either half captured at the wrong moment, in the
    wrong order, or over the wrong rows, this is where it shows.
    """
    lm = _bf16_lm()
    ids = mx.array([PROMPT[:24]], dtype=mx.int32)
    chunks = [STEP, STEP, STEP - 1]  # begin() hashes ids[:, :-1]
    spec = {"schema": 1, "kind": "layers", "layers": list(layer_ids), "keep": KEEP}
    hidden, stats, cache = _run_two_box(monkeypatch, lm, ids, capture=spec, chunks=chunks)

    ref, ref_cache = _single_box_capture(
        _bf16_lm(),
        ids[:, :-1],
        capture_kwargs={"capture_layer_ids": list(layer_ids)},
        keep=KEEP,
        chunks=chunks,
    )
    want, _ = ref.finish()
    assert len(hidden) == len(want) == len(layer_ids)
    for got, expect in zip(hidden, want):
        assert got.shape == expect.shape == (1, KEEP, 128)
        assert mx.array_equal(got, expect), "the merged window is not the capture"
    # the receipts, and the two halves each did their share
    assert stats["capture"]["capture_head_layers"] == head_side
    assert stats["capture"]["capture_tensors"] == tail_side
    assert stats["capture"]["capture_bytes"] == tail_side * KEEP * 128 * 2
    assert stats["capture"]["capture_rows_covered"] == sum(chunks)
    # and the prefill it rode on is still the prefill
    assert _digest(_cache_arrays(cache)) == _digest(_cache_arrays(ref_cache))


def test_a_chunk_wider_than_c_crosses_the_wire_and_captures(monkeypatch):
    """A11b, on two real boxes: the LAST chunk is ``C + tail_min - 1`` wide.

    Everything that sizes itself off a chunk has to take it: the head's
    envelope (``chunks=``), the boundary header the sender writes, the tail's
    receiver -- which validates ``S`` against ``envelope.chunks[idx]`` and
    allocates ``B*S*HC*D*2`` bytes for it -- the tail's own forward over layers
    ``[split, n)``, and both halves' capture windows.  The reference is one box
    running the SAME two chunks, because that is what the plan promises: the
    peer runs the loop's chunks, merged one included.
    """
    lm = _bf16_lm()
    chunks = [STEP, STEP + TAIL_MIN - 1]  # 8, 11: the widest the merge can make
    ids = mx.array([PROMPT[: sum(chunks) + 1]], dtype=mx.int32)
    spec = {"schema": 1, "kind": "layers", "layers": [0, 1], "keep": KEEP}
    hidden, stats, cache = _run_two_box(
        monkeypatch, lm, ids, capture=spec, chunks=chunks
    )
    assert list(stats["envelope"]["chunks"]) == chunks
    assert stats["envelope"]["depth"] == sum(chunks) == int(ids.shape[1]) - 1

    ref, ref_cache = _single_box_capture(
        _bf16_lm(),
        ids[:, :-1],
        capture_kwargs={"capture_layer_ids": [0, 1]},
        keep=KEEP,
        chunks=chunks,
    )
    want, _ = ref.finish()
    assert len(hidden) == len(want) == 2
    for got, expect in zip(hidden, want):
        assert got.shape == expect.shape == (1, KEEP, 128)
        assert mx.array_equal(got, expect)
    assert _digest(_cache_arrays(cache)) == _digest(_cache_arrays(ref_cache))


def test_the_mtp_whole_hidden_capture_comes_back_from_the_tail(monkeypatch):
    """MTP reads the pre-final-norm hidden after the LAST layer -- tail only."""
    lm = _bf16_lm()
    ids = mx.array([PROMPT[:24]], dtype=mx.int32)
    chunks = [STEP, STEP, STEP - 1]
    spec = {"schema": 1, "kind": "hidden", "layers": [], "keep": KEEP}
    hidden, stats, _ = _run_two_box(monkeypatch, lm, ids, capture=spec, chunks=chunks)

    ref, _ = _single_box_capture(
        _bf16_lm(), ids[:, :-1], capture_kwargs={"return_hidden": True},
        keep=KEEP, chunks=chunks,
    )
    want, _ = ref.finish()
    assert len(hidden) == len(want) == 1
    assert hidden[0].shape == (1, KEEP, 128)
    assert mx.array_equal(hidden[0], want[0])
    assert stats["capture"]["capture_head_layers"] == 0


def test_a_tail_that_does_not_return_the_window_fails_the_request(monkeypatch):
    """Before ``install_state``, so the fallback throws away nothing it needs."""
    lm = _bf16_lm()
    ids = mx.array([PROMPT[:24]], dtype=mx.int32)
    spec = {"schema": 1, "kind": "layers", "layers": [0, 1], "keep": KEEP}
    with pytest.raises(ValueError, match="capture"):
        _run_two_box(
            monkeypatch, lm, ids, capture=spec, chunks=[STEP, STEP, STEP - 1],
            drop_capture=True,
        )


def test_a_tail_that_cannot_serve_the_capture_says_so_before_the_first_chunk(
    monkeypatch,
):
    """The refusal is a round trip, not a prefill.

    A capture the tail cannot parse is refused at the ack, where the head has
    sent no chunk yet and a single-box prefill still costs the request nothing
    but the dial.
    """
    lm = _bf16_lm()
    ids = mx.array([PROMPT[:24]], dtype=mx.int32)
    seen = []

    class _RefusingSpec:
        """A tail whose rail cannot serve this request's capture.

        Patched on the TAIL's module only (``pipeline_runtime`` bound the class
        at import), so the head builds a perfectly good spec and learns the
        refusal the way a real head does: from the ack.
        """

        @staticmethod
        def parse(obj, *, n_layers):
            seen.append(obj)
            raise CaptureUnsupported("no drafter rail on this tail")

    monkeypatch.setattr(pp, "CaptureSpec", _RefusingSpec)
    spec = {"schema": 1, "kind": "layers", "layers": [0, 1], "keep": KEEP}
    with pytest.raises(ValueError, match="capture_unsupported"):
        _run_two_box(monkeypatch, lm, ids, capture=spec, chunks=[STEP, STEP, STEP - 1])
    assert seen, "the tail parsed the request before acknowledging it"


# ------------------------------------------ 4. the served call site, end to end


class _SplitLoopbackHead:
    """A peer that is this box, split at the same layer the real one is.

    Unlike the A4/A5 double this one does NOT run ``model(...)``: it runs the
    two halves through ``pipeline_prefill_head`` and ``pipeline_forward`` and
    captures each half's own layers, which is exactly what a head and a tail do
    between them.  So an admitted run has to land on the single-box cache AND on
    the single-box drafter context, and the second one is new here.
    """

    def __init__(self, settings=None, split=SPLIT, n_layers=2, fail_at=None,
                 lose_capture=False):
        self.settings = settings
        self.split = split
        self.n_layers = n_layers
        self.sock = object()
        self.stats = {"wire_send_s": 0.0,
                      "handoff": {"handoff_bytes": 7, "handoff_wire_recv_s": 0.5}}
        self.calls = []
        self.chunks = []
        self.fail_at = fail_at
        self.lose_capture = lose_capture
        self.spec = None

    def connect(self):
        return self

    def ping(self):
        return True

    def abort(self):
        self.calls.append("abort")
        self.sock = None

    def begin(self, tokens, chunk, *, input_ids, capture=None, chunks=None):
        self.calls.append(("begin", int(tokens), int(chunk)))
        self.spec = (
            CaptureSpec.parse(capture, n_layers=self.n_layers)
            if capture is not None
            else None
        )
        self.head_ids = self.spec.head_layers(self.split) if self.spec else []
        self.tail_ids = self.spec.tail_layers(self.split) if self.spec else []
        self.head_window = (
            TrailingHiddenWindow(self.spec.keep) if self.head_ids else None
        )
        self.tail_window = (
            TrailingHiddenWindow(self.spec.keep)
            if self.spec is not None
            and (self.tail_ids or self.spec.kind == "hidden")
            else None
        )

    def local_caches(self, cache):
        return [c for c in cache[: self.split] if c is not None]

    def prefill_chunk(self, model, input_ids, inputs_embeds, cache):
        idx = len(self.chunks)
        if self.fail_at is not None and idx == self.fail_at:
            raise OSError("the tail went away mid-chunk")
        self.chunks.append(int(input_ids.shape[1]))
        self.calls.append(("prefill_chunk", int(input_ids.shape[1])))
        head_sink = [] if self.head_window is not None else None
        h = model.pipeline_prefill_head(
            inputs=input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            split=self.split,
            hidden_sink=head_sink,
            capture_layer_ids=self.head_ids,
        )
        tail_sink = [] if self.tail_window is not None else None
        kwargs = {} if tail_sink is None else dict(
            hidden_sink=tail_sink, capture_layer_ids=self.tail_ids
        )
        model.model.pipeline_forward(
            h, cache, self.split, self.n_layers, **kwargs
        )
        mx.eval([c.state for c in cache] + (head_sink or []) + (tail_sink or []))
        if self.head_window is not None:
            self.head_window.append(head_sink)
        if self.tail_window is not None:
            self.tail_window.append(tail_sink)

    def finalize(self, cache):
        if self.lose_capture:
            # Where the real head refuses it: ``PipelineHead.finalize`` reads
            # the window off ``done`` BEFORE ``install_state``, so a peer that
            # did not send one fails the request while the fallback still costs
            # only a re-prefill (pinned against a real tail in section 3).
            raise ValueError("pipeline peer returned no capture")
        self.calls.append("finalize")
        return self.stats

    def take_hidden(self):
        head = self.head_window.window() if self.head_window is not None else []
        tail = self.tail_window.window() if self.tail_window is not None else []
        return head + tail

    def close(self):
        self.calls.append("bye")
        self.sock = None


def _arm(monkeypatch, *, min_tokens=16, fail_at=None, lose_capture=False,
         hosts="127.0.0.1:39210"):
    made = []

    class Factory(_SplitLoopbackHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers, fail_at=fail_at,
                             lose_capture=lose_capture)
            made.append(self)

    monkeypatch.setattr(pr, "PipelineHead", Factory)
    monkeypatch.setattr(pr, "POOL", pr.PipelinePool())
    if hosts:
        monkeypatch.setenv("MLX_VLM_PIPELINE_HOSTS", hosts)
    else:
        monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    monkeypatch.setenv("MLX_VLM_PIPELINE_SPLIT", str(SPLIT))
    monkeypatch.setenv("MLX_VLM_PIPELINE_MIN_TOKENS", str(min_tokens))
    monkeypatch.setenv("MLX_VLM_PIPELINE_MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("MLX_VLM_PIPELINE_SOURCE_REVISION", "b" * 40)
    return made


def _spec_batch(lm, *, drafter, kind, rows=None, vault=None, rungs=None):
    rows = rows or [PROMPT]
    padded = _left_pad_prompts(rows)
    batch = PromptProcessingBatch(
        model=lm,
        uids=list(range(len(rows))),
        input_ids=rows,
        max_tokens=[4] * len(rows),
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=STEP,
        draft_model=drafter,
        draft_kind=kind,
    )
    if vault is not None:
        batch._vault = vault
        batch._apc_meta = [
            {"prefix_len": 0, "vault_rungs": list(rungs or []),
             "full_input_ids": list(rows[0])}
        ]
    return batch


def _drain_spec(batch):
    """Prefill to the handoff and return what the drafter is handed."""
    steps = []
    while batch.needs_processing():
        steps.append(batch.prompt_step())
    gen = batch.generate(
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        stop_criteria=lambda token: False,
    )
    return {
        "cache": _digest(_cache_arrays(gen.prompt_cache)),
        "hidden": gen.hidden,
        "offset": batch.target_hidden_offset,
        "steps": steps,
    }


def _single_box_spec(monkeypatch, **kw):
    monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    return _drain_spec(_spec_batch(_lm(), **kw))


@pytest.mark.parametrize("layer_ids", [(0, 1), (1,), (0,)])
def test_the_default_served_config_now_reaches_the_peer(monkeypatch, layer_ids):
    """The whole point of A6: a DFlash-attached request is ADMITTED.

    And admitted without paying for it -- the prompt cache, the drafter's
    round-1 context and its RoPE origin all come out bit-equal to the run that
    stayed on one box.
    """
    made = _arm(monkeypatch)
    got = _drain_spec(_spec_batch(_lm(), drafter=_StubDFlash(layer_ids), kind="dflash"))
    assert got["steps"] == PIPELINED_CHUNKS
    assert made[0].calls[-1] == "finalize", "the request was served two-box"
    assert pr.METRICS.snapshot()["pp_used"] == 1
    assert pr.METRICS.snapshot()["pp_bypass_reason"] == {}

    want = _single_box_spec(
        monkeypatch, drafter=_StubDFlash(layer_ids), kind="dflash"
    )
    assert got["cache"] == want["cache"], "the two-box prefill moved the cache"
    assert got["offset"] == want["offset"] == len(PROMPT) - KEEP
    assert got["hidden"].shape == want["hidden"].shape
    assert got["hidden"].shape[1] == KEEP
    assert mx.array_equal(got["hidden"], want["hidden"]), (
        "the drafter's context is not the single-box context"
    )


def test_the_drafter_context_survives_a_merged_pipelined_chunk(monkeypatch):
    """A11b x A6: the peer runs the MERGED chunk, and the stitch still holds.

    ``T = 42 = 5*8 + 2`` puts the last cell inside L35's merge window, so the
    plan is ``[8, 8, 8, 8, 9]`` and the head keeps exactly one token.  That is
    the geometry A11 refused to send (it gave the fifth chunk back), and it is
    the one where the adopted window has to stand for a chunk WIDER than C:
    ``adopt_window(rows_covered=41)`` then one remainder forward, against a
    single box that appended all five chunks itself.  The drafter's array and
    its RoPE origin are the same either way, or the two-box answer is a fluent
    wrong one.
    """
    rows = [list(range(3, 45))]
    made = _arm(monkeypatch)
    got = _drain_spec(
        _spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash", rows=rows)
    )
    assert made[0].chunks == [STEP] * 4 + [STEP + 1], "the merged chunk is the peer's"
    assert got["steps"] == [STEP] * 4 + [STEP + 1]
    assert pr.METRICS.snapshot()["pp_schedule_shortened"] == 0

    want = _single_box_spec(monkeypatch, drafter=_StubDFlash(), kind="dflash", rows=rows)
    assert got["cache"] == want["cache"]
    assert got["offset"] == want["offset"] == len(rows[0]) - KEEP
    assert mx.array_equal(got["hidden"], want["hidden"])


def test_the_mtp_served_request_reaches_the_peer_too(monkeypatch):
    """MTP's window is ``mtp_prime_window()`` and its capture is tail-only."""
    _arm(monkeypatch)
    got = _drain_spec(_spec_batch(_lm(), drafter=_StubMTP(), kind="mtp"))
    assert got["steps"] == PIPELINED_CHUNKS
    assert pr.METRICS.snapshot()["pp_used"] == 1
    want = _single_box_spec(monkeypatch, drafter=_StubMTP(), kind="mtp")
    assert got["cache"] == want["cache"]
    assert got["offset"] == want["offset"]
    assert mx.array_equal(got["hidden"], want["hidden"])


def test_the_capture_and_the_vault_collapse_ride_the_same_request(monkeypatch):
    """A5 and A6 on one prefill: the default served config is both at once."""
    from mlx_vlm.tests.test_pipeline_batch_prefill import _FakeVault

    _arm(monkeypatch)
    vault = _FakeVault()
    got = _drain_spec(
        _spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash", vault=vault,
                    rungs=[STEP * 2, DEPTH])
    )
    assert vault.depths() == [DEPTH], "the collapsed rung, at full depth"
    snap = pr.METRICS.snapshot()
    assert snap["pp_used"] == 1 and snap["pp_ladder_collapsed"] == 1
    assert snap["pp_bypass_reason"] == {}

    ref_vault = _FakeVault()
    want = _single_box_spec(
        monkeypatch, drafter=_StubDFlash(), kind="dflash", vault=ref_vault,
        rungs=[DEPTH],
    )
    assert got["cache"] == want["cache"]
    assert mx.array_equal(got["hidden"], want["hidden"])


def test_a_peer_that_loses_the_window_costs_a_re_prefill_and_nothing_else(
    monkeypatch,
):
    """The failure this rail exists for.

    A capture that does not come back cannot be papered over: the drafter would
    be primed on the remainder forward alone -- a fluent worse answer, arrived
    at silently.  So it fails the request into the single-box path, which
    reproduces the never-tried run bit for bit.
    """
    made = _arm(monkeypatch, lose_capture=True)
    got = _drain_spec(_spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash"))
    assert made[0].calls[-1] == "abort" and "finalize" not in made[0].calls
    # the first three chunks return normally; the fourth is the one that
    # finalizes, so it is the one that fails and the whole prompt runs again
    assert got["steps"] == PIPELINED_CHUNKS[:3] + PIPELINED_CHUNKS
    snap = pr.METRICS.snapshot()
    assert snap["pp_failed"] == 1 and snap["pp_used"] == 0

    want = _single_box_spec(monkeypatch, drafter=_StubDFlash(), kind="dflash")
    assert got["cache"] == want["cache"]
    assert got["offset"] == want["offset"]
    assert mx.array_equal(got["hidden"], want["hidden"])


def test_the_batch_refuses_an_empty_window_even_if_finalize_did_not(monkeypatch):
    """Belt and braces, and the braces are the ones that matter.

    ``finalize`` refuses a window that did not arrive; this is the second guard,
    at the point of USE, so a future head that returned an empty list from
    ``take_hidden`` could not quietly prime the drafter on the remainder forward
    alone.
    """
    batch = _spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash")
    with pytest.raises(RuntimeError, match="no speculative capture"):
        batch._pipeline_adopt_capture(None, DEPTH)
    with pytest.raises(RuntimeError, match="no speculative capture"):
        batch._pipeline_adopt_capture([], DEPTH)


def test_a_peer_that_dies_mid_prefill_still_gets_the_context_right(monkeypatch):
    _arm(monkeypatch, fail_at=2)
    got = _drain_spec(_spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash"))
    assert got["steps"] == PIPELINED_CHUNKS[:2] + PIPELINED_CHUNKS
    want = _single_box_spec(monkeypatch, drafter=_StubDFlash(), kind="dflash")
    assert got["cache"] == want["cache"]
    assert mx.array_equal(got["hidden"], want["hidden"])


def test_the_window_is_counted_in_the_bytes_the_link_carried(monkeypatch):
    """``pp_handoff_bytes`` is what came back, not part of what came back."""
    _arm(monkeypatch)
    _drain_spec(_spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash"))
    # the double reports 7 handoff bytes; the capture is real
    assert pr.METRICS.snapshot()["pp_handoff_bytes"] == 7


# --------------------------------------------------- 5. what is still refused


def test_a_drafter_with_no_finite_window_is_still_refused(monkeypatch):
    """``MLX_VLM_SPEC_PREFILL_CTX_TRIM=0``: the window is the whole prompt.

    Not a correctness problem -- it is 3.2 GB on the wire for a 131k prompt
    against 50.3 MB for the trailing 2047 rows, so the request stays on the box
    that can answer it without the link.
    """
    monkeypatch.setenv("MLX_VLM_SPEC_PREFILL_CTX_TRIM", "0")
    _arm(monkeypatch)
    batch = _spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash")
    assert batch._prefill_hidden.keep is None
    batch._pipeline_open()
    assert batch._pipeline is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"] == {
        "speculative_hidden_capture": 1
    }


def test_a_capture_shape_this_rail_has_no_merge_for_is_refused_by_name(monkeypatch):
    """``capture_unsupported`` is additive and, today, unreachable by design.

    ``chunk_capture_kwargs_for`` emits exactly two shapes and both are served.
    A third one added later must fall out of the pipeline by NAME rather than be
    handed a merge that was written for two, so the refusal is wired now and
    exercised with a capture kwarg the accumulator would carry but this rail
    cannot place.
    """
    _arm(monkeypatch)
    batch = _spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash")
    batch._chunk_capture_kwargs = {"capture_gdn_states": True}
    batch._pipeline_open()
    assert batch._pipeline is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"] == {"capture_unsupported": 1}


def test_the_ladder_is_still_asked_first(monkeypatch):
    """Reason PRIORITY is part of the histogram's meaning: an unserveable rung
    is refused as a rung, not as a capture."""
    _arm(monkeypatch)
    batch = _spec_batch(_lm(), drafter=_StubDFlash(), kind="dflash",
                        vault=object(), rungs=[STEP + 4])
    batch._pipeline_open()
    assert batch._pipeline is None
    assert pr.METRICS.snapshot()["pp_bypass_reason"] == {"apc_checkpoint_ladder": 1}


def test_generate_step_keeps_the_historical_refusal():
    """The other call site has no merge, so its bool still means what it meant."""
    from mlx_vlm.pipeline_runtime import _pipeline_bypass_reason
    from mlx_vlm.models.cache import ArraysCache

    args = dict(ladder=False, warm=False, pixel_values=None, mask=None,
                cache=[ArraysCache(2)], input_ids=mx.array([[1, 2]]),
                kv_quantized=False)
    assert _pipeline_bypass_reason(capture=True, **args) == (
        "speculative_hidden_capture"
    )
    assert _pipeline_bypass_reason(capture=None, **args) is None
    assert _pipeline_bypass_reason(capture="capture_unsupported", **args) == (
        "capture_unsupported"
    )


# ------------- 8. A5b: the default served config is DFlash2 AND APC exact ON
#
# A6 removed the capture refusal and A5b removes the APC-exact one, so this is
# the first configuration in which BOTH of the default server's reasons for
# staying single-box are gone at once.  What the combination adds over either
# alone is the drafter's ``hidden_tail``: the APC checkpoint carries one, it is
# built by ``_hidden_tail_for_store`` from the SAME accumulator the pipeline
# seeds with the merged two-box window, and a tail that is short or misordered
# is a quietly worse turn 2 rather than a failure.

CKPT_SPLIT = DEPTH + 4  # 36: the column lands inside the post-finalize remainder


def _apc_manager():
    from mlx_vlm.apc import APCManager

    return APCManager(num_blocks=8, block_size=16)


def _spec_apc_batch(lm, manager, checkpoint_len, **kw):
    batch = _spec_batch(lm, **kw)
    batch._apc_manager = manager
    batch._apc_mode = "exact"
    batch._apc_meta = [
        {
            "prefix_len": 0,
            "checkpoint_len": int(checkpoint_len),
            "full_input_ids": list(PROMPT),
            "extra_hash": 0,
            "vault_rungs": [],
        }
    ]
    return batch


def _spec_apc_arm(monkeypatch, *, pipelined, checkpoint_len=CKPT_SPLIT, **kw):
    pr.METRICS.reset()
    manager = _apc_manager()
    if pipelined:
        _arm(monkeypatch)
    else:
        monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    out = _drain_spec(_spec_apc_batch(_lm(), manager, checkpoint_len, **kw))
    out["entries"] = {len(e.token_ids): e for e in manager._exact_cache.values()}
    snap = pr.METRICS.snapshot()
    out["hist"] = snap["pp_bypass_reason"]
    out["used"] = snap["pp_used"]
    out["shortened"] = snap["pp_schedule_shortened_reason"]
    return out


def test_dflash_plus_apc_exact_is_the_first_admitted_default_config(monkeypatch):
    """Both default-ON refusals gone: no bypass, and the peer really ran."""
    arm = _spec_apc_arm(
        monkeypatch, pipelined=True, drafter=_StubDFlash(), kind="dflash"
    )
    assert arm["hist"] == {}, arm["hist"]
    assert arm["used"] == 1
    assert sorted(arm["entries"]) == [CKPT_SPLIT, len(PROMPT)]
    assert arm["steps"] == PIPELINED_CHUNKS + [4]


def test_the_checkpoint_hidden_tail_survives_the_two_box_merge(monkeypatch):
    """The tail is the drafter's turn-2 context, and it comes off the merge.

    On one box the accumulator holds every chunk this box ran.  On two it holds
    the window the peer returned, merged and adopted at ``finalize``, plus the
    remainder's own capture.  ``_hidden_tail_for_store`` reads that accumulator
    at the checkpoint column, so if the merge lost a row or reordered the layers
    the stored tail would differ HERE -- and nowhere else, because the prompt
    cache is unaffected by it.
    """
    pp = _spec_apc_arm(
        monkeypatch, pipelined=True, drafter=_StubDFlash(), kind="dflash"
    )
    ref = _spec_apc_arm(
        monkeypatch, pipelined=False, drafter=_StubDFlash(), kind="dflash"
    )
    assert pp["cache"] == ref["cache"], "the prefill itself moved"
    assert pp["steps"] == ref["steps"]
    assert _digest([pp["hidden"]]) == _digest([ref["hidden"]])
    assert pp["offset"] == ref["offset"]
    assert sorted(pp["entries"]) == sorted(ref["entries"]) == [CKPT_SPLIT, len(PROMPT)]
    for length in sorted(ref["entries"]):
        got, want = pp["entries"][length], ref["entries"][length]
        assert got.token_ids == want.token_ids
        assert _digest(_cache_arrays(got.prompt_cache)) == _digest(
            _cache_arrays(want.prompt_cache)
        ), f"the {length} snapshot moved"
        if want.hidden_tail is None:
            assert got.hidden_tail is None
        else:
            assert got.hidden_tail is not None
            assert _digest(got.hidden_tail) == _digest(want.hidden_tail)
            assert [t.shape for t in got.hidden_tail] == [
                t.shape for t in want.hidden_tail
            ]


def test_the_checkpoint_tail_is_the_window_over_the_checkpoint_not_the_prompt(
    monkeypatch,
):
    """A shape pin under the tail identity above, so a merge that produced the
    RIGHT bytes for the wrong span cannot pass as equal-to-equal."""
    pp = _spec_apc_arm(
        monkeypatch, pipelined=True, drafter=_StubDFlash(), kind="dflash"
    )
    tail = pp["entries"][CKPT_SPLIT].hidden_tail
    assert tail, "a dflash drafter's checkpoint carries a tail"
    assert all(int(t.shape[1]) == min(KEEP, CKPT_SPLIT) for t in tail), [
        t.shape for t in tail
    ]


# ------------- 9. A5c: the drafter's context across a SHORTENED schedule
#
# A5b admitted the exact column only when it already lay outside the pipelined
# part; A5c moves the boundary instead, which changes what the peer captures
# (one chunk less) and what this box captures (one chunk more).  The merge has
# to be indifferent to that -- ``adopt_window`` is told how many rows the window
# stands for and the remainder appends the rest -- and this is where that is
# checked, because the drafter's context is the one output that would be merely
# WORSE rather than wrong if the split moved.

CKPT_SWALLOWED = DEPTH - STEP  # 24: inside the last pipelined chunk


def test_the_shortened_schedule_hands_the_drafter_the_single_box_context(monkeypatch):
    """DFlash2 + APC exact + ``r < guard``: the config the smoke run bypassed.

    The peer now runs three chunks instead of four, so the window it returns
    covers 24 rows instead of 32 and this box captures the other 16 itself.
    ``finish()`` must still produce the single-box arrays AND the single-box
    ``target_hidden_offset`` -- the drafter's RoPE origin -- and the checkpoint
    entry must still carry the tail cut from that same accumulator.
    """
    pp = _spec_apc_arm(
        monkeypatch, pipelined=True, checkpoint_len=CKPT_SWALLOWED,
        drafter=_StubDFlash(), kind="dflash",
    )
    assert pp["hist"] == {} and pp["used"] == 1
    assert pp["shortened"] == {"exact_column": 1}

    ref = _spec_apc_arm(
        monkeypatch, pipelined=False, checkpoint_len=CKPT_SWALLOWED,
        drafter=_StubDFlash(), kind="dflash",
    )
    assert pp["steps"] == ref["steps"] == PIPELINED_CHUNKS
    assert pp["cache"] == ref["cache"], "the prefill itself moved"
    assert pp["offset"] == ref["offset"]
    assert pp["hidden"].shape == ref["hidden"].shape
    assert _digest([pp["hidden"]]) == _digest([ref["hidden"]])
    assert sorted(pp["entries"]) == sorted(ref["entries"]) == [
        CKPT_SWALLOWED, len(PROMPT)
    ]
    for length in sorted(ref["entries"]):
        got, want = pp["entries"][length], ref["entries"][length]
        assert got.token_ids == want.token_ids
        assert _digest(_cache_arrays(got.prompt_cache)) == _digest(
            _cache_arrays(want.prompt_cache)
        ), f"the {length} snapshot moved"
        if want.hidden_tail is None:
            assert got.hidden_tail is None
        else:
            assert got.hidden_tail is not None
            assert _digest(got.hidden_tail) == _digest(want.hidden_tail)


def test_the_shortened_window_is_the_depth_the_peer_actually_ran(monkeypatch):
    """``adopt_window`` is told ``sum(chunks)``, and the schedule is the plan's.

    If the adopt had kept measuring ``k*C`` while the peer ran ``k-1`` chunks,
    the window would be accounted 8 rows too deep and every drafter offset after
    it would be wrong by 8 -- silently, and only for prompts with ``r < guard``.
    """
    made = _arm(monkeypatch)
    pr.METRICS.reset()
    manager = _apc_manager()
    batch = _spec_apc_batch(
        _lm(), manager, CKPT_SWALLOWED, drafter=_StubDFlash(), kind="dflash"
    )
    _drain_spec(batch)
    assert made[0].chunks == PIPELINED_CHUNKS[:-1]
    assert made[0].head_window.window(), "the peer still captured its half"
    assert pr.METRICS.snapshot()["pp_schedule_shortened"] == 1
