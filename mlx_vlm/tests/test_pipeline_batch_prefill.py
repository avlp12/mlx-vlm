"""The SERVED prefill is the batch path, so that is where the pipeline has to be.

``generate_step`` had the only two-box call site, and the server never calls it:
its GPU thread runs ``BatchGenerator``, whose prefill is
``PromptProcessingBatch.prompt_step``.  A pipeline wired only into
``generate_step`` is therefore a pipeline no request can reach, which looks
exactly like a pipeline that is switched off.

These tests pin the second call site:

1.  every refusal has a NAME, and the name is decided by the batch's OWN facts
    (B > 1, right padding, a warm/APC prefix, a checkpoint ladder, a drafter
    hidden capture, too few tokens, the peer already busy);
2.  an admitted request calls ``begin -> prefill_chunk* -> finalize -> close``
    in that order, with the chunk schedule the loop actually runs;
3.  a peer that dies mid-prefill costs the request a re-prefill and nothing
    else: the cache and the prompt logits come out BIT-EQUAL to a run that
    never tried;
4.  with the feature off the prefill is byte-for-byte the prefill it was --
    pinned against a digest measured on ``a6634a75``, the commit before this
    call site existed;
5.  ``/metrics`` sees the served path: ``pp_used`` counts batch-path requests.

The peer is a double.  A ``prefill_chunk`` that runs the WHOLE stack locally is
a loopback of the two-box arrangement, so an admitted run must reproduce the
single-box cache exactly -- which makes (2) an identity test as well as an
order test.
"""

import contextlib
import hashlib
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm import pipeline_runtime as pr
from mlx_vlm.generate.ar import PromptProcessingBatch, _left_pad_prompts
from mlx_vlm.models.glm5_next.config import TextConfig
from mlx_vlm.models.glm5_next.language import LanguageModel

STEP = 8
PROMPT = list(range(3, 43))  # 40 tokens -> 4 pipelined chunks of 8, 8 left over
PIPELINED_CHUNKS = [STEP] * 4

# ``_cache_digest`` of the prompt cache after a cold 40-token prefill at
# ``prefill_step_size=8``, measured on a clean worktree of a6634a75 (the commit
# this call site is added on top of) with the identical fixture, CPU,
# MLX_DEFAULT_DEVICE=cpu.  It is the "the feature costs nothing when it is off"
# guard: the pipeline call site must not have moved one byte of the prefill.
A6634A75_COLD_CACHE = (
    "c30a699949b5d11aad4af522ff4d11a1a93ca7d54087fa6734b2519b95214f5a"
)

ON_GPU = mx.default_device() == mx.gpu


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


def _batch(lm, rows=None, *, step=STEP, **kwargs):
    rows = rows or [PROMPT]
    padded = _left_pad_prompts(rows)
    return PromptProcessingBatch(
        model=lm,
        uids=list(range(len(rows))),
        input_ids=rows,
        max_tokens=[1] * len(rows),
        inputs_embeds=lm.model.embed_tokens(padded),
        prompt_kwargs={},
        prefill_step_size=step,
        **kwargs,
    )


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


def _cache_digest(prompt_cache):
    arrays = _cache_arrays(prompt_cache)
    mx.eval(arrays)
    h = hashlib.sha256()
    for a in arrays:
        h.update(repr(tuple(a.shape)).encode())
        h.update(memoryview(np.asarray(a.astype(mx.float32))).tobytes())
    return h.hexdigest()


def _drain(batch):
    """Run the batch's prefill to completion; return (cache digest, prompt logprobs)."""
    steps = []
    while batch.needs_processing():
        steps.append(batch.prompt_step())
    seen = {}

    def sampler(logprobs):
        seen["lp"] = mx.array(logprobs)
        return mx.argmax(logprobs, axis=-1)

    gen = batch.generate(sampler=sampler, stop_criteria=lambda token: False)
    mx.eval(seen["lp"])
    return _cache_digest(gen.prompt_cache), seen["lp"], steps


# ------------------------------------------------------------------ the double


class _LoopbackHead:
    """A peer that is this box.

    ``prefill_chunk`` runs the WHOLE stack into the caller's cache, which is what
    the real head + real tail together do, so an admitted run must land on the
    single-box cache byte for byte.  Everything else records the verb order.
    """

    def __init__(self, settings=None, split=1, n_layers=2, fail_at=None):
        self.settings = settings
        self.split = split
        self.n_layers = n_layers
        self.sock = object()
        self.stats = {"wire_send_s": 0.0, "handoff": {"handoff_bytes": 7, "handoff_wire_recv_s": 0.5}}
        self.calls = []
        self.chunks = []
        self.fail_at = fail_at

    # -- pool surface
    def connect(self):
        self.calls.append("connect")
        return self

    def ping(self):
        return True

    def abort(self):
        self.calls.append("abort")
        self.sock = None

    # -- request surface
    def begin(self, tokens, chunk, *, input_ids):
        self.calls.append(("begin", int(tokens), int(chunk), tuple(input_ids.shape)))

    def local_caches(self, cache):
        return []

    def prefill_chunk(self, model, input_ids, inputs_embeds, cache):
        idx = len(self.chunks)
        if self.fail_at is not None and idx == self.fail_at:
            raise OSError("the tail went away mid-chunk")
        self.chunks.append(int(input_ids.shape[1]))
        self.calls.append(("prefill_chunk", int(input_ids.shape[1])))
        model(
            input_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            n_to_process=int(input_ids.shape[1]),
        )
        mx.async_eval([c.state for c in cache])

    def finalize(self, cache):
        self.calls.append("finalize")
        return self.stats

    def close(self):
        self.calls.append("bye")
        self.sock = None


def _arm(monkeypatch, *, min_tokens=16, fail_at=None, hosts="127.0.0.1:39210"):
    """Point the real gate at a double and return the head it will hand out."""
    made = []

    class Factory(_LoopbackHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers, fail_at=fail_at)
            made.append(self)

    monkeypatch.setattr(pr, "PipelineHead", Factory)
    monkeypatch.setattr(pr, "POOL", pr.PipelinePool())
    if hosts:
        monkeypatch.setenv("MLX_VLM_PIPELINE_HOSTS", hosts)
    else:
        monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    monkeypatch.setenv("MLX_VLM_PIPELINE_SPLIT", "1")
    monkeypatch.setenv("MLX_VLM_PIPELINE_MIN_TOKENS", str(min_tokens))
    monkeypatch.setenv("MLX_VLM_PIPELINE_MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("MLX_VLM_PIPELINE_SOURCE_REVISION", "b" * 40)
    return made


def _hist():
    return pr.METRICS.snapshot()["pp_bypass_reason"]


@contextlib.contextmanager
def _pipeline_off():
    """A reference arm that cannot take the pipeline, whatever this test armed."""
    saved = os.environ.pop("MLX_VLM_PIPELINE_HOSTS", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["MLX_VLM_PIPELINE_HOSTS"] = saved


def _single_box_reference():
    with _pipeline_off():
        return _drain(_batch(_lm()))


# ---------------------------------------------------------------- 1. the gate


def test_the_schedule_is_the_chunk_loop_and_nothing_else():
    """The peer gets the chunks the LOOP runs; ``generate()`` keeps the rest.

    ``prompt_step`` stops while ``remaining > step``, so the pipelined part is
    ``ceil(T/C) - 1`` chunks of ``C`` and the 1..C-token remainder is a
    full-stack forward on this box AFTER finalize.  If this arithmetic and the
    loop ever disagree, ``finalize`` refuses the envelope -- it is pinned here
    so the disagreement is caught without a peer.
    """
    lm = _lm()
    batch = _batch(lm)
    assert batch._pipeline_chunk_schedule() == PIPELINED_CHUNKS
    assert sum(PIPELINED_CHUNKS) + STEP == len(PROMPT)
    # exactly the n's the loop hands out, in order
    ns = []
    while batch.needs_processing():
        ns.append(batch.prompt_step())
    assert ns == PIPELINED_CHUNKS
    # a prompt that does not chunk at all is not a pipeline candidate
    assert _batch(lm, [PROMPT[:STEP]])._pipeline_chunk_schedule() == []


@pytest.mark.parametrize(
    "reason,rows,kwargs,patch",
    [
        ("batch_not_one", [PROMPT, list(range(9, 49))], {}, {}),
        (
            "right_pad_batch",
            [PROMPT, list(range(9, 49))],
            {"right_pad_per_row": [0, 3], "suffix_lens": [40, 37]},
            {},
        ),
        ("warm_prefix", [PROMPT], {}, {"_apc_meta": [{"prefix_len": 4}]}),
        (
            "apc_checkpoint_ladder",
            [PROMPT],
            {},
            {"_vault": object(), "_apc_meta": [{"vault_rungs": [16]}]},
        ),
        (
            "speculative_hidden_capture",
            [PROMPT],
            {},
            {"_chunk_capture_kwargs": {"capture_layer_ids": [0]}},
        ),
    ],
)
def test_every_refusal_has_a_name(monkeypatch, reason, rows, kwargs, patch):
    _arm(monkeypatch)
    lm = _lm()
    batch = _batch(lm, rows, **kwargs)
    for k, v in patch.items():
        setattr(batch, k, v)
    batch._pipeline_open()
    assert batch._pipeline is None
    assert _hist() == {reason: 1}, _hist()


def test_the_apc_exact_checkpoint_is_a_ladder_too(monkeypatch):
    """The APC single checkpoint and the vault ladder are one refusal.

    Both ask for a snapshot at a chunk boundary, and at a chunk boundary half
    the KV is on the peer.  A5 replaces this with ONE full-depth checkpoint
    taken after ``finalize``; until then a request that wants either one stays
    on the box that can give it.
    """
    _arm(monkeypatch)
    lm = _lm()
    batch = _batch(lm)
    batch._apc_manager = object()
    batch._apc_mode = "exact"
    batch._apc_meta = [{"checkpoint_len": 16, "prefix_len": 0}]
    batch._pipeline_open()
    assert batch._pipeline is None and _hist() == {"apc_checkpoint_ladder": 1}


def test_a_short_prompt_stays_on_one_box(monkeypatch):
    """A8: ``MLX_VLM_PIPELINE_MIN_TOKENS`` is the routing policy's 16k rule."""
    _arm(monkeypatch, min_tokens=16384)
    lm = _lm()
    batch = _batch(lm)
    batch._pipeline_open()
    assert batch._pipeline is None and _hist() == {"below_min_tokens": 1}


def test_the_default_min_tokens_is_the_policy_number(monkeypatch):
    """A8.  ``POLICY_long_prompt_pp_routing_2026-09-05.md`` rule 1 is >= 16k."""
    monkeypatch.delenv("MLX_VLM_PIPELINE_MIN_TOKENS", raising=False)
    monkeypatch.setenv("MLX_VLM_PIPELINE_HOSTS", "127.0.0.1:1")
    assert pr.DEFAULT_MIN_TOKENS == 16384
    assert pr.PipelineSettings.from_env().min_tokens == 16384


def test_turning_the_pipeline_on_later_in_the_process_works(monkeypatch):
    """A8.  The ``_CTX = _DISABLED`` latch made ``disabled`` permanent.

    A process that reached ``maybe_open_pipeline`` once before
    ``MLX_VLM_PIPELINE_HOSTS`` was set -- a test module, a server whose peer is
    configured after the model loads -- could never use the pipeline again, and
    its later requests were not even counted, because the latch returned before
    the bypass was recorded.
    """
    monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    assert pr.maybe_open_pipeline(object(), 100000) is None
    assert _hist() == {"disabled": 1}

    _arm(monkeypatch)  # the peer is configured NOW
    lease = pr.maybe_open_pipeline(_lm(), 100000)
    assert lease is not None, "the disabled decision must not have latched"
    lease.close()

    # and a refusal after that is still counted, every time
    monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    pr.maybe_open_pipeline(object(), 100000)
    pr.maybe_open_pipeline(object(), 100000)
    assert _hist()["disabled"] == 3


def test_a_second_eligible_request_falls_back_instead_of_queueing(monkeypatch):
    """One tail, one stage-B stack: the second request goes single-box NOW.

    Queueing behind the first would turn a free fallback into a wait of up to a
    whole 131k prefill.
    """
    _arm(monkeypatch)
    lm = _lm()
    first = _batch(lm)
    first._pipeline_open()
    assert first._pipeline is not None
    second = _batch(lm)
    second._pipeline_open()
    assert second._pipeline is None
    assert _hist() == {"pp_busy": 1}
    first._pipeline_release()
    third = _batch(lm)
    third._pipeline_open()
    assert third._pipeline is not None, "the slot is given back"
    third._pipeline_release()


def test_a_peer_that_is_down_is_a_fallback_not_a_failure(monkeypatch):
    class Refusing(_LoopbackHead):
        def __init__(self, settings, split, n_layers):
            super().__init__(settings, split, n_layers)

        def connect(self):
            raise ConnectionRefusedError("tail is down")

    _arm(monkeypatch)
    monkeypatch.setattr(pr, "PipelineHead", Refusing)
    lm = _lm()
    batch = _batch(lm)
    digest, logprobs, steps = _drain(batch)
    assert batch._pipeline is None
    assert _hist() == {"peer_unreachable": 1}
    assert steps == PIPELINED_CHUNKS
    ref_digest, ref_lp, _ = _single_box_reference()
    assert digest == ref_digest
    assert mx.array_equal(logprobs, ref_lp)


# ------------------------------------------------------- 2. the admitted path


def test_an_admitted_request_runs_the_five_verbs_in_order(monkeypatch):
    made = _arm(monkeypatch)
    lm = _lm()
    batch = _batch(lm)
    digest, logprobs, steps = _drain(batch)

    assert len(made) == 1
    head = made[0]
    assert head.calls == [
        "connect",
        ("begin", sum(PIPELINED_CHUNKS) + 1, STEP, (1, sum(PIPELINED_CHUNKS) + 1)),
        *[("prefill_chunk", n) for n in PIPELINED_CHUNKS],
        "finalize",
    ], head.calls
    assert head.chunks == PIPELINED_CHUNKS
    assert steps == PIPELINED_CHUNKS
    assert batch._pipeline is None, "the lease is released at finalize"
    assert _hist() == {}

    # loopback identity: the same cache and the same prompt logits as one box
    ref_digest, ref_lp, _ = _single_box_reference()
    assert digest == ref_digest
    assert mx.array_equal(logprobs, ref_lp)


def test_the_served_path_increments_pp_used(monkeypatch):
    _arm(monkeypatch)
    _drain(_batch(_lm()))
    snap = pr.pipeline_metrics_snapshot()
    assert snap["pp_used"] == 1
    assert snap["pp_failed"] == 0
    assert snap["pp_handoff_bytes"] == 7
    for key in ("pp_used", "pp_failed", "pp_bypass_reason", "pp_handoff_bytes",
                "pp_wire_s", "pp_breaker_state", "pp_pool_idle", "pp_enabled"):
        assert key in snap, key


def test_a_finished_request_returns_the_socket_to_the_pool(monkeypatch):
    made = _arm(monkeypatch)
    _drain(_batch(_lm()))
    assert made[0].calls[-1] == "finalize", "close() is not bye"
    assert pr.POOL.idle_count() == 1
    _drain(_batch(_lm()))
    assert len(made) == 1, "the socket outlives the request"
    assert pr.METRICS.snapshot()["pp_used"] == 2


# --------------------------------------------------------- 3. the failure model


def test_a_mid_prefill_failure_re_prefills_single_box_bit_for_bit(monkeypatch):
    """The plan's failure model, measured.

    Half the layers are written on this box and half were never written, so the
    partial cache can only be thrown away.  What the client must see is a
    request that is merely SLOWER: the cache and the prompt logits have to come
    out bit-equal to a run that never tried.
    """
    made = _arm(monkeypatch, fail_at=2)
    lm = _lm()
    batch = _batch(lm)
    digest, logprobs, steps = _drain(batch)

    head = made[0]
    assert head.chunks == PIPELINED_CHUNKS[:2], "it died on the third chunk"
    assert head.calls[-1] == "abort", "a failed request discards its connection"
    assert "finalize" not in head.calls
    # the two pipelined chunks are re-run on this box, so the loop runs 4 + 2
    assert steps == PIPELINED_CHUNKS[:2] + PIPELINED_CHUNKS
    assert batch._pipeline is None and batch._pipeline_restore is None

    ref_digest, ref_lp, ref_steps = _single_box_reference()
    assert ref_steps == PIPELINED_CHUNKS
    assert digest == ref_digest, "the fallback prefill is not the cold prefill"
    assert mx.array_equal(logprobs, ref_lp)

    snap = pr.pipeline_metrics_snapshot()
    assert snap["pp_failed"] == 1 and snap["pp_used"] == 0
    assert pr.POOL.idle_count() == 0


def test_a_failed_request_gives_the_slot_back(monkeypatch):
    _arm(monkeypatch, fail_at=0)
    _drain(_batch(_lm()))
    assert pr.acquire_pipeline_slot() is True
    pr.release_pipeline_slot()


def test_cancelling_a_prefill_releases_the_lease(monkeypatch):
    """``BatchGenerator.remove`` drops the batch; the peer must hear about it."""
    made = _arm(monkeypatch)
    lm = _lm()
    batch = _batch(lm)
    batch.prompt_step()
    assert batch._pipeline is not None

    from mlx_vlm.generate.ar import BatchGenerator

    BatchGenerator._release_prompt_batch_pipeline(batch)
    assert batch._pipeline is None
    assert made[0].calls[-1] == "abort"
    assert pr.acquire_pipeline_slot() is True
    pr.release_pipeline_slot()


# ------------------------------------------------- 4. the feature, switched off


def test_with_the_feature_off_the_gate_is_never_evaluated(monkeypatch):
    """One ``os.environ.get`` and not one array touched.

    ``pipeline_bypass_reason`` reads the cache, the mask and the token ids; with
    ``MLX_VLM_PIPELINE_HOSTS`` unset none of that may happen, or an env-off
    server would pay a cache walk per prefill for a feature it is not using.
    """
    _arm(monkeypatch, hosts=None)

    def explode(*a, **k):
        raise AssertionError("the gate ran with the pipeline switched off")

    monkeypatch.setattr(pr, "pipeline_bypass_reason", explode)
    monkeypatch.setattr(pr, "maybe_open_pipeline", explode)
    batch = _batch(_lm())
    digest, _, steps = _drain(batch)
    assert steps == PIPELINED_CHUNKS
    assert batch._pipeline is None and batch._pipeline_declined is True
    assert pr.METRICS.snapshot()["pp_bypass_reason"] == {}
    assert digest == _single_box_reference()[0]


@pytest.mark.skipif(ON_GPU, reason="the digest is a CPU byte pin")
def test_the_cold_prefill_is_byte_identical_to_a6634a75(monkeypatch):
    """The guard that the call site did not become a prefill change.

    Measured on a clean detached worktree of ``a6634a75`` with this fixture.
    """
    monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    digest, _, steps = _drain(_batch(_lm()))
    assert steps == PIPELINED_CHUNKS
    assert digest == A6634A75_COLD_CACHE
