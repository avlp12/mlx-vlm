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
    def begin(self, tokens, chunk, *, input_ids, capture=None):
        self.capture = capture
        self.calls.append(("begin", int(tokens), int(chunk), tuple(input_ids.shape)))

    def take_hidden(self):
        return None

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
        # A10-0 split the blanket ``warm_prefix`` refusal in the BATCH path in
        # two, by the uncached suffix against the chunk size.  Same prompt, same
        # 4-token cached prefix, same refusal -- only C moves.
        (
            "warm_suffix_ge_chunk",  # S = 40 > C = 8: A10 would have chunks to send
            [PROMPT],
            {},
            {"_apc_meta": [{"prefix_len": 4}]},
        ),
        (
            "warm_suffix_lt_chunk",  # S = 40 <= C = 64: ceil(S/C) - 1 == 0, ever
            [PROMPT],
            {"step": 64},
            {"_apc_meta": [{"prefix_len": 4}]},
        ),
        (
            # A5 admits a rung the post-finalize checkpoint can stand in for.
            # This one it cannot: 12 is not a multiple of the chunk size, so
            # ``_next_apc_checkpoint_column`` would clamp a chunk to land on it
            # and the peer's schedule has no such chunk in it.
            "apc_checkpoint_ladder",
            [PROMPT],
            {},
            {"_vault": object(), "_apc_meta": [{"vault_rungs": [12]}]},
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

    Extended for A5, which touched the vault ladder's STORE path (the chunk
    loop's capture and the post-finalize one now share
    ``_insert_vault_rungs``) and its GATE.  Neither may be visible with the
    feature off, so the env-off arm is run twice -- once as a plain cold
    prefill and once with a vault ladder in hand -- and both land on the same
    pinned digest, with the ladder stored WHOLE (no collapse, no counter).
    """
    monkeypatch.delenv("MLX_VLM_PIPELINE_HOSTS", raising=False)
    digest, _, steps = _drain(_batch(_lm()))
    assert steps == PIPELINED_CHUNKS
    assert digest == A6634A75_COLD_CACHE

    vault = _FakeVault()
    ladder_digest, _, ladder_steps = _drain(_vault_batch(_lm(), vault, LADDER))
    assert ladder_steps == PIPELINED_CHUNKS
    assert ladder_digest == A6634A75_COLD_CACHE, "the ladder store moved the prefill"
    assert vault.depths() == LADDER, "every rung, at its own depth"
    snap = pr.METRICS.snapshot()
    assert snap["pp_ladder_collapsed"] == 0 and snap["pp_bypass_reason"] == {}


# ------------------------------------------- 5. A5: the one checkpoint PP can take
#
# The vault has been default-ON since d1a3c3a4, so ``apc_checkpoint_ladder``
# refused every request in the default served config: a pipeline that is
# reachable only with the vault switched off is a pipeline nobody reaches.  A5
# narrows the gate to the rungs the single post-finalize checkpoint cannot stand
# in for, and takes that one checkpoint through the SAME capture the chunk loop
# uses -- which is what these pin.


class _FakeVault:
    """Records what the store side hands it, in order."""

    def __init__(self):
        self.inserts = []

    def insert(self, tokens, prefix_len, fragments, harvest_provenance=None, **kw):
        self.inserts.append(
            {
                "tokens": list(tokens),
                "prefix_len": int(prefix_len),
                "fragments": fragments,
                "harvest_provenance": harvest_provenance,
            }
        )
        return True

    def depths(self):
        return [i["prefix_len"] for i in self.inserts]


def _fragments_digest(fragments):
    """sha256 over every array in a captured rung, shapes included."""
    assert fragments is not None, "capture_fragments refused the cache"
    arrays = []
    stack = [f.payload for f in fragments]
    while stack:
        item = stack.pop()
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
    mx.eval(arrays)
    h = hashlib.sha256()
    for a in arrays:
        h.update(repr(tuple(a.shape)).encode())
        h.update(memoryview(np.asarray(a.astype(mx.float32))).tobytes())
    return h.hexdigest(), len(arrays)


DEPTH = sum(PIPELINED_CHUNKS)  # 32: the deepest rung align_boundaries admits
LADDER = [STEP * 2, DEPTH]  # 16, 32 -- a geometric ladder in miniature


def _vault_batch(lm, vault, rungs, **kwargs):
    batch = _batch(lm, **kwargs)
    batch._vault = vault
    batch._apc_meta = [
        {"prefix_len": 0, "vault_rungs": list(rungs), "full_input_ids": list(PROMPT)}
    ]
    return batch


def test_the_deepest_rung_is_exactly_the_depth_pp_can_serve():
    """Not a lucky case -- the only case.

    ``align_boundaries`` admits a rung only if it is a positive multiple of the
    chunk size strictly below the prompt length, and the largest such multiple
    IS ``k*C``.  So a full-depth checkpoint can always stand in for the DEEPEST
    rung the ladder asks for; the narrowing turns on nothing else.
    """
    from mlx_vlm.context_vault import align_boundaries, boundary_ladder

    for total in (40, 41, 47, 48, 64, 129):
        depth = (-(-total // STEP) - 1) * STEP
        admissible = align_boundaries(range(1, total), STEP, total)
        assert admissible and max(admissible) == depth, (total, depth, admissible)
        ladder = boundary_ladder(total, stride=STEP * 2, step=STEP)
        assert all(r <= depth for r in ladder), (total, ladder)


def test_a_servable_ladder_is_admitted_and_collapsed(monkeypatch):
    _arm(monkeypatch)
    vault = _FakeVault()
    batch = _vault_batch(_lm(), vault, LADDER)
    digest, logprobs, steps = _drain(batch)

    assert steps == PIPELINED_CHUNKS, "the collapse must not move a chunk boundary"
    assert vault.depths() == [DEPTH], "one rung, at full depth"
    assert _hist() == {}, "no refusal"
    snap = pr.METRICS.snapshot()
    assert snap["pp_used"] == 1
    assert snap["pp_ladder_collapsed"] == 1
    assert snap["pp_ladder_rungs_skipped"] == 1, "the 16 rung is the one dropped"

    # ... and the prefill itself is still the single-box prefill
    ref_digest, ref_lp, _ = _single_box_reference()
    assert digest == ref_digest
    assert mx.array_equal(logprobs, ref_lp)


def test_the_collapsed_rung_is_bit_identical_to_the_single_box_rung(monkeypatch):
    """The loopback double against one box, at the same depth, byte for byte.

    This is the ``align_boundaries`` contract restated for the two-box path: a
    rung at a multiple of the chunk size restores to the same cache a
    straight-through cold prefill would have had at that length.  If the
    post-finalize capture used a different function, a different row snapshot or
    a different moment, the payload would differ here.
    """
    pp_vault = _FakeVault()
    _arm(monkeypatch)
    _drain(_vault_batch(_lm(), pp_vault, LADDER))

    ref_vault = _FakeVault()
    with _pipeline_off():
        _drain(_vault_batch(_lm(), ref_vault, [DEPTH]))

    assert pp_vault.depths() == ref_vault.depths() == [DEPTH]
    got, n_got = _fragments_digest(pp_vault.inserts[0]["fragments"])
    want, n_want = _fragments_digest(ref_vault.inserts[0]["fragments"])
    assert n_got == n_want and n_got > 0
    assert got == want, "the pipelined rung is not the single-box rung"
    # the keys the vault indexes on, and the provenance it records, too
    assert pp_vault.inserts[0]["tokens"] == ref_vault.inserts[0]["tokens"]

    def _prov(entry):  # everything but the wall clock the capture happened at
        return {
            k: v
            for k, v in (entry["harvest_provenance"] or {}).items()
            if k != "harvest_at"
        }

    assert _prov(pp_vault.inserts[0]) == _prov(ref_vault.inserts[0])
    assert _prov(pp_vault.inserts[0]), "provenance is recorded, not merely equal"


@pytest.mark.parametrize(
    "rungs,why",
    [
        ([STEP + 4], "unaligned: it would clamp a chunk the peer does not have"),
        ([DEPTH + STEP], "deeper than the pipelined part"),
    ],
)
def test_a_rung_the_full_depth_checkpoint_cannot_serve_is_still_refused(
    monkeypatch, rungs, why
):
    _arm(monkeypatch)
    batch = _vault_batch(_lm(), _FakeVault(), rungs)
    batch._pipeline_open()
    assert batch._pipeline is None, why
    assert _hist() == {"apc_checkpoint_ladder": 1}


def test_the_apc_exact_checkpoint_is_still_refused_whole(monkeypatch):
    """A5 narrows the VAULT clause and nothing else.

    APC's checkpoint sits at a length the block policy chose from the prompt,
    not at ``k*C``, and its store is keyed on ``full_input_ids[:checkpoint_len]``
    -- a k*C snapshot is not a slower answer to it, it is a different one.
    """
    _arm(monkeypatch)
    batch = _vault_batch(_lm(), _FakeVault(), [DEPTH])
    batch._apc_manager = object()
    batch._apc_mode = "exact"
    batch._apc_meta[0]["checkpoint_len"] = 16
    batch._pipeline_open()
    assert batch._pipeline is None and _hist() == {"apc_checkpoint_ladder": 1}


def test_no_rung_fires_while_half_the_cache_is_on_the_peer(monkeypatch):
    """The reason the collapse happens at OPEN and not at finalize.

    A rung at 16 lands exactly on a chunk boundary.  If it were still pending
    there, ``_store_vault_checkpoints`` would capture a cache whose stage-B
    layers had never been written -- an entry that restores to a fluent wrong
    answer rather than to a slow one.
    """
    _arm(monkeypatch)
    vault = _FakeVault()
    batch = _vault_batch(_lm(), vault, LADDER)
    seen = []
    while batch.needs_processing():
        batch.prompt_step()
        seen.append((batch._pipeline_chunks_done, list(vault.depths())))
    # nothing stored until the last chunk, whose step is the one that finalizes
    assert [s[1] for s in seen[:-1]] == [[] for _ in seen[:-1]], seen
    assert seen[-1][1] == [DEPTH]


def test_a_dead_peer_gives_the_whole_ladder_back(monkeypatch):
    """The collapse is a property of the PP attempt, not of the request.

    A peer that dies mid-prefill sends the request back to column 0 on a fresh
    cache, single-box -- which can serve every rung.  If the collapse were not
    undone, a tail that died at 03:00 would quietly turn a vault-on server into
    a vault-storing-one-rung server.
    """
    _arm(monkeypatch, fail_at=2)
    vault = _FakeVault()
    digest, logprobs, _ = _drain(_vault_batch(_lm(), vault, LADDER))
    assert vault.depths() == LADDER, "the full ladder, stored by the fallback"

    ref_vault = _FakeVault()
    with _pipeline_off():
        ref_digest, ref_lp, _ = _drain(_vault_batch(_lm(), ref_vault, LADDER))
    assert digest == ref_digest
    assert mx.array_equal(logprobs, ref_lp)
    got = [_fragments_digest(i["fragments"]) for i in vault.inserts]
    want = [_fragments_digest(i["fragments"]) for i in ref_vault.inserts]
    assert got == want, "the fallback's rungs are the never-tried run's rungs"
    assert pr.METRICS.snapshot()["pp_ladder_collapsed"] == 1


def test_a_store_that_blows_up_after_finalize_does_not_cost_the_prefill(monkeypatch):
    """The prefill is COMPLETE by then.

    Letting a best-effort store reach ``_pipeline_step``'s handler would throw
    away a full, correct cache and pay the whole prompt again.
    """
    _arm(monkeypatch)

    def explode(self, batch_idx, meta, rungs):
        raise RuntimeError("the vault fell over")

    monkeypatch.setattr(PromptProcessingBatch, "_insert_vault_rungs", explode)
    batch = _vault_batch(_lm(), _FakeVault(), LADDER)
    digest, logprobs, steps = _drain(batch)
    assert steps == PIPELINED_CHUNKS, "no re-prefill"
    assert pr.METRICS.snapshot()["pp_used"] == 1
    ref_digest, ref_lp, _ = _single_box_reference()
    assert digest == ref_digest and mx.array_equal(logprobs, ref_lp)


def test_with_no_vault_the_ladder_code_is_not_reached(monkeypatch):
    """The default-off shape: no vault, no meta, no collapse, no counter."""
    _arm(monkeypatch)
    batch = _batch(_lm())
    _drain(batch)
    snap = pr.METRICS.snapshot()
    assert snap["pp_ladder_collapsed"] == 0 and snap["pp_ladder_rungs_skipped"] == 0
    assert batch._pipeline_ladder is None


# ---------------------------------- 6. A5: the session tier after a PP turn 1
#
# The rail on A5 that matters most in production: the multi-turn win
# (MLX_VLM_APC_SAVE_SESSION, default ON) must survive a pipelined turn 1.  It
# should, structurally -- the session rung is captured from the cache AFTER the
# response finishes, downstream of everything the pipeline touches -- and this
# is the measurement of "should".


class _PickGen:
    """Only what ``_vault_pick_for`` touches -- the same duck type
    ``test_session_restore._Gen`` uses, which is why the method is called
    unbound here rather than through a BatchGenerator nobody needs."""

    def __init__(self, vault, model=None):
        self.vault = vault
        self.model = model if model is not None else _lm()
        self.apc_manager = None

    def _vault_prefix_trim_is_safe(self):
        return True

    def _apc_extra_hash(self, kw):
        return 0


def _turn1_cache(monkeypatch, *, pipelined):
    """Turn 1's post-response cache: prompt + the token generate() produced."""
    ctx = contextlib.nullcontext()
    if pipelined:
        _arm(monkeypatch)
    else:
        ctx = _pipeline_off()
    with ctx:
        batch = _batch(_lm())
        while batch.needs_processing():
            batch.prompt_step()
        first = {}

        def sampler(logprobs):
            tok = mx.argmax(logprobs, axis=-1)
            first["tok"] = tok
            return tok

        gen = batch.generate(sampler=sampler, stop_criteria=lambda t: False)
        mx.eval(first["tok"])
        used = batch._pipeline_declined and pipelined
        assert not pipelined or pr.METRICS.snapshot()["pp_used"] == 1, used
        return gen.prompt_cache, PROMPT + [int(first["tok"][0].item())]


def test_a_pp_turn_1_leaves_the_session_tier_exactly_where_one_box_does(monkeypatch):
    """Turn 2 must find the same rung, at the same depth, with the same bytes.

    Two arms, identical but for the peer: capture the end-of-turn rung the way
    ``BatchGenerator.capture_session`` does (``snapshot_prompt_cache_row`` ->
    ``record_session_turn``), then ask the READ side -- ``_vault_pick_for``,
    called unbound against a duck-typed generator exactly as the session tests
    do -- what turn 2 gets.
    """
    from mlx_vlm import apc as _apc
    from mlx_vlm import context_vault as cv
    from mlx_vlm.generate.ar import BatchGenerator

    monkeypatch.setenv("MLX_VLM_APC_SAVE_SESSION", "1")
    assert cv.session_tier_active()
    turn2 = None
    picks = {}
    warm = {}
    for arm in ("pipelined", "single_box"):
        pr.METRICS.reset()
        cache, key = _turn1_cache(monkeypatch, pipelined=arm == "pipelined")
        vault = cv.ContextVault("identity-for-the-test", budget_bytes=1 << 30)
        row = _apc.snapshot_prompt_cache_row(cache, 0)
        assert row, "the session capture reads the row off the finished cache"
        assert cv.record_session_turn(
            vault, key, row, completed=True, session_id="conv-1", adopt=False
        ), cv.session_skip_counts()

        turn2 = key + [97, 98, 99]  # turn 1 plus the next user message
        pick = BatchGenerator._vault_pick_for(_PickGen(vault), turn2, {}, None)
        assert pick is not None, f"{arm}: turn 2 missed the session tier"
        picks[arm] = {
            "prefix_len": pick["prefix_len"],
            "source": pick.get("source"),
            "cached_tokens": pick["prefix_len"],
        }
        warm[arm] = _cache_digest(pick["warm_cache"])

    assert picks["pipelined"] == picks["single_box"], picks
    assert picks["pipelined"]["source"] == "vault-session"
    # The rung sits at the cache's depth, which is the prompt: ``generate()``
    # EMITS the first token, it does not forward it.  What matters here is not
    # the number but that both arms produce the same one, and that it is deeper
    # than the k*C rung the collapse keeps (32) -- so turn 2 restores the whole
    # of turn 1 and prefills only the new user message.
    assert picks["pipelined"]["prefix_len"] == len(PROMPT) > DEPTH
    assert warm["pipelined"] == warm["single_box"], "the restored cache moved"


def test_the_session_rung_is_not_the_collapsed_prefill_rung(monkeypatch):
    """The two tiers are separate stores and the PP path must keep them so.

    The collapse writes ONE prefill rung at ``k*C``; the session rung is written
    at the end of the turn, at prompt+generated.  A turn-2 prompt is deeper than
    both, so the pick must be the session one -- if the collapse had leaked into
    the session trie, turn 2 would restore a rung 9 tokens shallower and
    re-prefill the whole tail of turn 1.
    """
    from mlx_vlm import apc as _apc
    from mlx_vlm import context_vault as cv
    from mlx_vlm.generate.ar import BatchGenerator

    monkeypatch.setenv("MLX_VLM_APC_SAVE_SESSION", "1")
    vault = cv.ContextVault("identity-for-the-test", budget_bytes=1 << 30)
    _arm(monkeypatch)
    batch = _vault_batch(_lm(), vault, LADDER)
    while batch.needs_processing():
        batch.prompt_step()
    first = {}

    def sampler(logprobs):
        first["tok"] = mx.argmax(logprobs, axis=-1)
        return first["tok"]

    gen = batch.generate(sampler=sampler, stop_criteria=lambda t: False)
    mx.eval(first["tok"])
    key = PROMPT + [int(first["tok"][0].item())]
    assert cv.record_session_turn(
        vault,
        key,
        _apc.snapshot_prompt_cache_row(gen.prompt_cache, 0),
        completed=True,
        session_id="conv-1",
        adopt=False,
    )
    turn2 = key + [97, 98, 99]
    pick = BatchGenerator._vault_pick_for(_PickGen(vault, gen.model), turn2, {}, None)
    assert pick is not None
    assert pick["prefix_len"] == len(PROMPT) > DEPTH, "the session rung, not k*C"
    assert pr.METRICS.snapshot()["pp_ladder_collapsed"] == 1


# ------------------------- 7. A5b: the APC exact checkpoint a PP prefill CAN take
#
# A5 refused APC exact WHOLE, and the server turns APC exact on by default
# (``APC_ENABLED`` defaults to "1", ``runtime_config``).  So the first real
# two-box served smoke bypassed every request with ``apc_checkpoint_ladder`` --
# at 8192, at 32768 and at 131072 tokens alike -- and the feature was
# unreachable in the configuration it ships in.
#
# The refusal was wrong about WHERE the checkpoint column is.  It is
# ``len(prompt) - APC_EXACT_PREFIX_GUARD_TOKENS`` (16 by default), so with a
# remainder ``r = T - k*C`` the column sits in the REMAINDER whenever ``r > 16``
# -- and the remainder is the part this box runs itself, after ``finalize``, over
# all ``n_layers``.  Nothing has to be moved or reconstructed: the same
# ``prompt_step`` clamps the same chunk to the same column and the same
# ``_store_apc_exact_checkpoints`` writes the same ``store_exact_cache``.
#
# These pin that, the ordering fix that stops a length refusal from being
# reported as a shape refusal, and the one case that is still unserveable.

CKPT_SPLIT = DEPTH + 4  # 36: inside the remainder -- the DEFAULT served shape
CKPT_AT_DEPTH = DEPTH  # 32: exactly k*C, taken at finalize
CKPT_SWALLOWED = DEPTH - 8  # 24: inside the pipelined part -- unserveable


def _apc_manager():
    from mlx_vlm.apc import APCManager

    return APCManager(num_blocks=8, block_size=16)


def _apc_batch(lm, manager, checkpoint_len, *, vault=None, rungs=None, **kwargs):
    """A batch shaped the way the server shapes one with APC exact ON."""
    batch = _batch(lm, **kwargs)
    batch._apc_manager = manager
    batch._apc_mode = "exact"
    if vault is not None:
        batch._vault = vault
    batch._apc_meta = [
        {
            "prefix_len": 0,
            "checkpoint_len": int(checkpoint_len),
            "full_input_ids": list(PROMPT),
            "extra_hash": 0,
            "vault_rungs": list(rungs or []),
        }
    ]
    return batch


def _apc_entries(manager):
    """Every exact entry the prefill left behind, keyed by its token length.

    There are TWO on a cold APC-exact prefill and both are part of the contract:
    the CHECKPOINT at ``checkpoint_len``, written by
    ``_store_apc_exact_checkpoints`` from inside the chunk loop, and the
    post-prefill HARVEST at the whole prompt, written by ``generate()``.  A5b
    moves neither; comparing the whole set is what proves it.
    """
    return {len(e.token_ids): e for e in manager._exact_cache.values()}


def _entry_digest(entry):
    return _cache_digest(entry.prompt_cache)


def _prov(entry):  # everything but the wall clock the capture happened at
    return {k: v for k, v in (entry.provenance or {}).items() if k != "harvest_at"}


def _apc_arm(monkeypatch, *, pipelined, checkpoint_len, **kwargs):
    """Run one prefill to completion and hand back its APC store."""
    pr.METRICS.reset()
    manager = _apc_manager()
    ctx = contextlib.nullcontext()
    if pipelined:
        _arm(monkeypatch)
    else:
        ctx = _pipeline_off()
    with ctx:
        digest, logprobs, steps = _drain(
            _apc_batch(_lm(), manager, checkpoint_len, **kwargs)
        )
    return {
        "manager": manager,
        "cache": digest,
        "logprobs": logprobs,
        "steps": steps,
        "hist": _hist(),
        "used": pr.METRICS.snapshot()["pp_used"],
    }


@pytest.mark.parametrize("total", [8192, 16384, 32768, 40960, 131072])
def test_the_served_checkpoint_column_is_in_the_remainder(total):
    """The arithmetic the smoke run contradicted, taken from the served code.

    ``_apc_exact_checkpoint_len`` is what the server puts in ``checkpoint_len``;
    ``k*C`` is what ``_pipeline_chunk_schedule`` hands the peer.  At the shipped
    chunk size the column is deeper than the pipelined part at EVERY length the
    routing policy admits, which is why A5's blanket refusal cost the whole
    feature rather than an edge of it.
    """
    from mlx_vlm.generate.ar import BatchGenerator

    class _Gen:
        apc_mode = "exact"

        def __init__(self, manager):
            self.apc_manager = manager

        def _apc_media_token_ids(self):
            return ()

    chunk = 2048
    column = BatchGenerator._apc_exact_checkpoint_len(
        _Gen(_apc_manager()), list(range(total))
    )
    depth = (-(-total // chunk) - 1) * chunk
    assert column == total - 16, "the guard is APC_EXACT_PREFIX_GUARD_TOKENS"
    assert column >= depth, (total, column, depth)
    assert 0 < total - depth <= chunk


def test_the_apc_exact_request_the_server_sends_is_admitted(monkeypatch):
    """No ``apc_checkpoint_ladder``, and the peer is actually used."""
    arm = _apc_arm(monkeypatch, pipelined=True, checkpoint_len=CKPT_SPLIT)
    assert arm["hist"] == {}, arm["hist"]
    assert arm["used"] == 1
    assert sorted(_apc_entries(arm["manager"])) == [CKPT_SPLIT, len(PROMPT)], (
        "the checkpoint was taken, and so was the post-prefill harvest"
    )


def test_the_remainder_is_split_at_the_column_exactly_as_one_box_splits_it(monkeypatch):
    """Item 3: the column inside the remainder needs no new code.

    ``finalize`` leaves the head holding ``k*C`` and the remainder is single-box
    ``prompt_step``, so the loop clamps its own chunk to the column the way it
    always has.  Pinned as the STEP SEQUENCE because that is the thing that
    would silently differ: a remainder run as one forward instead of
    ``4 + 4`` is mathematically equal and not bit-identical.
    """
    pp = _apc_arm(monkeypatch, pipelined=True, checkpoint_len=CKPT_SPLIT)
    ref = _apc_arm(monkeypatch, pipelined=False, checkpoint_len=CKPT_SPLIT)
    assert ref["steps"] == pp["steps"] == PIPELINED_CHUNKS + [4], pp["steps"]
    assert pp["steps"][: len(PIPELINED_CHUNKS)] == PIPELINED_CHUNKS
    assert pp["cache"] == ref["cache"]
    assert mx.array_equal(pp["logprobs"], ref["logprobs"])


@pytest.mark.parametrize(
    "checkpoint_len,steps",
    [
        (CKPT_SPLIT, PIPELINED_CHUNKS + [4]),
        (CKPT_AT_DEPTH, PIPELINED_CHUNKS),
    ],
    ids=["inside-the-remainder", "exactly-k*C"],
)
def test_the_pipelined_snapshot_is_the_single_box_snapshot(
    monkeypatch, checkpoint_len, steps
):
    """The identity that makes the admission safe, on both admitted geometries.

    A checkpoint stored at the right length but from the wrong cache is not a
    slower entry, it is a wrong one -- so this compares the ENTRY: its key, its
    payload byte for byte, its provenance, and the tail slot.  ``exactly-k*C``
    is the geometry where the store fires inside ``_pipeline_step`` itself,
    right after ``finalize``; ``inside-the-remainder`` is the one where it fires
    from a later, wholly single-box ``prompt_step``.
    """
    pp = _apc_arm(monkeypatch, pipelined=True, checkpoint_len=checkpoint_len)
    ref = _apc_arm(monkeypatch, pipelined=False, checkpoint_len=checkpoint_len)
    assert pp["steps"] == ref["steps"] == steps
    assert pp["hist"] == {} and pp["used"] == 1

    got, want = _apc_entries(pp["manager"]), _apc_entries(ref["manager"])
    assert sorted(got) == sorted(want) == [checkpoint_len, len(PROMPT)]
    for length in sorted(want):
        g, w = got[length], want[length]
        assert g.token_ids == w.token_ids == tuple(PROMPT[:length])
        assert g.extra_hash == w.extra_hash
        assert _entry_digest(g) == _entry_digest(w), f"the {length} snapshot moved"
        assert _prov(g) == _prov(w)
        assert _prov(g), "provenance is recorded, not merely equal"
        assert g.hidden_tail == w.hidden_tail is None


def test_the_second_request_hits_the_pipelined_entry_identically(monkeypatch):
    """Turn 2's read side: same prefix length, same restored bytes, both arms.

    ``lookup_exact_cache`` is what decides ``cached_tokens`` and therefore the
    TTFT path, so an entry that stores but does not SERVE would look like a
    working pipeline and a cache that quietly went cold.
    """
    pp = _apc_arm(monkeypatch, pipelined=True, checkpoint_len=CKPT_SPLIT)
    ref = _apc_arm(monkeypatch, pipelined=False, checkpoint_len=CKPT_SPLIT)
    seen = {}
    for name, arm in (("pipelined", pp), ("single_box", ref)):
        cache, prefix_len = arm["manager"].lookup_exact_cache(
            PROMPT, 0, max_prefix_tokens=len(PROMPT) - 1
        )
        assert cache is not None, f"{name}: turn 2 missed the exact entry"
        seen[name] = (prefix_len, _cache_digest(cache))
    assert seen["pipelined"] == seen["single_box"], seen
    assert seen["pipelined"][0] == CKPT_SPLIT


def test_a_column_the_pipelined_part_would_swallow_is_still_refused(monkeypatch):
    """The one case A5b cannot serve, and it keeps the existing name.

    A column strictly inside ``[0, k*C)`` would have to stop a chunk the peer's
    schedule does not have, and would snapshot a cache whose stage-B layers were
    never written.  Reachable only when the remainder is shorter than the guard
    (``T mod C in [1, 15]`` at the shipped chunk size); there the request
    prefills on the box that can give it, as it did before A5b.
    """
    _arm(monkeypatch)
    batch = _apc_batch(_lm(), _apc_manager(), CKPT_SWALLOWED)
    batch._pipeline_open()
    assert batch._pipeline is None and _hist() == {"apc_checkpoint_ladder": 1}


def test_below_min_tokens_is_named_before_any_shape_reason(monkeypatch):
    """Item 2: a length refusal must not be reported as a shape refusal.

    The 8192-token request in a >= 16k-routed config was reported as
    ``apc_checkpoint_ladder``, which says "this request's SHAPE is unserveable"
    about a request the routing policy never intended to route at all.  The
    histogram is the only thing an operator reads to find out why the peer is
    idle, so the order is part of its meaning: arm this batch with BOTH an
    unserveable APC column and an unserveable vault rung and it still reports
    the length.
    """
    _arm(monkeypatch, min_tokens=len(PROMPT) + 1)
    batch = _apc_batch(
        _lm(), _apc_manager(), CKPT_SWALLOWED, vault=object(), rungs=[STEP + 4]
    )
    batch._pipeline_open()
    assert batch._pipeline is None and _hist() == {"below_min_tokens": 1}


def test_the_vault_ladder_and_the_apc_column_are_served_together(monkeypatch):
    """The default served config is BOTH: vault ON since d1a3c3a4, APC exact ON
    since ``APC_ENABLED`` defaulted to "1".  The collapse keeps the deepest rung
    at ``k*C`` and the APC column is taken in the remainder; neither store moves
    a chunk boundary and both entries match the single-box run."""
    pr.METRICS.reset()
    manager, vault = _apc_manager(), _FakeVault()
    _arm(monkeypatch)
    pp_steps = _drain(
        _apc_batch(_lm(), manager, CKPT_SPLIT, vault=vault, rungs=LADDER)
    )[2]
    assert _hist() == {}, _hist()
    assert pp_steps == PIPELINED_CHUNKS + [4]
    assert vault.depths() == [DEPTH]
    assert pr.METRICS.snapshot()["pp_ladder_collapsed"] == 1

    ref_manager, ref_vault = _apc_manager(), _FakeVault()
    with _pipeline_off():
        ref_steps = _drain(
            _apc_batch(_lm(), ref_manager, CKPT_SPLIT, vault=ref_vault, rungs=[DEPTH])
        )[2]
    assert ref_steps == pp_steps
    got, want = _apc_entries(manager), _apc_entries(ref_manager)
    assert sorted(got) == sorted(want) == [CKPT_SPLIT, len(PROMPT)]
    for length in sorted(want):
        assert got[length].token_ids == want[length].token_ids
        assert _entry_digest(got[length]) == _entry_digest(want[length])
    assert _fragments_digest(vault.inserts[0]["fragments"]) == _fragments_digest(
        ref_vault.inserts[0]["fragments"]
    )


def test_a_pp_turn_1_with_apc_exact_on_leaves_the_session_tier_alone(monkeypatch):
    """Item 4 with the new admission: the session rung is still the single-box
    one.  ``MLX_VLM_APC_SAVE_SESSION`` writes AFTER the response, from the
    finished cache, and the APC exact store must not have moved that cache."""
    from mlx_vlm import apc as _apc
    from mlx_vlm import context_vault as cv
    from mlx_vlm.generate.ar import BatchGenerator

    monkeypatch.setenv("MLX_VLM_APC_SAVE_SESSION", "1")
    assert cv.session_tier_active()
    picks, warm = {}, {}
    for arm in ("pipelined", "single_box"):
        pr.METRICS.reset()
        manager = _apc_manager()
        ctx = contextlib.nullcontext()
        if arm == "pipelined":
            _arm(monkeypatch)
        else:
            ctx = _pipeline_off()
        with ctx:
            batch = _apc_batch(_lm(), manager, CKPT_SPLIT)
            while batch.needs_processing():
                batch.prompt_step()
            first = {}

            def sampler(logprobs):
                first["tok"] = mx.argmax(logprobs, axis=-1)
                return first["tok"]

            gen = batch.generate(sampler=sampler, stop_criteria=lambda t: False)
            mx.eval(first["tok"])
        assert arm != "pipelined" or pr.METRICS.snapshot()["pp_used"] == 1
        key = PROMPT + [int(first["tok"][0].item())]
        vault = cv.ContextVault("identity-for-the-test", budget_bytes=1 << 30)
        assert cv.record_session_turn(
            vault,
            key,
            _apc.snapshot_prompt_cache_row(gen.prompt_cache, 0),
            completed=True,
            session_id="conv-1",
            adopt=False,
        ), cv.session_skip_counts()
        pick = BatchGenerator._vault_pick_for(
            _PickGen(vault), key + [97, 98, 99], {}, None
        )
        assert pick is not None, f"{arm}: turn 2 missed the session tier"
        picks[arm] = (pick["prefix_len"], pick.get("source"))
        warm[arm] = _cache_digest(pick["warm_cache"])
    assert picks["pipelined"] == picks["single_box"] == (len(PROMPT), "vault-session")
    assert warm["pipelined"] == warm["single_box"]


# ------------------------------- 8. A10-0: which warm requests A10 could serve
#
# ``warm_prefix`` was one name for two facts, and the difference between them
# decides whether the rest of A10 gets built at all.  PP's unit of work is a
# CHUNK: ``_pipeline_chunk_schedule`` hands the peer ``ceil(S/C) - 1`` chunks of
# ``C``, so a warm request whose uncached suffix ``S`` is at most ``C`` has zero
# chunks to give and would stay single-box under every version of A10 -- while
# one with ``S > C`` is exactly the shape the prefix push would pay for.  One
# name could not tell an operator which of those the fleet actually sends.
#
# This stage is INSTRUMENTATION ONLY.  Every request that stayed on one box
# still stays on one box; the two new names replace ``warm_prefix`` in the batch
# path and sum to what it counted, ``generate_step``'s bool call site keeps the
# old string, and the numbers the gate decided on are recorded per request and
# bucketed into ``/metrics``.


def _session_warm(monkeypatch, new_tokens):
    """A genuinely RESTORED cache plus the suffix that follows it.

    Turn 1 is prefilled for real and its end-of-turn rung is written to the
    session tier; ``_vault_pick_for`` hands back the same ``warm_cache`` /
    ``prefix_len`` the server would hand ``PromptProcessingBatch``.  So the warm
    facts under test come from the shipped restore path, not from a patched
    ``_apc_meta`` -- which is the one thing the parametrised table above cannot
    show.
    """
    from mlx_vlm import apc as _apc
    from mlx_vlm import context_vault as cv
    from mlx_vlm.generate.ar import BatchGenerator

    monkeypatch.setenv("MLX_VLM_APC_SAVE_SESSION", "1")
    lm = _lm()
    with _pipeline_off():
        batch = _batch(lm)
        while batch.needs_processing():
            batch.prompt_step()
        first = {}

        def sampler(logprobs):
            first["tok"] = mx.argmax(logprobs, axis=-1)
            return first["tok"]

        gen = batch.generate(sampler=sampler, stop_criteria=lambda t: False)
        mx.eval(first["tok"])
    key = PROMPT + [int(first["tok"][0].item())]
    vault = cv.ContextVault("identity-for-the-test", budget_bytes=1 << 30)
    assert cv.record_session_turn(
        vault,
        key,
        _apc.snapshot_prompt_cache_row(gen.prompt_cache, 0),
        completed=True,
        session_id="conv-1",
        adopt=False,
    ), cv.session_skip_counts()
    turn2 = key + list(range(45, 45 + int(new_tokens)))
    pick = BatchGenerator._vault_pick_for(_PickGen(vault, lm), turn2, {}, None)
    assert pick is not None and pick["prefix_len"] > 0, "turn 2 missed the session tier"
    return lm, pick, turn2


def _warm_batch(lm, pick, turn2, *, step=STEP):
    prefix_len = int(pick["prefix_len"])
    suffix = turn2[prefix_len:]
    batch = _batch(
        lm,
        [suffix],
        step=step,
        warm_cache=pick["warm_cache"],
        apc_meta=[{"prefix_len": prefix_len, "full_input_ids": list(turn2)}],
    )
    return batch, prefix_len, len(suffix)


@pytest.mark.parametrize(
    "new_tokens,step,expected",
    [
        (5, STEP, "warm_suffix_lt_chunk"),  # S = 6 <= C = 8
        (39, STEP, "warm_suffix_ge_chunk"),  # S = 40 > C = 8
        (39, 64, "warm_suffix_lt_chunk"),  # the SAME suffix, at a bigger C
    ],
)
def test_a_restored_cache_is_named_by_its_suffix(
    monkeypatch, new_tokens, step, expected
):
    """The name follows S vs C, on a cache the session tier actually restored."""
    lm, pick, turn2 = _session_warm(monkeypatch, new_tokens)
    _arm(monkeypatch, min_tokens=1)
    pr.METRICS.reset()
    batch, prefix_len, suffix_len = _warm_batch(lm, pick, turn2, step=step)
    batch._pipeline_open()
    assert batch._pipeline is None, "A10-0 changes no behaviour: still single-box"
    assert _hist() == {expected: 1}, _hist()
    # and the name agrees with the arithmetic it claims to predict
    assert (suffix_len > step) == (expected == "warm_suffix_ge_chunk")
    assert bool(batch._pipeline_chunk_schedule()) == (suffix_len > step)


def test_the_warm_split_is_recorded_per_request(monkeypatch):
    """``warm_prefix_len`` / ``warm_suffix_len`` / ``chunk_size``, on the batch.

    The per-request record is what makes the aggregate auditable: a histogram
    bucket with no way to see the numbers behind it cannot be checked against
    the restore that produced it.
    """
    lm, pick, turn2 = _session_warm(monkeypatch, 39)
    _arm(monkeypatch, min_tokens=1)
    pr.METRICS.reset()
    batch, prefix_len, suffix_len = _warm_batch(lm, pick, turn2)
    assert batch._pipeline_warm_stats is None, "nothing recorded before the gate"
    batch._pipeline_open()
    assert batch._pipeline_warm_stats == {
        "warm_prefix_len": prefix_len,
        "warm_suffix_len": suffix_len,
        "chunk_size": STEP,
    }
    # the prefix is the RESTORED depth and the suffix is the new user message,
    # so the two of them are the whole turn-2 prompt and neither is the prompt
    assert prefix_len + suffix_len == len(turn2)
    assert prefix_len == len(PROMPT)


def test_a_cold_request_records_no_warm_split(monkeypatch):
    """The recorder fires on the warm arm and nowhere else."""
    _arm(monkeypatch)
    batch = _batch(_lm())
    batch._pipeline_open()
    assert batch._pipeline is not None, "the cold request is the admitted one"
    assert batch._pipeline_warm_stats is None
    snap = pr.METRICS.snapshot()
    assert snap["pp_warm_requests"] == 0
    assert snap["pp_warm_suffix_tokens_hist"] == {}


def test_a_warm_request_a_shape_reason_refuses_first_is_not_counted_warm(monkeypatch):
    """``pp_warm_requests`` stays reconcilable with the bypass histogram.

    The gate names ONE refusal and asks the ladder before warm, so a warm
    request carrying an unserveable rung is recorded as ``apc_checkpoint_ladder``
    -- exactly as it was under ``warm_prefix``.  Counting it warm as well would
    make ``pp_warm_requests`` bigger than the two warm names put together, which
    is the one property that lets an operator trust either number.
    """
    _arm(monkeypatch)
    batch = _batch(_lm())
    batch._vault = object()
    batch._apc_meta = [{"prefix_len": 4, "vault_rungs": [12]}]
    batch._pipeline_open()
    assert _hist() == {"apc_checkpoint_ladder": 1}
    assert batch._pipeline_warm_stats is None
    assert pr.METRICS.snapshot()["pp_warm_requests"] == 0


def test_the_two_names_sum_to_what_warm_prefix_counted(monkeypatch):
    """Additive, not lossy: every request that WAS ``warm_prefix`` is still one.

    Six warm requests over the two shapes; the histogram must carry no
    ``warm_prefix`` key, the two new keys must sum to six, and
    ``pp_warm_requests`` must equal that sum.
    """
    _arm(monkeypatch, min_tokens=1)
    lm = _lm()
    for step, count in ((STEP, 4), (64, 2)):
        for _ in range(count):
            batch = _batch(lm, step=step)
            batch._apc_meta = [{"prefix_len": 4}]
            batch._pipeline_open()
            assert batch._pipeline is None
    hist = _hist()
    assert "warm_prefix" not in hist
    assert hist == {"warm_suffix_ge_chunk": 4, "warm_suffix_lt_chunk": 2}, hist
    snap = pr.METRICS.snapshot()
    assert snap["pp_warm_requests"] == sum(hist.values()) == 6
    assert sum(snap["pp_warm_suffix_tokens_hist"].values()) == 6


def test_generate_step_keeps_the_historical_warm_prefix_name():
    """Only the BATCH path is split.

    ``generate_step`` passes a bool -- it has no suffix length to hand over --
    and its reason strings are a receipt other rails already read, so the split
    must not reach it.  Pinned as a call, not as a comment.
    """
    kw = dict(
        ladder=False,
        capture=False,
        pixel_values=None,
        mask=None,
        cache=[],
        input_ids=mx.zeros((1, 4), dtype=mx.int32),
        kv_quantized=False,
    )
    assert pr.pipeline_bypass_reason(**kw, warm=True) == "warm_prefix"
    assert pr.pipeline_bypass_reason(**kw, warm=False) is None
    # and the batch path's own strings pass through untouched
    for name in ("warm_suffix_lt_chunk", "warm_suffix_ge_chunk"):
        assert pr.pipeline_bypass_reason(**kw, warm=name) == name


@pytest.mark.parametrize(
    "suffix_len,chunk,bucket",
    [
        (0, 8192, "le_512"),
        (512, 8192, "le_512"),
        (513, 8192, "le_2048"),
        (2048, 8192, "le_2048"),
        (2049, 8192, "le_c"),
        (8192, 8192, "le_c"),
        (8193, 8192, "le_2c"),
        (16384, 8192, "le_2c"),
        (16385, 8192, "le_4c"),
        (32768, 8192, "le_4c"),
        (32769, 8192, "gt_4c"),
        # C = 2048: the absolute edges catch first, so ``le_c`` is unreachable
        (2048, 2048, "le_2048"),
        (4096, 2048, "le_2c"),
        (8192, 2048, "le_4c"),
        (8193, 2048, "gt_4c"),
        # no chunked prefill configured: only the absolute edges are defined
        (2049, 0, "gt_4c"),
    ],
)
def test_the_warm_suffix_buckets_are_first_match_in_order(suffix_len, chunk, bucket):
    assert pr.warm_suffix_bucket(suffix_len, chunk) == bucket
    assert bucket in pr.WARM_SUFFIX_BUCKETS


def test_the_metrics_snapshot_carries_the_warm_keys(monkeypatch):
    """``/metrics`` -> ``server.pipeline_prefill`` is where the gate is read.

    The A10-0 decision -- build A10-1..7 only if ``warm_suffix_ge_chunk`` is a
    real share of served prefills -- is taken off a diff of this block, so the
    two keys have to be in the snapshot the server actually exports, and the
    counter has to be diffable (monotone over the process, reset only by
    ``reset()``).
    """
    snap = pr.pipeline_metrics_snapshot()
    assert snap["pp_warm_requests"] == 0
    assert snap["pp_warm_suffix_tokens_hist"] == {}
    pr.note_pipeline_warm(prefix_len=131072, suffix_len=32768, chunk_size=8192)
    pr.note_pipeline_warm(prefix_len=32768, suffix_len=500, chunk_size=8192)
    snap = pr.pipeline_metrics_snapshot()
    assert snap["pp_warm_requests"] == 2
    assert snap["pp_warm_suffix_tokens_hist"] == {"le_4c": 1, "le_512": 1}
    # and the per-request record is the return value, so a caller never has to
    # read the aggregate back to learn what it just recorded
    assert pr.note_pipeline_warm(
        prefix_len=1, suffix_len=2, chunk_size=3
    ) == {"warm_prefix_len": 1, "warm_suffix_len": 2, "chunk_size": 3}
