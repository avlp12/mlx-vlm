"""CPU-only regressions for the opt-in PP transport contract."""

import copy
import socket
import threading

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx_vlm.pipeline_prefill import (
    PrefillEnvelope,
    _recv_json,
    _send_json,
    handoff_recv,
    handoff_send,
    install_state,
)
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache


def envelope():
    return PrefillEnvelope.create(
        model_sha256="a" * 64,
        source_revision="b" * 40,
        split=1,
        n_layers=3,
        input_ids=mx.array([[1, 2, 3]]),
        chunk=2,
    )


def caches():
    a = ArraysCache(size=2)
    a.state = [mx.ones((1, 2, 2)), mx.ones((1, 1, 2, 2))]
    b = CacheList(KVCache(), KVCache())
    b.state = [
        [mx.ones((1, 1, 3, 2)), mx.ones((1, 1, 3, 2))],
        [mx.ones((1, 1, 3, 3)), mx.zeros((1, 1, 3, 0))],
    ]
    return [None, a, b]


def schema_meta():
    from mlx_vlm.pipeline_prefill import collect_state

    return [dict(layer=i, desc=d) for i, d, _ in collect_state(caches())]


def test_envelope_roundtrip_binds_exact_prefix_and_schedule():
    e = envelope()
    assert e.depth == 3 and e.chunks == (2, 1)
    assert PrefillEnvelope.from_dict(e.to_dict()) == e
    for field, value in [("depth", 4), ("batch", 2), ("schema", 0), ("chunks", [3])]:
        raw = e.to_dict()
        raw[field] = value
        with pytest.raises(ValueError):
            PrefillEnvelope.from_dict(raw).require_match(e)


def test_handoff_roundtrip_and_fresh_atomic_install():
    e = envelope()
    left, right = socket.socketpair()
    left.settimeout(2)
    right.settimeout(2)
    errors = []

    def send():
        try:
            handoff_send(left, caches(), envelope=e)
        except Exception as exc:
            errors.append(exc)

    th = threading.Thread(target=send)
    th.start()
    try:
        result = handoff_recv(
            right, rebuild=True, expected=e, expected_meta=schema_meta()
        )
        old = caches()
        old[2][1]._pool = "stale"
        fresh = [None, ArraysCache(size=2), CacheList(KVCache(), KVCache())]
        install_state(
            old,
            result["states"],
            fresh_caches=fresh,
            expected=e,
            expected_meta=schema_meta(),
        )
        assert old[2] is fresh[2]
        assert not hasattr(old[2][1], "_pool")
        assert old[2][1].offset == 3
        assert bool(mx.all(old[1].state[0] == 1))
        assert result["handoff_tensors"] == 6
    finally:
        th.join(2)
        left.close()
        right.close()
    assert not th.is_alive() and not errors


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_sha256", "c" * 64),
        ("source_revision", "c" * 40),
        ("token_sha256", "d" * 64),
        ("request_id", "e" * 32),
    ],
)
def test_reject_identity_before_payload(field, value):
    e = envelope()
    raw = e.to_dict()
    raw[field] = value
    left, right = socket.socketpair()
    try:
        _send_json(left, {"cmd": "handoff", "envelope": raw, "meta": []})
        with pytest.raises(ValueError, match="mismatch"):
            handoff_recv(right, rebuild=True, expected=e, expected_meta=schema_meta())
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda meta: meta.pop(),
        lambda meta: meta.append(copy.deepcopy(meta[0])),
        lambda meta: meta[0].update(layer=0),
        lambda meta: meta[0]["desc"]["items"][0].update(nbytes=999),
        lambda meta: meta[0]["desc"]["items"][0].update(dtype="unknown"),
        lambda meta: meta[0]["desc"]["items"][0].update(shape=[-1]),
    ],
)
def test_reject_malformed_metadata_before_payload(mutate):
    from mlx_vlm.pipeline_prefill import collect_state

    e = envelope()
    meta = [{"layer": i, "desc": d} for i, d, _ in collect_state(caches())]
    mutate(meta)
    left, right = socket.socketpair()
    right.settimeout(0.1)
    try:
        _send_json(left, {"cmd": "handoff", "envelope": e.to_dict(), "meta": meta})
        with pytest.raises(ValueError):
            handoff_recv(right, rebuild=True, expected=e, expected_meta=schema_meta())
    finally:
        left.close()
        right.close()


def test_bad_depth_does_not_partially_install():
    old = caches()
    saved = list(old)
    states = {1: old[1].state, 2: old[2].state}
    states[2][1] = [mx.zeros((1, 1, 2, 3)), states[2][1][1]]
    fresh = [None, ArraysCache(2), CacheList(KVCache(), KVCache())]
    with pytest.raises(ValueError, match="shape"):
        install_state(
            old,
            states,
            fresh_caches=fresh,
            expected=envelope(),
            expected_meta=schema_meta(),
        )
    assert all(a is b for a, b in zip(old, saved))


def test_socket_timeout_is_bounded():
    left, right = socket.socketpair()
    right.settimeout(0.01)
    try:
        with pytest.raises(TimeoutError):
            _recv_json(right)
    finally:
        left.close()
        right.close()


def test_repeated_runtime_requests_reset_cache_stats_and_exit_tail(monkeypatch):
    """Real loopback sockets and queues; tiny fake forwards, no model load."""
    from types import SimpleNamespace
    from mlx_vlm import pipeline_prefill as pp
    from mlx_vlm.pipeline_runtime import PipelineHead, PipelineSettings

    class FakeModel:
        def __init__(self):
            layers = [
                SimpleNamespace(
                    is_linear=i != 2,
                    input_layernorm=SimpleNamespace(weight=mx.ones((2,))),
                    self_attn=SimpleNamespace(
                        num_heads=1,
                        head_dim=2,
                        conv_kernel_size=3,
                        kv_lora_rank=2,
                        indexer=SimpleNamespace(head_dim=1),
                    ),
                )
                for i in range(3)
            ]
            self.lm = SimpleNamespace(
                layers=layers,
                hc_mult=1,
                pipeline_forward=self.forward,
                pipeline_finish=lambda h: h[:, :, 0, :],
            )
            self.language_model = SimpleNamespace(
                model=self.lm, pipeline_prefill_head=self.head, _logits=lambda h: h
            )

        def make_cache(self):
            return [ArraysCache(2), ArraysCache(2), CacheList(KVCache(), KVCache())]

        def forward(self, h, cache, lo, hi, inputs=None):
            if h is None:
                h = mx.ones((1, inputs.shape[1], 1, 2), dtype=mx.bfloat16)
            for i in range(lo, hi):
                if i != 2:
                    cache[i].state = [
                        mx.ones((1, 2, 6), dtype=mx.bfloat16),
                        mx.ones((1, 1, 2, 2)),
                    ]
                else:
                    n = h.shape[1]
                    cache[i][0].update_and_fetch(
                        mx.ones((1, 1, n, 2), dtype=mx.bfloat16),
                        mx.ones((1, 1, n, 2), dtype=mx.bfloat16),
                    )
                    cache[i][1].update_and_fetch(
                        mx.ones((1, 1, n, 3), dtype=mx.bfloat16),
                        mx.zeros((1, 1, n, 0), dtype=mx.float32),
                    )
            return h

        def head(self, inputs, inputs_embeds, cache, split):
            return self.forward(None, cache, 0, split, inputs)

    tail_model = FakeModel()
    monkeypatch.setattr(
        pp,
        "load_stage",
        lambda *a: (tail_model, tail_model.make_cache(), [1, 2], 3, 0.0),
    )
    reservation = socket.socket()
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
    reservation.close()
    args = SimpleNamespace(
        model="unused",
        split=1,
        layers=3,
        prune=False,
        bind="127.0.0.1",
        port=port,
        model_sha256="a" * 64,
        source_revision="b" * 40,
        connect_timeout=2.0,
        io_timeout=2.0,
        depth=2,
        transport="socket",
    )
    errors = []

    def tail():
        try:
            pp.run_tail(args)
        except BaseException as exc:
            errors.append(exc)

    th = threading.Thread(target=tail)
    th.start()
    head = PipelineHead(
        PipelineSettings(
            ("127.0.0.1", port),
            None,
            "1",
            1,
            "unused",
            "socket",
            model_sha256="a" * 64,
            source_revision="b" * 40,
            io_timeout=2.0,
        ),
        1,
        3,
    )
    requests = []
    try:
        head.connect(timeout=2)
        model = FakeModel()
        for tokens in (4, 6):
            ids = mx.arange(tokens, dtype=mx.int32)[None, :]
            cache = model.make_cache()
            cache[2][1]._pool = "stale"
            head.begin(tokens, 2, input_ids=ids)
            for start in range(0, tokens - 1, 2):
                part = ids[:, start : min(start + 2, tokens - 1)]
                head.prefill_chunk(model, part, None, cache)
            stats = head.finalize(cache)
            assert sum(stats["chunks"]) == tokens - 1
            assert cache[2][0].offset == tokens - 1
            assert not hasattr(cache[2][1], "_pool")
            assert stats["tail"]["handoff"]["handoff_pack_s"] >= 0
            requests.append(stats["envelope"]["request_id"])
        assert len(set(requests)) == 2
        head.close()
    finally:
        head.abort()
        th.join(3)
    assert not th.is_alive() and not errors


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"ladder": True}, "apc_checkpoint_ladder"),
        ({"capture": True}, "speculative_hidden_capture"),
        ({"warm": True}, "warm_prefix"),
        ({"kv_quantized": True}, "quantized_kv"),
        ({"mask": mx.array([[1, 0]])}, "padded_or_custom_mask"),
    ],
)
def test_existing_capture_and_unsupported_state_bypass(override, reason):
    from mlx_vlm.pipeline_runtime import pipeline_bypass_reason

    args = dict(
        ladder=False,
        capture=False,
        warm=False,
        pixel_values=None,
        mask=None,
        cache=[ArraysCache(2)],
        input_ids=mx.array([[1, 2]]),
        kv_quantized=False,
    )
    args.update(override)
    assert pipeline_bypass_reason(**args) == reason


def test_sender_failure_does_not_hang_full_queue():
    import queue
    from mlx_vlm.pipeline_prefill import _queue_put

    q = queue.Queue(1)
    q.put(1)
    with pytest.raises(RuntimeError, match="worker failed"):
        _queue_put(q, 2, ["disconnected"], 0.1)
    with pytest.raises(TimeoutError, match="progress timeout"):
        _queue_put(q, 2, [], 0.01)


def test_fresh_glm_cachelist_bypasses_no_state_property():
    from mlx_vlm.pipeline_runtime import pipeline_bypass_reason

    assert (
        pipeline_bypass_reason(
            ladder=False,
            capture=False,
            warm=False,
            pixel_values=None,
            mask=None,
            cache=[ArraysCache(2), CacheList(KVCache(), KVCache())],
            input_ids=mx.array([[1, 2]]),
            kv_quantized=False,
        )
        is None
    )


@pytest.mark.parametrize("change", ["dtype", "shape"])
def test_coherent_but_wrong_model_schema_rejected_before_payload(change):
    meta = schema_meta()
    d = meta[0]["desc"]["items"][0]
    if change == "dtype":
        d["dtype"] = "float16"
        d["nbytes"] //= 2
    else:
        d["shape"][-1] *= 2
        d["nbytes"] *= 2
    left, right = socket.socketpair()
    right.settimeout(0.1)
    try:
        e = envelope()
        _send_json(left, dict(cmd="handoff", envelope=e.to_dict(), meta=meta))
        with pytest.raises(ValueError, match="schema"):
            handoff_recv(right, rebuild=True, expected=e, expected_meta=schema_meta())
    finally:
        left.close()
        right.close()


def test_tail_local_stop_interrupts_idle_socket(tmp_path):
    import time
    from mlx_vlm.pipeline_prefill import StopAwareSocket

    stop = tmp_path / "STOP"
    left, right = socket.socketpair()
    peer = StopAwareSocket(right, stop, 20.0)
    timer = threading.Timer(0.02, lambda: stop.write_text("rail"))
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError, match="STOP"):
            _recv_json(peer)
        assert time.monotonic() - started < 1.0
    finally:
        timer.join()
        left.close()
        peer.close()


@pytest.mark.parametrize("fail", [False, True])
def test_generate_step_admits_cold_cache_and_cleans_up(monkeypatch, fail):
    import sys
    from unittest.mock import MagicMock
    from mlx_vlm import pipeline_runtime
    from mlx_vlm.generate import ar

    generate_module = sys.modules["mlx_vlm.generate"]
    monkeypatch.setenv("MLX_VLM_PIPELINE_HOSTS", "127.0.0.1:1")
    model = MagicMock()
    model.no_chunked_prefill = False
    model.language_model.return_value = MagicMock(
        logits=mx.zeros((1, 1, 4)), cross_attention_states=None, encoder_outputs=None
    )
    embedding = MagicMock()
    embedding.inputs_embeds = mx.zeros((1, 5, 4))
    embedding.to_dict.return_value = {}
    model.get_input_embeddings.return_value = embedding
    cold = [ArraysCache(2), CacheList(KVCache(), KVCache())]
    monkeypatch.setattr(
        generate_module.cache, "make_prompt_cache", lambda *a, **k: cold
    )
    monkeypatch.setattr(generate_module, "make_logits_processors", lambda *a, **k: [])
    monkeypatch.setattr(
        generate_module, "make_sampler", lambda **k: lambda _: mx.array([0])
    )
    pipeline = MagicMock()
    pipeline.local_caches.return_value = []
    if fail:
        pipeline.prefill_chunk.side_effect = RuntimeError("peer stopped")
    opened = MagicMock(return_value=pipeline)
    monkeypatch.setattr(pipeline_runtime, "maybe_open_pipeline", opened)
    ids = mx.array([[1, 2, 3, 4, 5]])
    gen = ar.generate_step(ids, model, None, None, max_tokens=1, prefill_step_size=2)
    if fail:
        with pytest.raises(RuntimeError, match="peer stopped"):
            next(gen)
        pipeline.finalize.assert_not_called()
        model.language_model.assert_not_called()
    else:
        next(gen)
        assert pipeline.prefill_chunk.call_count == 2
        pipeline.finalize.assert_called_once_with(cold)
    opened.assert_called_once()
    pipeline.begin.assert_called_once()
    assert bool(mx.all(pipeline.begin.call_args.kwargs["input_ids"] == ids))
    pipeline.close.assert_called_once()
    gen.close()


def test_exact_schema_matches_real_tiny_glm_bf16_prefill():
    """Exercise actual KDA/DSA cache writers with tiny random CPU weights."""
    from types import SimpleNamespace
    from mlx.utils import tree_map
    from mlx_vlm.tests.test_glm5_chunked_spec_prefill import _tiny_text_config
    from mlx_vlm.models.glm5_next.language import LanguageModel
    from mlx_vlm.pipeline_prefill import (
        expected_state_meta,
        collect_state,
        _require_state_meta,
    )

    cfg = _tiny_text_config()
    cfg.num_hidden_layers = 3
    cfg.layer_types = [
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
    ]
    cfg.mlp_layer_types = ["dense", "dense", "dense"]
    lm = LanguageModel(cfg)
    lm.update(tree_map(lambda a: a.astype(mx.bfloat16), lm.parameters()))
    lm.eval()
    model = SimpleNamespace(language_model=lm)
    ids = mx.array([[1, 2, 3, 4]])
    e = PrefillEnvelope.create(
        model_sha256="a" * 64,
        source_revision="b" * 40,
        split=1,
        n_layers=3,
        input_ids=ids,
        chunk=2,
    )
    cache = lm.make_cache()
    for start in (0, 2):
        out = lm(ids[:, start : start + 2], cache=cache)
        mx.eval(out.logits, [c.state for c in cache])
    meta = [dict(layer=i, desc=d) for i, d, _ in collect_state(cache) if i >= e.split]
    _require_state_meta(meta, expected_state_meta(model, e))
    # The existing prototype head prunes tail modules; schema derivation must
    # still work from the retained architecture before any payload is received.
    lm.model.layers[1:] = [None, None]
    _require_state_meta(meta, expected_state_meta(model, e))
