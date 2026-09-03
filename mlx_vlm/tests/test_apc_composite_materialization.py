"""CPU regression for producer-thread materialization of composite fragments."""

from __future__ import annotations

import queue
import tempfile
import threading
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx_vlm import apc_adapters as A
from mlx_vlm import harvest_provenance as HP
from mlx_vlm.apc_adapters import Capability, CompositeAdapter, StateFragment
from mlx_vlm.context_vault import ContextVault, VaultTier
from mlx_vlm.context_vault_wire import plan_fragments
from mlx_vlm.models.cache import CacheList
from mlx_vlm.vault_disk import DiskPrefixVault, DiskVaultStats


class _StateCache:
    def __init__(self, state):
        self.state = state
        self.meta_state = ""


def _mixed_composite_cache():
    """Nested CompositeAdapter children with dict/list/tuple state leaves."""
    left = _StateCache(
        {
            "dense": mx.arange(128, dtype=mx.float16).reshape(8, 16)[:, ::2],
            "empty": mx.zeros((1, 1, 8, 0), dtype=mx.float32),
        }
    )
    right = _StateCache(
        [
            mx.arange(64, dtype=mx.bfloat16).reshape(4, 16)[:, ::2],
            ("metadata", mx.ones((8, 8), dtype=mx.float32) + 1),
        ]
    )
    return (left, CacheList(right))


def _expected_payload_bytes():
    """Independent oracle for the disk round-trip; never touches the capture."""
    arrays = [
        mx.arange(128, dtype=mx.float16).reshape(8, 16)[:, ::2],
        mx.zeros((1, 1, 8, 0), dtype=mx.float32),
        mx.arange(64, dtype=mx.bfloat16).reshape(4, 16)[:, ::2],
        mx.ones((8, 8), dtype=mx.float32) + 1,
    ]
    flats = [mx.view(mx.contiguous(array).flatten(), mx.uint8) for array in arrays]
    mx.eval(flats)
    return b"".join(bytes(memoryview(flat)) for flat in flats)


def test_eval_targets_walks_nested_statefragments_without_expanding_aliases():
    """Traversal agrees with the wire walker for special leaves and aliases."""
    fp16_view = mx.arange(32, dtype=mx.float16).reshape(4, 8)[:, ::2]
    zero_width = mx.zeros((1, 2, 0), dtype=mx.float32)
    bf16 = mx.ones((3, 4), dtype=mx.bfloat16)
    nested = StateFragment(
        Capability.COMPOSITE,
        3,
        payload=[
            StateFragment(Capability.CHECKPOINT, 3, {"fp16": fp16_view, "zero": zero_width}),
            {"bf16": StateFragment(Capability.CHECKPOINT, 3, [bf16, (A._ALIAS_TAG, 0)])},
        ],
    )

    targets = nested.eval_targets()
    assert [(tuple(a.shape), a.dtype) for a in targets] == [
        ((4, 4), mx.float16),
        ((1, 2, 0), mx.float32),
        ((3, 4), mx.bfloat16),
    ]
    mx.eval(targets)


def test_composite_capture_materializes_nested_statefragments_before_writer_handoff():
    """A disk-writer thread can flatten a composite captured on another stream.

    ``ContextVault.insert`` is the producer-side boundary.  It must force all
    nested fragment arrays there; otherwise a later writer-side ``mx.eval``
    reaches a foreign producer stream even while that producer remains alive.
    """
    checkpoints: queue.Queue = queue.Queue()
    writer_done = threading.Event()
    records = {}

    def producer():
        mx.set_default_device(mx.cpu)
        try:
            stream = mx.new_stream(mx.cpu)
            with mx.stream(stream):
                fragment = CompositeAdapter().capture(_mixed_composite_cache(), 8)
                assert fragment is not None
                # The pre-fix value is zero; keep running so the writer seam,
                # rather than this census alone, observes the bad handoff.
                records["producer_targets"] = len(fragment.eval_targets())
                vault = ContextVault("r31-composite-cpu", budget_bytes=4 << 20)
                assert vault.insert(list(range(8)), 8, [fragment])
                checkpoints.put(vault.lookup(list(range(8))))
                assert writer_done.wait(10), "writer did not finish"
                vault.clear()
        except BaseException as exc:  # surface worker errors in the test thread
            records["producer_error"] = exc
        finally:
            mx.clear_streams()

    def writer():
        mx.set_default_device(mx.cpu)
        try:
            checkpoint = checkpoints.get(timeout=10)
            assert checkpoint is not None
            manifest, flats = plan_fragments(checkpoint.fragments)
            assert manifest["total_bytes"] == 448, manifest
            mx.eval(flats)
            records["writer_bytes"] = sum(int(flat.size) for flat in flats)
        except BaseException as exc:  # surface worker errors in the test thread
            records["writer_error"] = exc
        finally:
            mx.clear_streams()
            writer_done.set()

    producer_thread = threading.Thread(target=producer)
    writer_thread = threading.Thread(target=writer)
    producer_thread.start()
    writer_thread.start()
    producer_thread.join(15)
    writer_thread.join(15)

    assert not producer_thread.is_alive()
    assert not writer_thread.is_alive()
    assert records == {"producer_targets": 4, "writer_bytes": 448}, records


def test_composite_eviction_persists_through_the_real_async_disk_writer():
    """The producer-side insert makes a fresh composite safe for disk handoff."""
    with tempfile.TemporaryDirectory(prefix="r31-composite-") as root:
        disk = DiskPrefixVault(
            Path(root),
            "r31-composite-cpu",
            cap_bytes=4 << 20,
            chunk_bytes=4 << 20,
            fsync=False,
            nocache=False,
            strict_git=False,
            stats=DiskVaultStats(),
        )
        # DiskPrefixVault owns this actual writer thread.  Pin its CPU default
        # immediately before the real _write_entry body, without changing the
        # production queue-only save_async contract.
        write_entry = disk._write_entry

        def cpu_write_entry(*args):
            mx.set_default_device(mx.cpu)
            return write_entry(*args)

        disk._write_entry = cpu_write_entry
        producer_ready = threading.Event()
        producer_release = threading.Event()
        records = {}

        def producer():
            mx.set_default_device(mx.cpu)
            try:
                stream = mx.new_stream(mx.cpu)
                with mx.stream(stream):
                    fragment = CompositeAdapter().capture(_mixed_composite_cache(), 8)
                    assert fragment is not None
                    vault = ContextVault("r31-composite-cpu", budget_bytes=448)
                    vault.disk = disk
                    provenance = HP.make(1)
                    assert vault.insert(list(range(8)), 8, [fragment], harvest_provenance=provenance)
                    # A distinct insert forces eviction of the just-captured
                    # composite.  No warm restore or explicit producer eval
                    # occurs between capture and save_async.
                    second = StateFragment(
                        Capability.CHECKPOINT, 8, payload=[mx.zeros((112,), dtype=mx.float32)]
                    )
                    assert vault.insert(list(range(100, 108)), 8, [second], harvest_provenance=provenance)
                    producer_ready.set()
                    assert producer_release.wait(10), "writer did not finish"
                    vault.clear()
            except BaseException as exc:  # surface worker errors in the test thread
                records["producer_error"] = exc
            finally:
                mx.clear_streams()

        producer_thread = threading.Thread(target=producer)
        try:
            producer_thread.start()
            assert producer_ready.wait(10), records
            assert disk.flush(10), "background writer did not drain"
            records["save_errors"] = disk.stats.snapshot()["disk_save_errors"]
            records["entry_count"] = len(disk.records())
            loaded = disk.load_entry(
                disk.records()[0]["key"], expect_prompt_sha=None, tier=VaultTier.PREFILL
            )
            assert loaded is not None
            _header, loaded_fragments, _tokens = loaded
            manifest, flats = plan_fragments(loaded_fragments)
            mx.eval(flats)
            records["dtypes"] = [record["dtype"] for record in manifest["offsets"]]
            records["payload_bytes"] = manifest["total_bytes"]
            records["roundtrip_bytes"] = (
                b"".join(bytes(memoryview(flat)) for flat in flats)
                == _expected_payload_bytes()
            )
        finally:
            producer_release.set()
            producer_thread.join(15)
            disk.close(timeout=10)

    assert not producer_thread.is_alive()
    assert records == {
        "save_errors": 0,
        "entry_count": 1,
        "dtypes": ["float16", "float32", "bfloat16", "float32"],
        "payload_bytes": 448,
        "roundtrip_bytes": True,
    }, records
