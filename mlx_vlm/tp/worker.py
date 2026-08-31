"""Rank-1 side of TP=2 serving: env, control codec, preflight, worker loop.

Deliberately free of any dependency on ``mlx_vlm.server``.  The worker holds a
shard and executes forwards; it has no business importing an HTTP framework, and
on a box that only ever runs rank 1 those packages need not be installed at all.
``mlx_vlm.server.tp_mode`` re-exports from here for the rank-0 side.

CONTROL PLANE = A DATA COLLECTIVE
---------------------------------
Every verb below rides the *same* ``all_sum`` the sharded layers already need:
rank 0 fills a fixed-width int32 vector, rank 1 contributes zeros, and the sum
hands both the same message.  There is no side channel, so the control stream
cannot desynchronise from the data stream -- it *is* one of them.

One consequence is worth stating because it shapes every verb: the vector is
one-directional by construction (rank 1 contributes zeros).  When rank 0 needs
an *answer* -- "do you hold this vault rung?" -- the answer is a second
collective in which the roles are swapped, not a socket.  Same transport, same
ordering guarantee.  ``OP_VAULT_RESTORE`` is the only verb that needs one.
"""
from __future__ import annotations

import logging
import os
from typing import List, NamedTuple, Optional, Sequence

logger = logging.getLogger(__name__)

ENV_HOSTS = "MLX_VLM_GLM5_TP_HOSTS"
ENV_RANK = "MLX_VLM_GLM5_TP_RANK"
ENV_WORKER_PY = "MLX_VLM_GLM5_TP_WORKER_PYTHON"
ENV_WORKER_SRC = "MLX_VLM_GLM5_TP_WORKER_SRC"
ENV_MAX_TOK = "MLX_VLM_GLM5_TP_MAX_TOKENS_PER_FORWARD"
# The peer may hold the same checkpoint under a different path
# (different home directory); default to rank 0's path.
ENV_WORKER_MODEL = "MLX_VLM_GLM5_TP_WORKER_MODEL"

# Bumped whenever the wire layout below changes.  Both ranks check it in
# preflight over a vector whose width never changes, so a version skew is a
# clean refusal instead of a shape-mismatched collective that hangs.
PROTO_VERSION = 2

OP_EXIT, OP_MAKE_CACHE, OP_FORWARD = 0, 1, 2
OP_ROLLBACK = 3         # speculative round rejected: roll my own shard back
OP_VAULT_STORE = 4      # checkpoint my own shard's cache under this name
OP_VAULT_RESTORE = 5    # rebuild my own shard's cache from this name (+ ack)

# Fixed-width header, then ``n`` payload words.
#   0 op | 1 epoch | 2 batch | 3 seqlen | 4 flags | 5 arg0 | 6..13 name
IDX_OP, IDX_EPOCH, IDX_B, IDX_S, IDX_FLAGS, IDX_ARG0 = range(6)
IDX_NAME = 6
NAME_WORDS = 8          # 8 x 16 bits = the 128-bit boundary name, exactly
HEADER = IDX_NAME + NAME_WORDS   # 14

# Flags
FLAG_CAPTURE = 1 << 0   # this forward is a speculative verify: allocate gdn_sink
FLAG_HAS_NAME = 1 << 1  # the name words are a rung name, not padding


def tp_hosts() -> List[str]:
    raw = os.environ.get(ENV_HOSTS, "")
    return [h.strip() for h in raw.split(",") if h.strip()]


def tp_enabled() -> bool:
    return len(tp_hosts()) > 1


def tp_rank() -> int:
    return int(os.environ.get(ENV_RANK, "0"))


def _max_tok() -> int:
    try:
        return max(64, int(os.environ.get(ENV_MAX_TOK, "8192")))
    except ValueError:
        return 8192


class TPUnavailable(RuntimeError):
    """Raised when TP cannot be brought up; the caller serves single-box."""


class TPDesync(RuntimeError):
    """Rank 0 was asked to do something rank 1 could not be told to mirror.

    Distinct from :class:`TPUnavailable`: unavailability happens at startup and
    degrades to single-box, whereas this happens mid-serve, when a fallback is
    no longer free.  It is raised rather than papered over because the failure
    it prevents -- two ranks summing halves of different computations -- is
    silent, and produces plausible tokens.
    """


# ------------------------------------------------------------------ preflight
def preflight(hosts: List[str], rank: int, timeout_s: float = 60.0) -> dict:
    """Seconds-long transport check before anything expensive.

    A four-minute sharded load per box is a bad place to discover the group will
    not form, so the same check the stage-3/4 campaign used runs at startup.

    It also settles the protocol handshake.  ``_ctrl_recv`` allocates
    ``HEADER + n`` words and ``_ctrl_send`` sends the same; if the two boxes run
    different revisions of this file those widths differ and the very first
    control collective mismatches -- which on jaccl is a hang, not an error.
    The handshake rides a vector whose width is frozen forever, so it can
    diagnose the skew instead of becoming another instance of it.
    """
    import mlx.core as mx

    from ..tp.transport import Deadman, all_sum, backend, init_tp, tp_size

    init_tp(hosts=hosts, rank=rank, backend="jaccl")
    x = mx.full((1, 1, 64), float(rank + 1))
    with Deadman(timeout_s, "tp preflight all_sum"):
        y = all_sum(x)
        mx.eval(y)
    got = float(y[0, 0, 0])
    want = float(sum(range(1, tp_size() + 1)))
    if abs(got - want) > 1e-6:
        raise TPUnavailable(f"preflight all_sum {got} != {want}")

    proto = _proto_handshake(timeout_s)
    return {"size": tp_size(), "backend": backend(), "all_sum": got,
            "fast_synch": os.environ.get("MLX_METAL_FAST_SYNCH"), **proto}


# The handshake vector's width is part of the ABI and must never change.
_HANDSHAKE_WORDS = 8


def _proto_handshake(timeout_s: float = 60.0) -> dict:
    """Agree on (protocol version, header width, payload width) or refuse."""
    import mlx.core as mx

    from ..tp.transport import Deadman, all_sum, tp_size

    mine = [PROTO_VERSION, HEADER, _max_tok(), 0, 0, 0, 0, 0]
    with Deadman(timeout_s, "tp proto handshake"):
        out = all_sum(mx.array([mine], dtype=mx.int32))
        mx.eval(out)
    total = [int(v) for v in out[0].tolist()]
    n = tp_size()
    for i, label in enumerate(("proto_version", "header_width", "max_tokens")):
        if total[i] != mine[i] * n:
            peer = total[i] - mine[i] * (n - 1)
            raise TPUnavailable(
                f"tp {label} mismatch: this rank has {mine[i]}, the peer(s) "
                f"sum to {peer}. The two boxes are running different revisions "
                f"of mlx_vlm.tp.worker; update the peer checkout "
                f"({ENV_WORKER_SRC}) and retry.")
    return {"proto_version": PROTO_VERSION, "header_width": HEADER,
            "max_tokens": _max_tok()}


# ------------------------------------------------------------------ control
class Ctrl(NamedTuple):
    """A decoded control message."""

    op: int
    epoch: int
    batch: int
    seqlen: int
    ids: Optional[List[int]]
    flags: int = 0
    arg0: int = 0
    name: str = ""

    @property
    def capture(self) -> bool:
        return bool(self.flags & FLAG_CAPTURE)


def name_to_words(name: str) -> List[int]:
    """A 32-hex-char boundary name as 8 x 16-bit words.

    Sixteen bits per word keeps every value far inside int32 so the all_sum
    (rank 1 contributes zeros) reproduces rank 0's word exactly, with no
    overflow to reason about.
    """
    if not name:
        return [0] * NAME_WORDS
    h = name.strip().lower()
    if len(h) != 4 * NAME_WORDS or any(c not in "0123456789abcdef" for c in h):
        raise TPUnavailable(
            f"boundary name must be {4 * NAME_WORDS} hex chars, got {name!r}")
    return [int(h[i * 4:(i + 1) * 4], 16) for i in range(NAME_WORDS)]


def words_to_name(words: Sequence[int]) -> str:
    return "".join(f"{int(w) & 0xFFFF:04x}" for w in words[:NAME_WORDS])


def encode(op: int, epoch: int, shape=None, flat=None, n: Optional[int] = None,
           *, flags: int = 0, arg0: int = 0, name: str = ""):
    """Control message as a flat int32 vector. Separated from transport so the
    codec is testable without a group."""
    n = _max_tok() if n is None else n
    buf = [0] * (HEADER + n)
    b = s = 0
    if flat is not None:
        b, s = int(shape[0]), int(shape[1])
        if len(flat) > n:
            raise TPUnavailable(
                f"forward of {len(flat)} tokens exceeds {ENV_MAX_TOK}={n}")
        buf[HEADER:HEADER + len(flat)] = [int(v) for v in flat]
    buf[IDX_OP], buf[IDX_EPOCH], buf[IDX_B], buf[IDX_S] = int(op), int(epoch), b, s
    # A presence bit, not "is it nonzero": an all-zero hash is a legitimate
    # (if improbable) rung name, and reading it as "no name" would silently
    # restore the wrong boundary rather than fail.
    if name:
        flags |= FLAG_HAS_NAME
    buf[IDX_FLAGS], buf[IDX_ARG0] = int(flags), int(arg0)
    buf[IDX_NAME:IDX_NAME + NAME_WORDS] = name_to_words(name)
    return buf


def decode(row) -> Ctrl:
    """Inverse of :func:`encode`."""
    op, epoch = int(row[IDX_OP]), int(row[IDX_EPOCH])
    b, s = int(row[IDX_B]), int(row[IDX_S])
    ids = [int(v) for v in row[HEADER:HEADER + b * s]] if (b and s) else None
    flags = int(row[IDX_FLAGS])
    name = (words_to_name(row[IDX_NAME:IDX_NAME + NAME_WORDS])
            if flags & FLAG_HAS_NAME else "")
    return Ctrl(op, epoch, b, s, ids, flags=flags,
                arg0=int(row[IDX_ARG0]), name=name)


def _flatten(ids):
    """Token ids as (shape, flat list), from an mx.array or a plain sequence."""
    if ids is None:
        return None, None
    if hasattr(ids, "reshape") and hasattr(ids, "shape"):
        return (ids.shape[0], ids.shape[1]), ids.reshape(-1).tolist()
    flat = [int(v) for v in ids]
    return (1, len(flat)), flat


def _ctrl_send(op: int, epoch: int, ids, *, flags: int = 0, arg0: int = 0,
               name: str = "") -> None:
    """Rank 0: publish a control message through the data collective."""
    import mlx.core as mx

    from ..tp.transport import all_sum

    shape, flat = _flatten(ids)
    msg = mx.array([encode(op, epoch, shape, flat, flags=flags, arg0=arg0,
                           name=name)], dtype=mx.int32)
    out = all_sum(msg)
    mx.eval(out)


def _ctrl_recv() -> Ctrl:
    """Rank 1: contribute zeros and read the sum."""
    import mlx.core as mx

    from ..tp.transport import all_sum

    n = _max_tok()
    out = all_sum(mx.zeros((1, HEADER + n), dtype=mx.int32))
    mx.eval(out)
    return decode(out[0].tolist())


def _ack_send(value: int) -> None:
    """Rank 1: answer the question rank 0 just asked, over the same collective."""
    import mlx.core as mx

    from ..tp.transport import all_sum

    out = all_sum(mx.array([[int(value)] + [0] * (_HANDSHAKE_WORDS - 1)],
                           dtype=mx.int32))
    mx.eval(out)


def _ack_recv() -> int:
    """Rank 0: contribute zeros and read the peer's answer."""
    import mlx.core as mx

    from ..tp.transport import all_sum

    out = all_sum(mx.zeros((1, _HANDSHAKE_WORDS), dtype=mx.int32))
    mx.eval(out)
    return int(out[0].tolist()[0])


# ------------------------------------------------------------------- worker
_OP_NAMES = {OP_EXIT: "EXIT", OP_MAKE_CACHE: "MAKE_CACHE", OP_FORWARD: "FORWARD",
             OP_ROLLBACK: "ROLLBACK", OP_VAULT_STORE: "VAULT_STORE",
             OP_VAULT_RESTORE: "VAULT_RESTORE"}
ENV_TRACE = "MLX_VLM_GLM5_TP_TRACE"


def _trace() -> bool:
    """Log every control message rank 1 acts on.

    Off by default (one line per decode step is a lot at 160 tok/s), on when
    diagnosing a stall -- which is the only time the question "what did rank 1
    last hear?" can be answered any other way than by guessing.
    """
    return os.environ.get(ENV_TRACE, "") not in ("", "0", "false", "False")


class _WorkerState:
    """Rank 1's mirror of everything rank 0 announces.

    Kept as an object (rather than closure locals) so the loop body is a plain
    dispatch that unit tests can drive one message at a time without a group.
    """

    def __init__(self, lm, vault=None):
        self.lm = lm
        self.vault = vault
        self.caches: dict = {}
        self.epoch: int = -1
        # gdn_states from the last capturing forward: what a ROLLBACK replays.
        self.last_gdn = None
        self.tokens_seen: int = 0

    # -- helpers ----------------------------------------------------------
    def cache_for(self, epoch: int):
        c = self.caches.get(epoch)
        if c is None:
            c = self.caches[epoch] = self.lm.make_cache()
        return c

    def new_cache(self, epoch: int):
        # One live cache in mode-level TP: rank 0 drives one conversation at a
        # time, and holding the old one would only pin its KV.
        c = self.lm.make_cache()
        self.caches = {epoch: c}
        self.epoch = epoch
        self.last_gdn = None
        return c

    # -- verbs ------------------------------------------------------------
    def handle(self, msg: Ctrl) -> bool:
        """Apply one control message. Returns False when told to exit."""
        if _trace():
            logger.info("ctrl op=%s epoch=%s b=%s s=%s cap=%s arg0=%s name=%s",
                        _OP_NAMES.get(msg.op, msg.op), msg.epoch, msg.batch,
                        msg.seqlen, msg.capture, msg.arg0, msg.name[:12])
        if msg.op == OP_EXIT:
            logger.info("tp worker: EXIT")
            return False

        if msg.op == OP_MAKE_CACHE:
            self.new_cache(msg.epoch)
            return True

        if msg.op == OP_FORWARD:
            import mlx.core as mx

            c = self.cache_for(msg.epoch)
            ids = mx.array(msg.ids, dtype=mx.int32).reshape(msg.batch, msg.seqlen)
            # A capturing forward must be capturing on BOTH ranks.  The capture
            # itself is numerically inert -- the sinks are appended to, never
            # read back into the residual, and _fused_kda_eligible explicitly
            # ``del``s gdn_sink -- but ``gdn_sink is not None`` is what makes a
            # KDA layer stash the block inputs this rank needs to roll its own
            # half back.  Rank 1 passes the empty list: it allocates the sink
            # without capturing hidden states, which only rank 0's drafter reads.
            kw = {"capture_layer_ids": []} if msg.capture else {}
            out = self.lm(ids, cache=c, **kw)
            mx.eval(out.logits)
            self.last_gdn = getattr(out, "gdn_states", None) if msg.capture else None
            self.tokens_seen += msg.batch * msg.seqlen
            return True

        if msg.op == OP_ROLLBACK:
            c = self.caches.get(msg.epoch)
            if c is None or self.last_gdn is None:
                raise TPDesync(
                    "tp worker: ROLLBACK with no captured round to roll back "
                    f"(epoch={msg.epoch}, cache={'yes' if c else 'no'})")
            accepted = list(msg.ids or [])
            self.lm.rollback_speculative_cache(
                c, self.last_gdn, accepted, int(msg.arg0))
            self.last_gdn = None
            return True

        if msg.op == OP_VAULT_STORE:
            self._vault_store(msg)
            return True

        if msg.op == OP_VAULT_RESTORE:
            self._vault_restore(msg)
            return True

        raise TPDesync(f"tp worker: unknown control op {msg.op}")

    # -- vault ------------------------------------------------------------
    def _vault_store(self, msg: Ctrl) -> None:
        """Checkpoint THIS rank's shard of the cache under the announced name.

        Nothing crosses the wire: rank 0 stores its half, rank 1 stores its
        half, and the only thing they agree on is what to call the pair.
        """
        if self.vault is None:
            return
        c = self.caches.get(msg.epoch)
        if c is None:
            return
        try:
            self.vault.store(msg.name, int(msg.arg0), c)
        except Exception:  # noqa: BLE001 - storing is best-effort on both ranks
            logger.warning("tp worker: vault store failed", exc_info=True)

    def _vault_restore(self, msg: Ctrl) -> None:
        """Rebuild this rank's cache from its own shard-local rung, then ack.

        The ack is the reason this verb needs a second collective.  The two
        vaults evict independently (separate budgets, separate LRU), so rank 1
        holding the rung is a *fact rank 0 does not have*.  Answering over a
        swapped-roles all_sum keeps it on the same transport as everything
        else; without it rank 0 could serve warm while rank 1 served cold, and
        the sum of those two halves is a plausible-looking wrong answer.
        """
        ok = False
        if self.vault is not None and msg.name:
            c = self.lm.make_cache()
            try:
                ok = bool(self.vault.restore(msg.name, int(msg.arg0), c))
            except Exception:  # noqa: BLE001
                logger.warning("tp worker: vault restore failed", exc_info=True)
                ok = False
            if ok:
                self.caches = {msg.epoch: c}
                self.epoch = msg.epoch
                self.last_gdn = None
        if not ok:
            # Rank 0 will fall back to a cold prefill and announce MAKE_CACHE.
            logger.info("tp worker: vault MISS for %s; answering cold", msg.name[:12])
        _ack_send(1 if ok else 0)


def worker_loop(model_path: str, hosts: List[str], rank: int) -> None:
    """Rank 1: hold a shard and execute whatever rank 0 announces."""
    import mlx.core as mx

    from ..generate import wired_limit
    from ..tp.load import load_sharded, materialize
    from ..tp.transport import tp_rank as _r, tp_size
    from ..tp.vault import shard_vault
    from ..tp.vault import topology_descriptor

    info = preflight(hosts, rank)
    logger.info("tp worker: joined %s", info)
    model, report = load_sharded(model_path, _r(), tp_size())
    peak = materialize(model)
    logger.info("tp worker: sharded %s peak %.1f GiB", report, peak)
    lm = model.language_model if hasattr(model, "language_model") else model
    # The topology descriptor is what keeps this rank's store from ever being
    # confused with rank 0's or with a single-box store.
    vault = shard_vault(topology_descriptor(report, model_path))
    wire = wired_limit(model, [mx.default_stream(mx.default_device())])
    wire.__enter__()
    state = _WorkerState(lm, vault=vault)
    try:
        while True:
            if not state.handle(_ctrl_recv()):
                return
    finally:
        # Order matters: leave wired_limit FIRST (it synchronises the stream and
        # puts the wired budget back), then drop the shard, then clear.  Skipping
        # this is what leaks 85.5 GiB of wired pages that survive the process.
        try:
            wire.__exit__(None, None, None)
        except Exception:
            logger.warning("tp worker: releasing wired limit failed",
                           exc_info=True)
        state.caches.clear()
        state.last_gdn = None
        state.vault = None
        lm = None
        model = None
        try:
            import gc

            gc.collect()
            mx.clear_cache()
            mx.set_wired_limit(0)
        except Exception:
            pass
        logger.info("tp worker: shard released")


def _install_clean_exit_handlers() -> None:
    """Turn SIGTERM/SIGINT into SystemExit so ``worker_loop``'s finally runs.

    THIS IS NOT COSMETIC.  Python's default SIGTERM disposition terminates the
    process without unwinding, so ``wired_limit.__exit__`` never runs and the
    shard is never dropped -- and the pages stay WIRED after the process is
    gone.  Measured 2026-09-01 on the peer: wired 206.3 GB / free 112.0 GB
    before, wired 305.6 GB / free 9.5 GB after a single SIGTERM of a worker
    holding an 85.5 GiB shard.  The whole shard leaked, permanently, and the
    box could no longer take a load.

    The clean path (rank 0 announces OP_EXIT, the loop returns, the finally
    runs) releases fully -- the same box sat at 7.1 GB wired after a clean
    cycle.  So the fix is simply to make the signal path take the clean path.

    Caveat, and it is the same one as everywhere else here: a process blocked
    inside a collective cannot run a Python signal handler until the collective
    returns.  This makes the ordinary case correct; it cannot rescue a wedged
    one.
    """
    import signal

    def _bail(signum, _frame):
        logger.info("tp worker: signal %s -- unwinding so the shard is released",
                    signum)
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass


def main():
    import argparse
    _install_clean_exit_handlers()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s rank%(rank)s %(message)s"
                        .replace("%(rank)s", str(tp_rank())),
                        force=True)
    # Belt and braces with the launcher's -u: a peer whose last words are lost
    # is a peer that cannot be diagnosed at all.
    for h in logging.getLogger().handlers:
        try:
            h.setStream(getattr(h, "stream", None)) if False else None
            h.flush()
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    hosts = tp_hosts()
    if len(hosts) < 2:
        raise SystemExit(f"{ENV_HOSTS} must list >= 2 hosts")
    worker_loop(a.model, hosts, tp_rank())


if __name__ == "__main__":
    main()
