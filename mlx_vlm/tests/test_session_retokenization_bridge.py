"""LW4 (2026-09-07): the strongest remaining hypothesis for why the session
tier's rung was unreachable even after the LW3 wiring fix (b8f37e1d) -- and
its fix.

Hypothesis: the session key is the model's own raw generated token ids
(generate/ar.py's ``note_generated`` accumulates exactly what was sampled,
one token at a time); a follow-up turn arrives as a RE-RENDERED conversation
(the fork's own lossless split/render, a7b3edbd + 82fc910c) that gets
tokenized FRESH. BPE is not injective at merge boundaries: two adjacent
generated tokens can decode to text that a single fresh ``encode`` call
merges into ONE different token. The ordinary token-trie walk
(``ContextVault.lookup``/``_walk``) then sees a real divergence in TOKEN
SPACE at that point, and the session rung becomes unreachable even though
the underlying TEXT is byte-identical.

Part 1 below proves the mismatch exists, with the real tokenizer, using the
literal ids/text example from this ledger entry's own diagnosis
(``encode("\\n") == [198]``, but ``encode(decode([198, 198])) == [271]``,
never ``[198, 198]`` back) plus a realistic in-context construction (two
separate generated tokens ``"."`` + ``"\\n"`` + ``"\\n"`` that a fresh encode
of the same text merges into ``"." + "\\n\\n"`` as ONE token). Part 2 proves
the fix (``context_vault.find_session_text_bridge`` + the splice it enables)
restores an exact, full-depth match.
"""

import os

import pytest

pytest.importorskip("transformers")
from transformers import AutoTokenizer  # noqa: E402

import mlx.core as mx  # noqa: E402

from mlx_vlm import context_vault as V  # noqa: E402
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402

MODEL_DIR = "/Users/gesicht/glm53flash/builds/GLM-5.3-Flash-vlm-q4-quasar"
H, D = 2, 8

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason=f"tokenizer fixture not present: {MODEL_DIR}",
)

_PREV = None


def setup_module():
    global _PREV
    _PREV = mx.default_device()
    mx.set_default_device(mx.cpu)


def teardown_module():
    if _PREV is not None:
        mx.set_default_device(_PREV)


def row_cache(n):
    """A cache shaped like GLM-5-Next's (KDA ArraysCache + DSA CacheList),
    holding ``n`` tokens -- same shape the other session-tier test modules
    use, so ``ContextVault.insert`` accepts a real, restorable fragment list
    rather than a hand-rolled placeholder."""
    c = [ArraysCache(size=2), CacheList(KVCache(), KVCache())]
    c[0][0] = mx.zeros((1, H, D, 4), mx.float32)
    c[0][1] = mx.zeros((1, H, D, D), mx.float32)
    lat = mx.zeros((1, H, n, D), mx.bfloat16)
    c[1].caches[0].update_and_fetch(lat, lat)
    idx = mx.zeros((1, 1, n, 2 * D + 1), mx.bfloat16)
    c[1].caches[1].update_and_fetch(idx, mx.zeros((1, 1, n, 0), mx.bfloat16))
    mx.eval([e.state for e in c])
    return c


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# --------------------------------------------------------------------- part 1


def test_two_generated_newline_tokens_reencode_as_one_different_token(tokenizer):
    """The minimal reproduction cited in the ledger: the model sampling '\\n'
    twice, one token at a time (id 198 each -- entirely ordinary
    autoregressive behaviour), decodes to the same text a single fresh
    encode() call tokenizes as ONE token (271), never back to [198, 198].
    """
    nl_id = tokenizer.encode("\n", add_special_tokens=False)
    assert nl_id == [198]
    raw_generated = nl_id + nl_id  # [198, 198] -- two separate sampling steps
    decoded = tokenizer.decode(raw_generated)
    assert decoded == "\n\n"
    reencoded = tokenizer.encode(decoded, add_special_tokens=False)
    assert reencoded == [271]
    assert reencoded != raw_generated, (
        "if this ever starts matching, the retokenisation-mismatch hypothesis "
        "no longer applies to this tokenizer/vocab and the bridge fix's "
        "justification should be re-examined"
    )


def test_midtext_merge_boundary_breaks_the_ordinary_trie_walk(tokenizer):
    """A realistic in-context case: the model emits '.', then '\\n', then
    '\\n' as three separate sampled tokens (plausible -- nothing requires a
    model to "know" in advance that a paragraph break is coming and emit a
    pre-merged token for it). A fresh encode() of the SAME decoded text
    merges '.' + '\\n\\n' into one token, so the raw generated ids and the
    query's own re-tokenisation of the identical text diverge WELL BEFORE
    the ids run out -- not at the very first generated token, and not
    because anything is missing, purely because of where a merge boundary
    happens to fall.
    """
    prefix_ids = tokenizer.encode("Step one", add_special_tokens=False)
    dot_id = tokenizer.encode(".", add_special_tokens=False)
    nl_id = tokenizer.encode("\n", add_special_tokens=False)
    suffix_ids = tokenizer.encode("Step two is done.", add_special_tokens=False)

    raw_generated = prefix_ids + dot_id + nl_id + nl_id + suffix_ids
    text = tokenizer.decode(raw_generated)
    assert text == "Step one.\n\nStep two is done."

    # What a FRESH encode of the identical text produces (this is what turn
    # 2's re-render naturally does -- there is no bug in the rendering or
    # the encode call itself, only in comparing its OUTPUT token-for-token
    # against the raw generated ids).
    fresh_reencode = tokenizer.encode(text, add_special_tokens=False)
    assert fresh_reencode != raw_generated

    divergence = _common_prefix_len(raw_generated, fresh_reencode)
    # The merge point is inside the '.' + '\n' + '\n' span, i.e. strictly
    # after "Step one" (8+ tokens in) and strictly before the full length --
    # demonstrating the mismatch is a MID-SEQUENCE phenomenon, not a
    # boundary-of-the-prompt artifact.
    assert 0 < divergence < len(raw_generated)
    print(
        f"\nraw_generated={raw_generated} fresh_reencode={fresh_reencode} "
        f"divergence_index={divergence} "
        f"(raw tok at divergence={raw_generated[divergence]!r} "
        f"decoded={tokenizer.decode([raw_generated[divergence]])!r}, "
        f"fresh tok at divergence={fresh_reencode[divergence]!r} "
        f"decoded={tokenizer.decode([fresh_reencode[divergence]])!r})"
    )


def test_session_lookup_misses_past_the_merge_point(tokenizer):
    """Constructive proof at the vault level (not just the tokenizer level):
    store a SESSION rung keyed by the raw (pre-merge) generated ids, then
    look it up with the query the way a re-rendered turn 2 would actually
    produce it (a fresh encode of the identical text) -- the ordinary
    ContextVault trie walk finds only the COMMON prefix up to the merge
    point, not the full stored depth, even though the underlying text was
    never altered.
    """
    prompt_ids = list(range(500))  # stand-in prompt, distinct from the vocab ids below
    prefix_ids = tokenizer.encode("Step one", add_special_tokens=False)
    dot_id = tokenizer.encode(".", add_special_tokens=False)
    nl_id = tokenizer.encode("\n", add_special_tokens=False)
    suffix_ids = tokenizer.encode("Step two is done.", add_special_tokens=False)
    raw_generated = prefix_ids + dot_id + nl_id + nl_id + suffix_ids
    stored_full = prompt_ids + raw_generated

    vault = V.ContextVault("retok", budget_bytes=1 << 30)
    frags = V.capture_fragments(row_cache(len(stored_full)), len(stored_full))
    inserted = vault.insert(
        stored_full, len(stored_full), frags, tier=V.VaultTier.SESSION,
        session_id="s1",
    )
    assert inserted

    text = tokenizer.decode(raw_generated)
    fresh_reencode = tokenizer.encode(text, add_special_tokens=False)
    query = prompt_ids + fresh_reencode + tokenizer.encode(
        "And the next turn.", add_special_tokens=False
    )

    hit = vault.lookup(query, tier=V.VaultTier.SESSION)
    diag = vault.diagnose_lookup(query, tier=V.VaultTier.SESSION)
    print(f"\nhit={hit!r} diag={diag}")

    assert diag["candidates"] == 1
    # The walk gets AT LEAST through the shared prompt (500 tokens); it must
    # NOT reach the full stored depth, because the tokens genuinely disagree
    # from the merge point onward.
    assert len(prompt_ids) <= diag["longest_common_prefix"] < len(stored_full)
    if hit is not None:
        assert hit.prefix_len < len(stored_full)


# --------------------------------------------------------------------- part 2


def test_find_session_text_bridge_matches_on_decoded_text(tokenizer):
    """The fix's detector: even though the raw ids disagree, the STORED
    text is a literal prefix of the QUERY's own decoded text, so
    find_session_text_bridge finds it -- comparing in text space, where
    BPE's non-injectivity does not apply.
    """
    prompt_ids = list(range(500))
    prefix_ids = tokenizer.encode("Step one", add_special_tokens=False)
    dot_id = tokenizer.encode(".", add_special_tokens=False)
    nl_id = tokenizer.encode("\n", add_special_tokens=False)
    suffix_ids = tokenizer.encode("Step two is done.", add_special_tokens=False)
    raw_generated = prefix_ids + dot_id + nl_id + nl_id + suffix_ids
    stored_full = prompt_ids + raw_generated

    vault = V.ContextVault("retok2", budget_bytes=1 << 30)
    frags = V.capture_fragments(row_cache(len(stored_full)), len(stored_full))
    assert vault.insert(
        stored_full, len(stored_full), frags, tier=V.VaultTier.SESSION,
        session_id="s1",
    )

    text = tokenizer.decode(raw_generated)
    fresh_reencode = tokenizer.encode(text, add_special_tokens=False)
    next_turn_ids = tokenizer.encode("And the next turn.", add_special_tokens=False)
    query = prompt_ids + fresh_reencode + next_turn_ids

    def decode(ids):
        return tokenizer.decode(list(ids))

    bridge = V.find_session_text_bridge(vault, query, decode=decode)
    assert bridge is not None
    stored_tokens, stored_text = bridge
    assert stored_tokens == stored_full
    assert stored_text == decode(stored_full)


def test_spliced_prompt_reaches_full_depth_and_preserves_text(tokenizer):
    """The end-to-end invariant the splice must uphold: after replacing the
    query's re-encoded span with the stored raw ids (and re-encoding only
    the genuinely NEW remainder), (a) an ordinary trie walk now finds the
    FULL stored depth -- no new matching logic needed downstream -- and
    (b) the spliced sequence decodes to the SAME text the un-spliced query
    would have (the splice changes tokenisation, never content).
    """
    prompt_ids = list(range(500))
    prefix_ids = tokenizer.encode("Step one", add_special_tokens=False)
    dot_id = tokenizer.encode(".", add_special_tokens=False)
    nl_id = tokenizer.encode("\n", add_special_tokens=False)
    suffix_ids = tokenizer.encode("Step two is done.", add_special_tokens=False)
    raw_generated = prefix_ids + dot_id + nl_id + nl_id + suffix_ids
    stored_full = prompt_ids + raw_generated

    vault = V.ContextVault("retok3", budget_bytes=1 << 30)
    frags = V.capture_fragments(row_cache(len(stored_full)), len(stored_full))
    assert vault.insert(
        stored_full, len(stored_full), frags, tier=V.VaultTier.SESSION,
        session_id="s1",
    )

    text = tokenizer.decode(raw_generated)
    fresh_reencode = tokenizer.encode(text, add_special_tokens=False)
    next_turn_text = "And the next turn."
    next_turn_ids = tokenizer.encode(next_turn_text, add_special_tokens=False)
    query = prompt_ids + fresh_reencode + next_turn_ids

    def decode(ids):
        return tokenizer.decode(list(ids))

    bridge = V.find_session_text_bridge(vault, query, decode=decode)
    assert bridge is not None
    stored_tokens, stored_text = bridge

    # The splice, exactly as server/generation.py's admission loop performs it.
    query_text = decode(query)
    remainder_text = query_text[len(stored_text):]
    remainder_ids = tokenizer.encode(remainder_text, add_special_tokens=False)
    spliced = list(stored_tokens) + list(remainder_ids)

    # (a) full-depth match now, via the plain, unmodified trie walk.
    diag_before = V.ContextVault.diagnose_lookup(vault, query, tier=V.VaultTier.SESSION)
    diag_after = V.ContextVault.diagnose_lookup(vault, spliced, tier=V.VaultTier.SESSION)
    print(f"\nbefore={diag_before}\nafter ={diag_after}")
    assert diag_before["longest_common_prefix"] < len(stored_full)
    assert diag_after["longest_common_prefix"] >= len(stored_full)
    hit = vault.lookup(spliced, tier=V.VaultTier.SESSION)
    assert hit is not None
    assert hit.prefix_len == len(stored_full)

    # (b) content-preserving: decoding the spliced sequence reproduces the
    # original query's text (the same tokenizer round-trip caveats apply as
    # to any encode/decode pair -- no worse than the status quo).
    assert decode(spliced) == query_text


def test_find_session_text_bridge_returns_none_without_a_vault():
    assert V.find_session_text_bridge(None, [1, 2, 3], decode=lambda ids: "") is None


def test_find_session_text_bridge_ignores_a_candidate_not_shorter_than_the_query():
    vault = V.ContextVault("retok4", budget_bytes=1 << 30)
    toks = [1, 2, 3, 4, 5]
    frags = V.capture_fragments(row_cache(len(toks)), len(toks))
    assert vault.insert(toks, len(toks), frags, tier=V.VaultTier.SESSION, session_id="s")

    bridge = V.find_session_text_bridge(
        vault, toks, decode=lambda ids: "same-length-or-shorter-query"
    )
    assert bridge is None
