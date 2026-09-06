"""Falsification test, then fix-verification test, for ledger I1372 / lever L31.

Original hypothesis (CONFIRMED, then fixed -- see commit history on this
branch): the server returned ``reasoning_content`` after ``.strip()``
(mlx_vlm/server/responses_state.py, ``_split_thinking`` / ``_clean_reasoning``,
plus the same ``.lstrip("\\n")`` pattern in ``ThinkingStreamState`` for the
streaming path), and on the next turn the chat template re-rendered the
assistant turn as ``'<think>' + reasoning_content + '</think>' + content``
(chat_template.jinja:143-153, ``clear_thinking`` defaults false).  The prior
turn's *session token stream* -- the thing the vault / exact-APC prefix match
actually keys against -- was ``'<think>'`` (already in the turn-1 prompt) +
the RAW generated token ids (whatever whitespace the model actually emitted)
+ ``'</think>'`` + content.  Stripping removed bytes the model emitted, so the
re-rendered turn-2 prompt was NOT a token-prefix extension of the turn-1
session stream, and the vault's exact-prefix reuse of the prior turn's
thinking prefill was defeated.

Fix applied (this branch): removed every ``.strip()`` / ``.lstrip("\\n")`` in
the marker-based and response-template-parser paths of ``_split_thinking``,
``_clean_reasoning``, and the streaming ``ThinkingStreamState`` (`feed`,
`_strip_open_marker`) -- see responses_state.py. Marker text itself
(`<think>`/`</think>`/`<|channel>thought`/`<channel|>`/etc.) is still removed;
only the surrounding whitespace bytes are now preserved verbatim.

Where the session token stream really comes from (cited, not re-derived here):

  * ``mlx_vlm/generate/ar.py:3984-3988`` (``BatchGenerator.insert``) seeds
    ``self._session_tokens[uid]`` with ``list(p)`` -- the *exact* prompt ids
    the model is fed, turn-1 ending in the ``<think>`` token (id 154841).
  * ``mlx_vlm/generate/ar.py:3997-4016`` (``note_generated``) appends emitted
    token ids to that same accumulator, one call per emitted token.
  * ``mlx_vlm/server/generation.py:2926-2936`` is the only caller of
    ``note_generated``; it passes ``[int(tok)]`` where ``tok = r.token`` --
    the raw sampled id for that step, straight from the decode loop.  The
    server does NOT re-tokenize decoded text to build the session key; it
    keeps the ids the model actually produced (including ``</think>``'s own
    token id, 154842, and whatever whitespace tokens preceded it).

This test module has no model loaded (CPU-only, tokenizer/template only), so
it cannot reproduce ``note_generated``'s literal per-step id stream.  It
approximates "the ids the model would have produced" by tokenizing a
plausible raw generation string with the *same* tokenizer the model uses.
That is an approximation of sampling, not a replay of it -- flagged here so
nobody mistakes this for an end-to-end capture.  It is sufficient to
demonstrate the claim under test, which is about text-level stripping vs.
token-level identity, not about sampling fidelity.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("transformers")
from transformers import AutoTokenizer  # noqa: E402

from mlx_vlm.server.responses_state import _split_thinking  # noqa: E402

MODEL_DIR = "/Users/gesicht/glm53flash/builds/GLM-5.3-Flash-vlm-q4-quasar"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_OPEN_ID = 154841
THINK_CLOSE_ID = 154842

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason=f"tokenizer/template fixture not present: {MODEL_DIR}",
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def chat_template():
    with open(os.path.join(MODEL_DIR, "chat_template.jinja")) as fh:
        return fh.read()


def _render(tokenizer, chat_template, messages, add_generation_prompt=True):
    """Render messages to text with the fork's own template, then tokenize.

    ``tokenizer.apply_chat_template(..., tokenize=True)`` returns a
    ``tokenizers.Encoding`` wrapper on this tokenizer's transformers version
    that is not indexable as plain ids (verified: raises ``TypeError:
    argument 'ids': 'tokenizers.Encoding' object cannot be interpreted as an
    integer`` from inside ``_decode``). Render to text and re-encode with
    ``add_special_tokens=False`` instead -- the template text already carries
    every special token as literal text (``<|user|>``, ``<think>``, ...).
    """
    text = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids


def _longest_common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _ctx(tokenizer, ids, around, radius=5):
    lo, hi = max(0, around - radius), around + radius
    window = ids[lo:hi]
    return repr("".join(tokenizer.decode([t]) for t in window))[:200]


RAW_GENERATION_VARIANTS = {
    # Plausible reasoning stream: model's first emitted token after the
    # prompt's trailing '<think>' is a newline, as chat-tuned models commonly
    # do before writing the first reasoning line.
    "with_leading_newline": "\nStep one...\n" + THINK_CLOSE + "Final answer.",
    # Same content, but the model happens not to emit a leading newline.
    # Included to show the (former) divergence was not solely about the
    # leading '\n'.
    "no_leading_newline": "Step one...\n" + THINK_CLOSE + "Final answer.",
    # Thinking-budget cutoff path: the budget criteria force-close thinking
    # with little/no real reasoning text, just the trailing newline the
    # budget-stop path is documented to produce before splicing '</think>'.
    "budget_stop_variant": "\n" + THINK_CLOSE + "Final answer.",
}


@pytest.mark.parametrize("variant_name", sorted(RAW_GENERATION_VARIANTS))
def test_split_thinking_is_lossless_and_preserves_session_prefix(
    tokenizer, chat_template, variant_name, capsys
):
    """Fix-verification: turn-2 prompt ids must equal turn-1 session ids as a
    full prefix (session ids = turn-1 prompt ids + raw generated ids), for
    every raw-generation shape that used to break this (I1372/L31). Before
    the fix, this parametrized test was named
    ``test_stripped_reasoning_breaks_session_prefix_reuse`` and asserted the
    OPPOSITE (divergence strictly inside the thinking span) -- see git log
    for the pre-fix version and ``test_old_stripped_behavior_...`` below for
    a standing regression guard against reintroducing that bug.
    """
    raw_generated = RAW_GENERATION_VARIANTS[variant_name]

    turn1_messages = [{"role": "user", "content": "Hello, what is 2+2?"}]
    turn1_text, turn1_prompt_ids = _render(tokenizer, chat_template, turn1_messages)
    assert turn1_text.endswith(THINK_OPEN), (
        "turn-1 prompt with add_generation_prompt must end with '<think>'; "
        f"got tail {turn1_text[-40:]!r}"
    )
    think_open_pos = len(turn1_prompt_ids) - 1
    assert turn1_prompt_ids[think_open_pos] == THINK_OPEN_ID

    # Approximation of the raw per-token id stream `note_generated` would have
    # accumulated (see module docstring for the caveat + ar.py citations).
    generated_ids = tokenizer.encode(raw_generated, add_special_tokens=False)
    session_ids_turn1 = turn1_prompt_ids + generated_ids

    # (c) the server's actual split, exactly as a client would receive it.
    # processor=None forces the marker-based `_split_thinking` branch; this
    # tokenizer's `response_template` is None (verified separately), so
    # passing the real tokenizer as `processor` would hit the same branch --
    # marker-based splitting IS the production path for this model.
    reasoning_content, content = _split_thinking(
        raw_generated,
        thinking_start_token=THINK_OPEN,
        thinking_end_token=THINK_CLOSE,
        starts_in_thinking=True,
        processor=None,
    )

    turn2_messages = turn1_messages + [
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
        },
        {"role": "user", "content": "And 3+3?"},
    ]
    turn2_text, turn2_ids = _render(tokenizer, chat_template, turn2_messages)

    divergence = _longest_common_prefix_len(session_ids_turn1, turn2_ids)

    print(f"\n[{variant_name}] raw_generated={raw_generated!r}")
    print(f"[{variant_name}] reasoning_content (lossless)={reasoning_content!r}")
    print(f"[{variant_name}] content={content!r}")
    print(
        f"[{variant_name}] len(session_ids_turn1)={len(session_ids_turn1)} "
        f"len(turn2_ids)={len(turn2_ids)} divergence_index={divergence} "
        f"think_open_pos={think_open_pos}"
    )
    print(
        f"[{variant_name}] session ctx @div: {_ctx(tokenizer, session_ids_turn1, divergence)}"
    )
    print(f"[{variant_name}] turn2   ctx @div: {_ctx(tokenizer, turn2_ids, divergence)}")

    # Fix invariant: the whole turn-1 session stream (prompt + raw generated
    # ids) must be a token-prefix of the turn-2 re-render, so the vault's
    # exact-prefix match can reuse the full thinking prefill.
    assert divergence >= len(session_ids_turn1), (
        f"[{variant_name}] REGRESSION: turn-2 ids are NOT a full prefix "
        "extension of turn-1 session ids -- the lossless split has regressed "
        f"back toward the pre-fix (stripping) behaviour. Divergence at "
        f"{divergence} of {len(session_ids_turn1)}; session ctx @div: "
        f"{_ctx(tokenizer, session_ids_turn1, divergence)}"
    )


def test_old_stripped_behavior_would_have_broken_prefix_reuse_regression_guard(
    tokenizer, chat_template
):
    """Regression guard documenting the PRE-FIX behaviour this branch removed.

    This does not call `_split_thinking` (that function is fixed now); it
    reimplements, inline, the exact `.strip()`-based split that
    `_clean_reasoning` / `_split_thinking` used to do, so that:

      (a) the historical bug is documented in a form future readers can run,
          rather than only in a commit message, and
      (b) if someone re-adds `.strip()` to the production split "to tidy up
          whitespace", this test's neighbour above
          (`test_split_thinking_is_lossless_and_preserves_session_prefix`)
          will fail loudly -- this test exists to explain *why* it fails
          when that happens.

    If this test's own assertions ever fail, it means the historical
    strip-based transform stopped breaking the prefix invariant for this
    fixture, which would be surprising and worth investigating on its own
    (e.g. a tokenizer/template change) rather than assumed benign.
    """
    raw_generated = RAW_GENERATION_VARIANTS["with_leading_newline"]

    turn1_messages = [{"role": "user", "content": "Hello, what is 2+2?"}]
    turn1_text, turn1_prompt_ids = _render(tokenizer, chat_template, turn1_messages)
    assert turn1_text.endswith(THINK_OPEN)
    think_open_pos = len(turn1_prompt_ids) - 1

    generated_ids = tokenizer.encode(raw_generated, add_special_tokens=False)
    session_ids_turn1 = turn1_prompt_ids + generated_ids

    # The old (pre-fix) `_split_thinking` marker branch, reproduced verbatim
    # for documentation purposes only -- see git history of responses_state.py
    # on this branch for the real removed code:
    #
    #   reasoning, content = text.split(end_marker, 1)
    #   reasoning = reasoning.replace(start_marker, "").strip()   # <- the bug
    #   content = content.strip()                                 # <- the bug
    old_reasoning, old_content = raw_generated.split(THINK_CLOSE, 1)
    old_reasoning = old_reasoning.replace(THINK_OPEN, "").strip()
    old_content = old_content.strip()

    turn2_messages = turn1_messages + [
        {
            "role": "assistant",
            "content": old_content,
            "reasoning_content": old_reasoning,
        },
        {"role": "user", "content": "And 3+3?"},
    ]
    turn2_text, turn2_ids = _render(tokenizer, chat_template, turn2_messages)

    divergence = _longest_common_prefix_len(session_ids_turn1, turn2_ids)
    print(
        f"\n[old-stripped] reasoning={old_reasoning!r} content={old_content!r} "
        f"len(session_ids_turn1)={len(session_ids_turn1)} divergence_index={divergence} "
        f"think_open_pos={think_open_pos}"
    )

    assert divergence < len(session_ids_turn1), (
        "the historical .strip()-based split no longer breaks the session "
        "prefix for this fixture -- investigate before trusting this guard"
    )
    assert divergence > think_open_pos


def test_jinja_scoping_does_not_leak_reasoning_across_turns(tokenizer, chat_template):
    """chat_template.jinja:143-149 sets `reasoning_content` inside the
    per-message `{% for %}` body via plain `{%- set -%}` (no namespace).
    Jinja2 clears loop-body `set` assignments at the end of each iteration
    (this is documented Jinja2 behaviour, not something this fork added), so
    an assistant turn with no reasoning_content of its own must render
    `<think></think>` and must NOT inherit an earlier turn's reasoning text.
    This is a sanity check that the round-trip bug above was a *stripping*
    problem, not a *scoping leak* -- both were flagged as things to verify.
    """
    # Must not itself contain the substrings "A1"/"A2" used as turn markers
    # below, or the naive text-split probing this test does would misfire.
    leaked_marker = "REASONING-MUST-NOT-LEAK-BETWEEN-TURNS"
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_content": leaked_marker},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},  # no reasoning_content key at all
        {"role": "user", "content": "Q3"},
    ]
    text, _ids = _render(tokenizer, chat_template, messages)

    # The A2 turn's own rendering (between the two assistant markers after A1)
    # must be exactly '<think></think>A2', not carrying A1's reasoning.
    after_a1 = text.split("A1", 1)[1]
    a2_region = after_a1.split("A2", 1)[0]
    assert a2_region.endswith(THINK_OPEN + THINK_CLOSE), (
        f"A2's turn should open with an empty thinking block; got {a2_region!r}"
    )
    assert leaked_marker not in after_a1.split("A2", 1)[0], (
        "A1's reasoning_content leaked into A2's rendered thinking block -- "
        "Jinja for-loop scoping regression"
    )
    assert leaked_marker not in text.split("A2", 1)[1], (
        "A1's reasoning_content leaked past A2 into later turns"
    )
