"""Falsification test for ledger I1372 / lever L31.

Hypothesis: in a multi-turn OpenAI chat, the server returns ``reasoning_content``
after ``.strip()`` (mlx_vlm/server/responses_state.py, ``_split_thinking`` /
``_clean_reasoning``), and on the next turn the chat template re-renders the
assistant turn as ``'<think>' + reasoning_content + '</think>' + content``
(chat_template.jinja:143-153, ``clear_thinking`` defaults false).  The prior
turn's *session token stream* -- the thing the vault / exact-APC prefix match
actually keys against -- was ``'<think>'`` (already in the turn-1 prompt) +
the RAW generated token ids (whatever whitespace the model actually emitted)
+ ``'</think>'`` + content.  If ``.strip()`` removes bytes the model emitted,
the re-rendered turn-2 prompt cannot literally be a token-prefix extension of
the turn-1 session stream, and the vault's exact-prefix reuse of the prior
turn's thinking prefill is defeated.

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
    # Included to show the divergence is not solely about the leading '\n'.
    "no_leading_newline": "Step one...\n" + THINK_CLOSE + "Final answer.",
    # Thinking-budget cutoff path: the budget critera force-closes thinking
    # with little/no real reasoning text, just the trailing newline the
    # budget-stop path is documented to produce before splicing '</think>'.
    "budget_stop_variant": "\n" + THINK_CLOSE + "Final answer.",
}


@pytest.mark.parametrize("variant_name", sorted(RAW_GENERATION_VARIANTS))
def test_stripped_reasoning_breaks_session_prefix_reuse(
    tokenizer, chat_template, variant_name, capsys
):
    """Falsification target: turn-2 prompt ids should equal turn-1 session ids
    as a prefix (session ids = turn-1 prompt ids + raw generated ids). This
    test documents that with the server's real (.strip()-ing) split, they do
    NOT -- divergence starts inside the thinking span, right where stripping
    ate bytes the model actually emitted. This is the EXPECTED (bug-confirming)
    outcome; the assertions pin the exact position so a future fix invalidates
    the pinned numbers loudly instead of silently.
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
    print(f"[{variant_name}] reasoning_content (stripped)={reasoning_content!r}")
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

    # The hypothesis under test: divergence happens strictly BEFORE the end
    # of the turn-1 session stream (i.e. the vault's exact-prefix match must
    # fail to reuse the full thinking span), and it happens after -- not
    # before -- the '<think>' token, i.e. inside the thinking span itself.
    assert divergence < len(session_ids_turn1), (
        f"[{variant_name}] EXPECTED-BUG-ABSENT: turn-2 ids were a full prefix "
        "extension of turn-1 session ids -- the stripped round-trip did NOT "
        "diverge. If this assertion fails, either the fix has already landed "
        "or this raw-generation variant no longer exercises the bug; re-check "
        "against the hypothesis in the module docstring before treating this "
        "as green."
    )
    assert divergence > think_open_pos, (
        f"[{variant_name}] divergence at index {divergence} is at/before the "
        f"'<think>' token position {think_open_pos}; expected the prompts to "
        "agree at least through the opening thinking marker."
    )


def test_lossless_reasoning_preserves_session_prefix(tokenizer, chat_template):
    """Control case: what the minimal fix would have to preserve.

    If the raw generated span (leading/trailing whitespace included) is kept
    verbatim as reasoning_content/content instead of `.strip()`-ed, the
    turn-2 prompt IS a full prefix extension of the turn-1 session ids -- the
    vault's exact-match reuse would work.  This is not what the server does
    today (see the parametrized test above); it demonstrates the invariant a
    fix must restore.
    """
    raw_generated = RAW_GENERATION_VARIANTS["with_leading_newline"]

    turn1_messages = [{"role": "user", "content": "Hello, what is 2+2?"}]
    turn1_text, turn1_prompt_ids = _render(tokenizer, chat_template, turn1_messages)
    assert turn1_text.endswith(THINK_OPEN)

    generated_ids = tokenizer.encode(raw_generated, add_special_tokens=False)
    session_ids_turn1 = turn1_prompt_ids + generated_ids

    # Lossless split: no stripping, keep every byte the model emitted on
    # either side of the literal '</think>' marker.
    reasoning_lossless, content_lossless = raw_generated.split(THINK_CLOSE, 1)

    turn2_messages = turn1_messages + [
        {
            "role": "assistant",
            "content": content_lossless,
            "reasoning_content": reasoning_lossless,
        },
        {"role": "user", "content": "And 3+3?"},
    ]
    turn2_text, turn2_ids = _render(tokenizer, chat_template, turn2_messages)

    divergence = _longest_common_prefix_len(session_ids_turn1, turn2_ids)
    print(
        f"\n[lossless] reasoning={reasoning_lossless!r} content={content_lossless!r} "
        f"len(session_ids_turn1)={len(session_ids_turn1)} divergence_index={divergence}"
    )

    assert divergence >= len(session_ids_turn1), (
        "lossless (unstripped) reasoning_content should make turn-2 ids a "
        f"full prefix extension of the turn-1 session ids; got divergence "
        f"at {divergence} of {len(session_ids_turn1)}"
    )


def test_jinja_scoping_does_not_leak_reasoning_across_turns(tokenizer, chat_template):
    """chat_template.jinja:143-149 sets `reasoning_content` inside the
    per-message `{% for %}` body via plain `{%- set -%}` (no namespace).
    Jinja2 clears loop-body `set` assignments at the end of each iteration
    (this is documented Jinja2 behaviour, not something this fork added), so
    an assistant turn with no reasoning_content of its own must render
    `<think></think>` and must NOT inherit an earlier turn's reasoning text.
    This is a sanity check that the round-trip bug above is a *stripping*
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
