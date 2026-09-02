# Session-tier restore: design note

Companion to the capture side (`context_vault.record_session_turn`,
`BatchGenerator.capture_session`). Capture without this is eviction pressure, so
the two go live together. Everything below is behind
`MLX_VLM_GLM5_VAULT_SESSION`, default OFF.

## Where it hooks

`BatchGenerator._vault_pick_for` (`mlx_vlm/generate/ar.py:2553`). That method
already does the whole job for the PREFILL tier: look up, check the hit is worth
taking, build a fresh prompt cache, restore into it, release the APC blocks the
plan no longer references, and return a `pick` dict with `source: "vault"`.

The session tier is a **second lookup competing under the same rule**, not a new
code path:

```
have    = pick["prefix_len"] if pick else 0          # APC's hit
p_hit   = vault.lookup(ids)                          # tier=PREFILL (default)
s_hit   = vault.lookup(ids, tier=VaultTier.SESSION)
```

Take the deepest of the three. On a session win, restore with
`vault.restore_into(fresh, s_hit, tier=VaultTier.SESSION)` — passing the tier is
mandatory, because `restore_into` refuses a tier mismatch by design, and a
PREFILL-tier call with a session rung returns False and falls through to a cold
prefill. Return `source: "vault-session"` so the two are separable in logs.

`_vault_prefix_trim_is_safe()` gates this tier exactly as it gates the other: a
model exposing `get_rope_index` remaps positions and a trimmed prefix would be
silently wrong rather than loud.

## The expected case is already the existing condition

The PREFILL guard is `have < hit.prefix_len < len(ids_list)`. The right-hand
strict inequality *is* "the stored prefix is a strict prefix of the new prompt",
which is the normal returning-turn shape: the stored rung covers
prompt+response of turn N, the new prompt is that plus the turn N+1 user
message, and only the new message is prefilled. No new condition is needed.

Two consequences worth stating rather than discovering:

* **Exact equality is rejected.** `hit.prefix_len == len(ids_list)` fails the
  strict `<` because the generate loop needs at least one column to process. A
  replayed identical prompt therefore re-prefills its last token. Correct, and
  cheap, but it is not zero.
* **The last sampled token is not in the rung.** Capture stores `prefix_len` as
  the cache's own offset, and the final sampled token was never fed back
  (`prefix_len_from_cache`). So a returning turn prefills that one token plus the
  new message. This is why the two sides must agree on the offset convention:
  the key is the full token list, the depth is the cache offset.

## Divergence needs no fallback code

The radix walk only descends an edge it fully matches
(`context_vault._walk` → `_common_len(edge, ...) != len(edge)` breaks). A prompt
that diverges from the stored transcript — an edited earlier turn, a different
system prompt, a branch — cannot return a node, so `lookup` yields `None`, the
pick is unchanged, and the request takes a full prefill. **The store is not
touched**: no eviction, no invalidation, no write on the miss path. The
diverged branch simply gets its own rung when *its* turn completes, and the two
share the trie prefix they actually have in common.

## What stays out

* No merging of a session rung with APC blocks. On a session win the APC blocks
  are released, same as the prefill tier — the plan that replaces them will
  never reference them.
* No change to `_vault_rungs_for`. Prefill rungs are still laid down on the way
  past, at the same geometric ladder, so a session miss still degrades to the
  8–12x prefix ladder rather than to cold.
* No TP. The server disables the vault wholesale under TP because the request
  path carries no rungs to rank 1 (`server/generation.py:1286-1290`); the
  session tier inherits that.

## Live validation

Ordered so each step gates the next:

1. a returning turn hits the session tier and reports `source: "vault-session"`;
2. the tokens actually prefilled equal (new user message + 1), not the whole
   transcript;
3. an edited earlier turn misses, takes a full prefill, and leaves
   `resident_bytes` unchanged.
