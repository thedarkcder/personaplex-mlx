"""Keep the system prompt (voice + persona) inside the attention window.

Upstream NVIDIA/personaplex pins the system prompt's KV prefix for exactly this
reason (commit 16419a2, "Endless conversation support with system prompt KV
pinning and attention sinking"). Without it the prompt scrolls out of the
window, so long conversations drift in voice and personality, and every fact the
application injected early is lost with it.

Two things make pinning correct in this MLX port:

* ``RotatingKVCache.max_size`` must be the *attention* window
  (``cfg.context``), not the RoPE limit (``cfg.max_seq_len``). The ring has to
  wrap exactly where attention stops looking. With ``max_size=4096`` and
  ``context=3000`` the ring wraps 1096 frames *after* attention has already
  thrown the prefix away, so the array handed to attention is a mix of current
  frames and stale ones from the previous cycle. Sizing the ring to the window
  removes that whole class of bug.
* ``keep`` marks the first N frames as never overwritten: writes start at
  ``keep`` and ``_idx`` wraps back to it, so the prompt survives while the
  conversation slides behind it.

RoPE needs no extra work: cached keys were rotated with their absolute offset
and are never re-encoded, so pinned frames keep their original positions while
the ring advances. Attention here runs one frame at a time with no mask, so the
order of keys inside the window is irrelevant - only which keys are present
matters.
"""
from __future__ import annotations

import mlx.core as mx
from moshi_mlx.modules import transformer as mlx_transformer
from moshi_mlx.modules.kv_cache import RotatingKVCache

_installed = False
# Copy of the prompt prefix, so a live session can prove whether it survived
# (``stats(verify=True)``). Taken on every conditioning, pinned or not.
_snapshot: list[mx.array] | None = None
_snapshot_frames = 0


def _make_windowed_rot_cache(self):
    """Replacement for ``Transformer.make_rot_cache``.

    Identical to the original except ``max_size`` is the attention window, so
    the ring cannot hold frames attention will never see.
    """
    num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat
    return [
        mlx_transformer.LayerCache(
            RotatingKVCache(
                head_dim=self.cfg.head_dim,
                n_kv_heads=num_kv_heads,
                max_size=self.cfg.context,
            )
        )
        for _ in self.layers
    ]


def install() -> None:
    """Route the main transformer's cache through the windowed layout.

    Must run before the model is built (``Lm(...)``), because caches are
    allocated there. Idempotent.
    """
    global _installed
    if _installed:
        return
    mlx_transformer.Transformer.make_rot_cache = _make_windowed_rot_cache
    _installed = True


def window_frames(caches) -> int:
    """Attention window in frames, as seen by the built caches."""
    return int(caches[0].self_attn.max_size)


def prompt_frames(caches) -> int:
    """Frames written by conditioning so far (the prefix to pin)."""
    return int(caches[0].self_attn.offset)


def pin_prompt(caches, frames: int | None = None) -> int:
    """Never overwrite the first ``frames`` KV entries of every layer.

    Called once, right after voice/persona conditioning, while the cache is
    still exactly the prompt. Returns the number of pinned frames.
    """
    if not caches:
        return 0
    if frames is None:
        frames = prompt_frames(caches)
    window = window_frames(caches)
    if frames <= 0:
        return 0
    if frames >= window:
        raise ValueError(
            f"system prompt is {frames} frames but the attention window is only "
            f"{window}; pinned prompt would leave no room for conversation "
            f"(raise PERSONAPLEX_LM_CONTEXT)"
        )
    for layer in caches:
        kv = layer.self_attn
        # Everything written so far must be the prompt: pinning after the
        # conversation has started would pin part of it as well.
        if int(kv.offset) != frames:
            raise RuntimeError(
                f"cache offset {int(kv.offset)} != pin point {frames}; "
                f"pin_prompt must run before the session starts"
            )
        kv.keep = frames
    snapshot(caches, frames)
    return frames


def snapshot(caches, frames: int | None = None) -> int:
    """Remember the prompt prefix so its survival can be checked later.

    Taken whether or not pinning is enabled, which is what makes the A/B
    measurement possible: with ``PERSONAPLEX_PIN_PROMPT=0`` the same check
    reports the prompt as gone once the conversation passes the window.
    """
    global _snapshot, _snapshot_frames
    if not caches:
        return 0
    if frames is None:
        frames = prompt_frames(caches)
    if frames <= 0:
        return 0
    for layer in caches:
        keys = layer.self_attn.keys
        if keys is None or keys.shape[2] < frames:
            raise RuntimeError("cache holds fewer frames than the prompt")
    _snapshot = [mx.array(layer.self_attn.keys[..., :frames, :]) for layer in caches]
    _snapshot_frames = frames
    return frames


def clear() -> None:
    """Forget the prompt snapshot (tests, or a fresh model instance)."""
    global _snapshot, _snapshot_frames
    _snapshot = None
    _snapshot_frames = 0


def prompt_intact(caches) -> bool | None:
    """Is the prompt prefix in the live cache bit-for-bit what it was?

    ``None`` when no snapshot was taken, otherwise a real comparison. Pinned
    entries are never rewritten, so exact equality is the correct test - and it
    catches the failures that matter (a cache reset, a restore that rebuilds
    the ring, a re-sized window), which behavioural probing only sees as vague
    drift.
    """
    if _snapshot is None:
        return None
    if len(caches) != len(_snapshot):
        return False
    for layer, saved in zip(caches, _snapshot):
        keys = layer.self_attn.keys
        if keys is None or keys.shape[2] < _snapshot_frames:
            return False
        current = keys[..., :_snapshot_frames, :]
        if not bool(mx.all(mx.equal(current, saved)).item()):
            return False
    return True


def stats(caches, verify: bool = False) -> dict:
    """Geometry of the live cache, plus the integrity check when asked.

    ``verify`` forces evaluation of the prompt slice, which runs GPU work and
    can stall a realtime loop, so it is opt-in (``/health?verify=1``).
    """
    if not caches:
        return {}
    window = window_frames(caches)
    pinned = int(caches[0].self_attn.keep)
    out = {
        "prompt_frames": _snapshot_frames,
        "pinned_frames": pinned,
        "window_frames": window,
        "sliding_frames": window - pinned,
        "cache_offset": int(caches[0].self_attn.offset),
    }
    if verify:
        out["prompt_intact"] = prompt_intact(caches)
    return out
