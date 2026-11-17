"""Tests for KV prompt pinning (kv_pinning.py).

No model needed: these exercise the cache geometry directly, which is where the
bug lives.

    python -m pytest tools/test_kv_pinning.py -q
    # or, without pytest:
    python tools/test_kv_pinning.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kv_pinning  # noqa: E402
from moshi_mlx.modules import transformer as mlx_transformer  # noqa: E402
from moshi_mlx.modules.kv_cache import RotatingKVCache  # noqa: E402

HEAD_DIM = 4


def make_cache(max_size, keep=0):
    return RotatingKVCache(
        head_dim=HEAD_DIM, n_kv_heads=1, max_size=max_size, keep=keep, step=4
    )


def step(cache, frame_id):
    """Feed one frame whose every component encodes its frame id."""
    k = mx.full((1, 1, 1, HEAD_DIM), float(frame_id), dtype=mx.float32)
    v = mx.full((1, 1, 1, HEAD_DIM), float(frame_id), dtype=mx.float32)
    return cache.update_and_fetch(k, v)


def ids(k):
    return [int(x) for x in k.reshape(-1, HEAD_DIM)[:, 0].tolist()]


def test_install_makes_the_ring_the_attention_window():
    kv_pinning.install()
    stub = SimpleNamespace(
        cfg=SimpleNamespace(num_heads=2, kv_repeat=1, head_dim=HEAD_DIM, context=64),
        layers=[None],
    )
    caches = mlx_transformer.Transformer.make_rot_cache(stub)
    assert caches[0].self_attn.max_size == 64, "ring must not outgrow attention"


def test_pinned_prefix_survives_wraparound():
    cache = make_cache(max_size=8)
    for i in range(3):  # the system prompt
        step(cache, i)
    kv_pinning.pin_prompt([mlx_transformer.LayerCache(cache)], frames=3)

    for i in range(3, 60):  # a long conversation
        k, _ = step(cache, i)

    got = ids(k)
    assert got[:3] == [0, 1, 2], f"prompt was overwritten: {got}"
    assert len(got) == 8, f"window grew past its size: {got}"


def test_window_holds_only_recent_frames():
    cache = make_cache(max_size=8)
    for i in range(3):
        step(cache, i)
    kv_pinning.pin_prompt([mlx_transformer.LayerCache(cache)], frames=3)

    for i in range(3, 60):
        k, _ = step(cache, i)

    tail = ids(k)[3:]
    # 5 sliding slots, all from the current cycle: no stale duplicates.
    assert sorted(tail) == [55, 56, 57, 58, 59], f"stale frames in window: {tail}"
    assert len(set(tail)) == len(tail), f"duplicate frames in window: {tail}"


def test_without_pinning_the_prompt_is_lost():
    """The bug this module exists to fix."""
    cache = make_cache(max_size=8, keep=0)
    for i in range(3):
        step(cache, i)
    for i in range(3, 60):
        k, _ = step(cache, i)
    assert 0 not in ids(k), "expected the prompt to be evicted when not pinned"


def test_pin_refuses_to_pin_past_the_window():
    cache = make_cache(max_size=8)
    for i in range(8):
        step(cache, i)
    try:
        kv_pinning.pin_prompt([mlx_transformer.LayerCache(cache)], frames=8)
    except ValueError as exc:
        assert "no room for conversation" in str(exc)
    else:
        raise AssertionError("expected ValueError for a full-window prompt")


def test_pin_refuses_to_run_mid_session():
    cache = make_cache(max_size=8)
    for i in range(4):
        step(cache, i)
    try:
        kv_pinning.pin_prompt([mlx_transformer.LayerCache(cache)], frames=3)
    except RuntimeError as exc:
        assert "must run before the session starts" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when offset != pin point")


def test_live_cache_can_prove_the_prompt_is_intact():
    cache = make_cache(max_size=8)
    for i in range(3):
        step(cache, i)
    kv_pinning.pin_prompt([mlx_transformer.LayerCache(cache)], frames=3)

    for i in range(3, 60):
        step(cache, i)
    assert kv_pinning.prompt_intact([mlx_transformer.LayerCache(cache)]) is True

    # The failure the check exists to catch: the cache got reset (new session
    # on a fresh ring) and the prompt is gone but keep would still say 3.
    cache.reset()
    assert kv_pinning.prompt_intact([mlx_transformer.LayerCache(cache)]) is False


def test_without_pinning_intact_is_unknown():
    kv_pinning.clear()
    cache = make_cache(max_size=8)
    for i in range(3):
        step(cache, i)
    assert kv_pinning.prompt_intact([mlx_transformer.LayerCache(cache)]) is None


def test_stats_reports_geometry_and_opt_in_verification():
    cache = make_cache(max_size=8)
    for i in range(3):
        step(cache, i)
    caches = [mlx_transformer.LayerCache(cache)]
    kv_pinning.pin_prompt(caches, frames=3)
    for i in range(3, 20):
        step(cache, i)

    plain = kv_pinning.stats(caches)
    assert plain["pinned_frames"] == 3
    assert plain["window_frames"] == 8
    assert plain["sliding_frames"] == 5
    assert plain["cache_offset"] == 20
    assert "prompt_intact" not in plain, "verification must be opt-in"
    assert kv_pinning.stats(caches, verify=True)["prompt_intact"] is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all pinning tests passed")
