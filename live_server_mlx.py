#!/usr/bin/env python3
"""
PersonaPlex MLX - Live Streaming WebSocket Server with Voice Conditioning

Real-time full-duplex voice conversation on Apple Silicon using MLX.
Includes voice conditioning support for proper response generation.
"""

import asyncio
import collections
import json
import os
import queue
import threading
import time
import copy
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
import rustymimi
import torch  # For loading .pt files
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import kv_pinning

# Must happen before the model is built: it changes how the transformer's KV
# caches are sized (window = cfg.context, not max_seq_len) so the system prompt
# can be pinned in place. See kv_pinning.py.
kv_pinning.install()

# Constants
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per frame

SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)
ZERO_TEXT_CODE = 3
# Number of depformer slices whose audio we actually decode (see trim_depformer).
USED_AUDIO_CODEBOOKS = 8
# rustymimi's streaming state overflows after exactly 4096 encoded frames (it
# writes 2 slots per frame into an 8192-entry buffer) and .reset() does not
# clear it - the tokenizer is then permanently broken. Swap in a prebuilt one
# before that point.
CODEC_FRAME_LIMIT = int(os.environ.get("PERSONAPLEX_CODEC_LIMIT", "3800"))
# Set PERSONAPLEX_LOG_TEXT=1 to print what the model says, token by token. Very
# useful for spotting a repetition loop without having to listen.
LOG_TEXT = os.environ.get("PERSONAPLEX_LOG_TEXT") == "1"

# Sampling. Text temperature is the main lever on how faithfully the model
# quotes facts (names, prices, dates) from its persona prompt: at 0.7 it likes
# to paraphrase and will happily shift "$110 for 8 hours" into "$110 for 4
# hours". 0.4-0.5 makes it read them back nearly verbatim at the cost of some
# liveliness. Audio temperature controls prosody only.
TEXT_TEMP = float(os.environ.get("PERSONAPLEX_TEXT_TEMP", "0.7"))
TEXT_TOP_K = int(os.environ.get("PERSONAPLEX_TEXT_TOP_K", "25"))
AUDIO_TEMP = float(os.environ.get("PERSONAPLEX_AUDIO_TEMP", "0.8"))
AUDIO_TOP_K = int(os.environ.get("PERSONAPLEX_AUDIO_TOP_K", "250"))

# What gets quantized. "all" is the default (transformer, depformer, heads).
# "transformer" leaves the depformer and the output/embedding heads at full
# precision: measured at ~29 ms/frame vs ~26 ms, and those layers are exactly
# where text token choice lives, so it is a cheap quality win.
QUANT_SCOPE = os.environ.get("PERSONAPLEX_QUANT_SCOPE", "all")

# --- Live session protocol -------------------------------------------------
# Stream every generated text token to the client as generation.text_delta.
EMIT_TEXT_DELTAS = os.environ.get("PERSONAPLEX_TEXT_DELTAS", "1") == "1"
# A context.inject update is only applied when the conversation is quiet, so the
# cache is never mutated in the middle of a turn. Both streams must have been
# below these levels for INJECT_QUIET_FRAMES frames (12.5 fps => 5 frames = 400ms).
INJECT_QUIET_FRAMES = int(os.environ.get("PERSONAPLEX_INJECT_QUIET", "5"))
INJECT_OUT_RMS = 0.010   # model speaking vs silence
INJECT_MIC_RMS = 0.010   # user speaking vs silence
# Each injected token costs one LM step (~30ms) and produces no audio, so keep
# updates short. 64 tokens ~= 5.1s of context for ~2s of wall clock: two or
# three short plain-text facts fit.
MAX_INJECT_TOKENS = int(os.environ.get("PERSONAPLEX_MAX_INJECT_TOKENS", "64"))
# The model has no speaker/role channel mid-conversation: text it reads sits in
# the same stream position as its own speech, so an injected fact is naturally
# attributed to whoever was talking. A provenance phrase in the text does NOT
# fix that, and makes it worse: the model reads the phrase aloud ("7429, source
# system, not user"). Measured both ways - keep the injection a bare fact.
# The knob stays for experiments.
INJECT_PREFIX = os.environ.get("PERSONAPLEX_INJECT_PREFIX", "")
# An injection normally waits for a pause. If the model never stops talking
# (it can loop), waiting forever would silently swallow the update, so after
# this long it is applied anyway and the report says "forced": true.
INJECT_MAX_WAIT_S = float(os.environ.get("PERSONAPLEX_INJECT_MAX_WAIT", "4.0"))
# Right after an update the model can continue the injected text as if it were
# its own sentence and say it aloud. Requirement: an update is never spoken, so
# the frames generated in this window are discarded and silence is fed back.
INJECT_MUTE_FRAMES = int(os.environ.get("PERSONAPLEX_INJECT_MUTE_FRAMES", "12"))
# Facts are replayed one by one after a reset, so an unbounded set would make a
# reset take longer and longer. Keep the most recent facts up to this budget.
MAX_CONTEXT_TOKENS = int(os.environ.get("PERSONAPLEX_MAX_CONTEXT_TOKENS",
                                        str(4 * MAX_INJECT_TOKENS)))

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
VOICES_DIR = WEIGHTS_DIR / "voices_cpu"

# Global model state
MODEL = None
CONFIG = None
VOICE_CACHE_STATE = None  # Saved KV cache after voice prompt
# Remembered so a model reload (/set-quant) can put the caller back on the same
# voice and persona instead of silently resetting them to the default.
CURRENT_VOICE = "NATF0"
CURRENT_PROMPT = (
    "You are a friendly voice assistant on a call. Answer in one or two short "
    "sentences. If you did not catch what the user said, say so and ask them to "
    "repeat instead of guessing."
)

# Mimi codecs. rustymimi's streaming state is NOT fully cleared by .reset():
# after ~4096 frames of audio in total the encoder's internal buffer overflows
# and every encode_step raises "narrow invalid args start + len > dim_len"
# forever after. So each session gets a freshly built pair, and a spare pair is
# built in the background while a session runs (building one takes ~1.4s).
_codec_lock = threading.Lock()
_codec_spare = None


def _build_codecs():
    path = str(WEIGHTS_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors")
    return (
        rustymimi.Tokenizer(path, num_codebooks=USED_AUDIO_CODEBOOKS),
        rustymimi.Tokenizer(path, num_codebooks=USED_AUDIO_CODEBOOKS),
    )


def _prebuild_codecs():
    global _codec_spare
    try:
        pair = _build_codecs()
    except Exception as e:
        print(f"Codec rebuild failed: {e}", flush=True)
        return
    with _codec_lock:
        if _codec_spare is None:
            _codec_spare = pair


def take_codecs():
    """Hand out a pristine (encoder, decoder) pair for one session."""
    global _codec_spare
    with _codec_lock:
        pair, _codec_spare = _codec_spare, None
    if pair is None:
        pair = _build_codecs()
    threading.Thread(target=_prebuild_codecs, daemon=True).start()
    return pair

_TEXT_TOKENIZER = None


def get_text_tokenizer():
    global _TEXT_TOKENIZER
    if _TEXT_TOKENIZER is None:
        import sentencepiece
        _TEXT_TOKENIZER = sentencepiece.SentencePieceProcessor(
            str(WEIGHTS_DIR / "tokenizer_spm_32k_3.model")
        )
    return _TEXT_TOKENIZER

# Lock for model access
MODEL_LOCK = asyncio.Lock()

# Pin the voice + persona prompt in the KV cache for the whole session (see
# kv_pinning.py). PERSONAPLEX_PIN_PROMPT=0 restores the old, unpinned behaviour
# so the difference can be measured.
PIN_PROMPT = os.environ.get("PERSONAPLEX_PIN_PROMPT", "1").lower() not in (
    "0", "false", "no", "off"
)

# For /health: process start time and the quantization actually in force.
START_TIME = time.time()
CURRENT_BITS = 0

# PersonaPlex's reference server wraps system text in `<system> ... <system>`,
# but with these MLX-converted weights that makes the model go silent: a whole
# session at out_avg ~0.0002, no speech at all. `[INST] ... [/INST]` is what this
# port has always used and what produces audio, so it stays the default.
# PERSONAPLEX_PROMPT_FORMAT=system keeps the other form available for testing.
PROMPT_FORMAT = os.environ.get("PERSONAPLEX_PROMPT_FORMAT", "inst")


def unwrap_prompt_text(text: str) -> str:
    """Remove one tag pair the caller supplied, in either format."""
    cleaned = text.strip()
    if cleaned.startswith("[INST]") and cleaned.endswith("[/INST]"):
        return cleaned[len("[INST]"):-len("[/INST]")].strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned[len("<system>"):-len("<system>")].strip()
    return cleaned


def wrap_prompt_text(text: str) -> str:
    """Wrap text as system context in the format this port actually accepts.

    A tag pair the caller supplied is unwrapped first, so wrapping is
    idempotent and any extra framing still lands inside the tags.
    """
    cleaned = unwrap_prompt_text(text)
    if PROMPT_FORMAT == "system":
        return f"<system> {cleaned} <system>"
    return f"[INST] {cleaned} [/INST]"

# Available voices
VOICES = {
    "NATF0": "Natural Female 1 (Kendra)",
    "NATF1": "Natural Female 2",
    "NATF2": "Natural Female 3",
    "NATF3": "Natural Female 4",
    "NATM0": "Natural Male 1",
    "NATM1": "Natural Male 2",
    "NATM2": "Natural Male 3",
    "NATM3": "Natural Male 4",
    "VARF0": "Variety Female 1",
    "VARF1": "Variety Female 2",
    "VARF2": "Variety Female 3",
    "VARF3": "Variety Female 4",
    "VARF4": "Variety Female 5",
    "VARM0": "Variety Male 1",
    "VARM1": "Variety Male 2",
    "VARM2": "Variety Male 3",
    "VARM3": "Variety Male 4",
    "VARM4": "Variety Male 5",
}


def get_config():
    """Get PersonaPlex 7B configuration."""
    from moshi_mlx.models.lm import LmConfig, DepFormerConfig
    from moshi_mlx.modules.transformer import TransformerConfig

    transformer = TransformerConfig(
        d_model=4096, num_heads=32, num_layers=32,
        dim_feedforward=int(4.125 * 4096), causal=True, norm_first=True,
        bias_ff=False, bias_attn=False, layer_scale=None,
        context=int(os.environ.get("PERSONAPLEX_LM_CONTEXT", "3000")),
        max_period=10000, use_conv_block=False, use_conv_bias=True,
        cross_attention=False, gating=True, norm="rms_norm",
        positional_embedding="rope", conv_layout=False, conv_kernel_size=3,
        kv_repeat=1, max_seq_len=4096,
    )
    depformer = DepFormerConfig(
        transformer=TransformerConfig(
            d_model=1024, num_heads=16, num_layers=6,
            dim_feedforward=int(4.125 * 1024), causal=True, norm_first=True,
            bias_ff=False, bias_attn=False, layer_scale=None, context=8,
            max_period=10000, use_conv_block=False, use_conv_bias=True,
            cross_attention=False, gating=True, norm="rms_norm",
            positional_embedding="none", conv_layout=False, conv_kernel_size=3,
            kv_repeat=1, max_seq_len=4096,
        ),
        num_slices=16,
        weights_per_step_schedule=None,
    )
    return LmConfig(
        transformer=transformer, depformer=depformer,
        text_in_vocab_size=32001, text_out_vocab_size=32000,
        audio_vocab_size=2049, audio_codebooks=16,
        audio_delays=[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        conditioners={}, demux_second_stream=False,
    )


def save_cache_state():
    """Save current KV cache state."""
    global VOICE_CACHE_STATE

    state = {
        'transformer': [],
        'depformer': []
    }

    # Save transformer cache
    for layer_cache in MODEL.transformer_cache:
        kv_cache = layer_cache.self_attn
        layer_state = {
            'keys': mx.array(kv_cache.keys) if kv_cache.keys is not None else None,
            'values': mx.array(kv_cache.values) if kv_cache.values is not None else None,
            'offset': kv_cache.offset,
            '_idx': kv_cache._idx if hasattr(kv_cache, '_idx') else 0,
        }
        state['transformer'].append(layer_state)

    # Save depformer cache (usually not needed but just in case)
    for layer_cache in MODEL.depformer_cache:
        kv_cache = layer_cache.self_attn
        layer_state = {
            'keys': mx.array(kv_cache.keys) if kv_cache.keys is not None else None,
            'values': mx.array(kv_cache.values) if kv_cache.values is not None else None,
            'offset': kv_cache.offset,
        }
        state['depformer'].append(layer_state)

    VOICE_CACHE_STATE = state
    print(f"Saved cache state (offset={state['transformer'][0]['offset']})")


def restore_cache_state():
    """Restore KV cache state from saved voice prompt state."""
    global VOICE_CACHE_STATE

    if VOICE_CACHE_STATE is None:
        print("No saved cache state to restore!")
        return

    # Restore transformer cache
    for i, layer_cache in enumerate(MODEL.transformer_cache):
        kv_cache = layer_cache.self_attn
        saved = VOICE_CACHE_STATE['transformer'][i]
        if saved['keys'] is not None:
            kv_cache.keys = mx.array(saved['keys'])
            kv_cache.values = mx.array(saved['values'])
            kv_cache.offset = saved['offset']
            if hasattr(kv_cache, '_idx'):
                kv_cache._idx = saved['_idx']

    # Depformer cache is reset between tokens anyway
    for layer_cache in MODEL.depformer_cache:
        layer_cache.self_attn.reset()


def forward_embeddings(embeddings: mx.array):
    """
    Forward pass through transformer using pre-computed embeddings.

    This is used for replaying voice prompt embeddings.
    """
    transformer_out = MODEL.transformer(embeddings, cache=MODEL.transformer_cache)
    transformer_out = MODEL.out_norm(transformer_out)
    text_logits = MODEL.text_linear(transformer_out)
    mx.eval(transformer_out)  # Force evaluation to update cache
    return transformer_out, text_logits


def step_embeddings(embedding: mx.array, text_sampler, audio_sampler):
    """
    Step through one embedding from voice prompt.

    Processes the embedding through the transformer and samples tokens
    to keep the depformer cache consistent.
    """
    # Forward through transformer
    transformer_out, text_logits = forward_embeddings(embedding)

    # Sample text token
    text_token, _ = text_sampler(text_logits)

    # Sample audio tokens through depformer
    for c in MODEL.depformer_cache:
        c.reset()

    audio_tokens = MODEL.depformer.sample(
        transformer_out,
        audio_sampler,
        text_token,
        MODEL.depformer_cache,
    )

    return text_token, audio_tokens


def step_silence(text_sampler, audio_sampler, num_frames=1):
    """Step through silence frames."""
    silence = mx.array(SILENCE_TOKENS.reshape(1, 8, 1))
    sine = mx.array(SINE_TOKENS.reshape(1, 8, 1))
    text_token = mx.array([[ZERO_TEXT_CODE]])

    # Full 16 codebooks: 8 silence (moshi) + 8 sine (user)
    audio_tokens = []
    for cb in range(8):
        audio_tokens.append(silence[:, cb, :])
    for cb in range(8):
        audio_tokens.append(sine[:, cb, :])

    for _ in range(num_frames):
        text_token, _ = MODEL.sample(
            text_token, audio_tokens, text_sampler, audio_sampler
        )
        mx.eval(text_token)


def conditioning_audio_tokens():
    """Audio input used while the model is *reading* rather than conversing.

    During prompt conditioning the moshi stream is silence and the user stream
    is a sine tone; that is what the model was trained on for text-only
    conditioning, so silent context injection reuses it verbatim.
    """
    silence = mx.array(SILENCE_TOKENS.reshape(1, 8, 1))
    sine = mx.array(SINE_TOKENS.reshape(1, 8, 1))
    out = []
    for cb in range(8):
        out.append(silence[:, cb, :])
    for cb in range(8):
        out.append(sine[:, cb, :])
    return out


def step_text_prompt(text_tokens: list[int], text_sampler, audio_sampler):
    """Step through text prompt tokens."""
    audio_tokens = conditioning_audio_tokens()

    text_token = mx.array([[ZERO_TEXT_CODE]])

    for token_id in text_tokens:
        text_token = mx.array([[token_id]])
        text_token, _ = MODEL.sample(
            text_token, audio_tokens, text_sampler, audio_sampler
        )
        mx.eval(text_token)


def inject_context_tokens(text_tokens: list[int], text_sampler, audio_sampler):
    """Push factual text into a live session's context without speaking it.

    Same mechanism as prompt conditioning (text tokens in, silence+sine audio
    in, generated audio thrown away), applied on top of an existing
    conversation instead of a freshly reset cache. Returns the text token the
    model should continue from.

    This advances the attention cache by len(text_tokens) frames and the caller
    must be holding MODEL_LOCK, since MODEL.sample mutates the shared cache.
    """
    audio_tokens = conditioning_audio_tokens()
    text_token = mx.array([[ZERO_TEXT_CODE]])
    for token_id in text_tokens:
        text_token = mx.array([[token_id]])
        text_token, _ = MODEL.sample(
            text_token, audio_tokens, text_sampler, audio_sampler
        )
        mx.eval(text_token)
    # Prompt conditioning ends with a silence frame for a reason: without it the
    # model is left in "reading" mode and stops speaking entirely. Mirror that.
    step_silence(text_sampler, audio_sampler, num_frames=1)
    return mx.array([[ZERO_TEXT_CODE]])



def load_voice_prompt(voice_name: str, text_prompt: str):
    """
    Load and process voice prompt.

    This replays the voice prompt embeddings through the transformer,
    then processes text prompt, setting up the KV cache with voice characteristics.
    """
    global VOICE_CACHE_STATE, CURRENT_VOICE, CURRENT_PROMPT
    CURRENT_VOICE, CURRENT_PROMPT = voice_name, text_prompt

    from moshi_mlx.utils import sampling
    import sentencepiece

    # Reset all caches first
    for c in MODEL.transformer_cache:
        c.reset()
    for c in MODEL.depformer_cache:
        c.reset()

    text_sampler = sampling.Sampler(temp=TEXT_TEMP, top_k=TEXT_TOP_K)
    audio_sampler = sampling.Sampler(temp=AUDIO_TEMP, top_k=AUDIO_TOP_K)

    # Load voice prompt embeddings
    voice_path = VOICES_DIR / f"{voice_name}.pt"
    print(f"Loading voice prompt: {voice_path}")

    # Load PyTorch state dict
    state = torch.load(voice_path, map_location='cpu', weights_only=False)
    # Convert bfloat16 to float32 for numpy compatibility
    embeddings = state['embeddings'].float().numpy()  # [51, 1, 1, 4096]

    # Replay voice prompt embeddings
    print(f"Replaying {embeddings.shape[0]} voice prompt frames...", flush=True)
    for i, emb in enumerate(embeddings):
        # emb shape: [1, 1, 4096]
        emb_mx = mx.array(emb)
        text_token, audio_tokens = step_embeddings(emb_mx, text_sampler, audio_sampler)
        mx.eval(text_token)
        if audio_tokens is not None:
            mx.eval(audio_tokens)

    # Check cache offset after voice prompt
    cache_offset = MODEL.transformer_cache[0].self_attn.offset
    print(f"Done loading voice prompt. Cache offset: {cache_offset}", flush=True)

    # Step through silence
    print("Loading audio silence...", flush=True)
    try:
        step_silence(text_sampler, audio_sampler, num_frames=1)
        print("Done loading audio silence.", flush=True)
    except Exception as e:
        print(f"Error in step_silence: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Process text prompt
    print("Loading text prompt...", flush=True)
    try:
        tokenizer = sentencepiece.SentencePieceProcessor(str(WEIGHTS_DIR / "tokenizer_spm_32k_3.model"))
        wrapped_prompt = wrap_prompt_text(text_prompt)
        text_tokens = tokenizer.encode(wrapped_prompt)
        step_text_prompt(text_tokens, text_sampler, audio_sampler)
        print("Done loading text prompt.", flush=True)
    except Exception as e:
        print(f"Error in step_text_prompt: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Final silence
    print("Loading final audio silence...", flush=True)
    try:
        step_silence(text_sampler, audio_sampler, num_frames=1)
        print("Done loading final audio silence.", flush=True)
    except Exception as e:
        print(f"Error in final step_silence: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Check final cache state
    final_offset = MODEL.transformer_cache[0].self_attn.offset
    print(f"Final cache offset: {final_offset}", flush=True)

    # Save cache state for restoration on new connections
    save_cache_state()

    # From here on the prompt is never overwritten: the conversation slides
    # behind it. Without this it leaves the attention window after ~4 minutes
    # (see kv_pinning.py).
    if PIN_PROMPT:
        pinned = kv_pinning.pin_prompt(MODEL.transformer_cache)
        window = kv_pinning.window_frames(MODEL.transformer_cache)
        print(f"Pinned {pinned} system prompt frames "
              f"(window {window} frames, sliding {window - pinned})", flush=True)
    else:
        # Still take the snapshot, so /health?verify=1 can show the prompt
        # being evicted once the conversation passes the window.
        kv_pinning.snapshot(MODEL.transformer_cache)
        print("System prompt pinning disabled (PERSONAPLEX_PIN_PROMPT=0)", flush=True)

    print("Voice conditioning complete!", flush=True)


def trim_depformer(model):
    """Drop the depformer slices whose output is never used.

    The model has 16 audio codebooks. Slices 0-7 are Moshi's own audio, slices
    8-15 predict the *user's* stream and are only meaningful when teacher forced
    with real user tokens (as happens during training). We always discard them,
    but running them costs ~15-20ms per frame, a big chunk of the 80ms budget.
    Slices are causal, so dropping the tail cannot change slices 0-7.
    """
    if len(model.depformer.slices) > USED_AUDIO_CODEBOOKS:
        model.depformer.slices = model.depformer.slices[:USED_AUDIO_CODEBOOKS]
        model.depformer_cache = model.depformer.slices[0].transformer.make_cache()


def quantize_model(model, bits: int, scope: str | None = None):
    """Quantize according to QUANT_SCOPE.

    "all" quantizes everything (fastest). "transformer" quantizes only the 7B
    body and leaves the depformer plus the text/audio heads at full precision -
    a few ms slower, but the heads are where the next text token is chosen, so
    fact recall and prosody both improve.
    """
    import mlx.nn as nn

    if bits <= 0:
        return
    target = model if (scope or QUANT_SCOPE) != "transformer" else model.transformer
    nn.quantize(target, group_size=64, bits=bits)


def load_model():
    """Load PersonaPlex model and Mimi codec."""
    global MODEL, CONFIG, CURRENT_BITS

    from moshi_mlx.models.lm import Lm

    print("Loading PersonaPlex 7B...")
    CONFIG = get_config()
    MODEL = Lm(CONFIG)
    MODEL.load_pytorch_weights(str(WEIGHTS_DIR / "model.safetensors"), CONFIG, strict=False)
    trim_depformer(MODEL)

    scope = "transformer only" if QUANT_SCOPE == "transformer" else "all layers"
    print(f"Quantizing to 4-bit ({scope})...")
    quantize_model(MODEL, 4)
    CURRENT_BITS = 4
    mx.eval(MODEL.parameters())

    print("Loading Mimi codec...")
    _prebuild_codecs()  # first session's pair; a spare is built per session

    print("Warming up...")
    MODEL.warmup()

    # Load default voice prompt
    print("\nLoading voice conditioning (this may take a moment)...")
    load_voice_prompt("NATF0", CURRENT_PROMPT)

    print("Ready!")


# FastAPI app
app = FastAPI(title="PersonaPlex MLX Live")


@app.on_event("startup")
async def startup_event():
    """Load models on server startup."""
    load_model()


@app.get("/")
async def get_client():
    return HTMLResponse(content=HTML_CLIENT, status_code=200)


@app.get("/health")
async def health(verify: int = 0):
    """Liveness/readiness probe.

    Deliberately cheap by default: it never touches the model or the lock, so
    it is safe to poll while a conversation is running. Returns 200 only when
    the model is loaded and voice-conditioned, 503 otherwise, so a probe can
    just read the status code.

    ``?verify=1`` adds ``prompt_intact``: an exact comparison of the pinned
    system-prompt KV prefix against what was pinned. That does GPU work, so it
    is opt-in rather than part of every poll.
    """
    loaded = MODEL is not None
    conditioned = VOICE_CACHE_STATE is not None
    ready = loaded and conditioned
    payload = {
        "status": "ok" if ready else "starting",
        "ready": ready,
        "model_loaded": loaded,
        "conditioned": conditioned,
        "voice": CURRENT_VOICE,
        "bits": CURRENT_BITS,
        "quant_scope": QUANT_SCOPE,
        "session_active": MODEL_LOCK.locked(),
        "sample_rate": SAMPLE_RATE,
        "frame_size": FRAME_SIZE,
        "frame_rate": FRAME_RATE,
        "uptime_s": round(time.time() - START_TIME, 1),
    }
    if loaded and MODEL.transformer_cache:
        payload.update(kv_pinning.stats(MODEL.transformer_cache, verify=bool(verify)))
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/voices")
async def get_voices():
    return VOICES


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing, for clients that probe it first.

    This is a compatibility stub only: the server does not implement
    /v1/chat/completions or /v1/realtime. Audio and text streaming live on the
    custom websocket protocol at /ws.
    """
    return {
        "object": "list",
        "data": [{
            "id": "personaplex-mlx",
            "object": "model",
            "created": int(START_TIME),
            "owned_by": "local",
        }],
    }


@app.post("/set-voice")
async def set_voice(voice: str = "NATF0", prompt: str = "You are a friendly voice assistant on a call."):
    """Change voice and persona. Requires re-conditioning."""
    if voice not in VOICES:
        return {"error": f"Unknown voice: {voice}"}
    if MODEL_LOCK.locked():
        # Re-conditioning rewrites the shared KV cache; doing that under a live
        # session would corrupt it mid-utterance.
        return {"error": "a live session is running; stop it before changing voice"}

    print(f"Changing voice to {voice} with prompt: {prompt[:50]}...")

    # Re-apply voice conditioning with new settings
    async with MODEL_LOCK:
        load_voice_prompt(voice, prompt)

    return {"status": "ok", "voice": voice}


@app.post("/set-quant")
async def set_quantization(bits: int = 4, scope: str | None = None):
    """Change model quantization level. Requires model reload."""
    global MODEL, QUANT_SCOPE, CURRENT_BITS
    from moshi_mlx.models.lm import Lm

    if bits not in [0, 4, 5, 6, 8]:
        return {"error": "bits must be 0, 4, 5, 6, or 8"}
    if scope is not None:
        if scope not in ("all", "transformer"):
            return {"error": "scope must be 'all' or 'transformer'"}
        QUANT_SCOPE = scope
    if MODEL_LOCK.locked():
        return {"error": "a live session is running; stop it before reloading the model"}

    print(f"Reloading model with {bits}-bit quantization ({QUANT_SCOPE})..."
          if bits > 0 else "Reloading model without quantization...")

    async with MODEL_LOCK:
        # Reload model from scratch
        MODEL = Lm(CONFIG)
        MODEL.load_pytorch_weights(str(WEIGHTS_DIR / "model.safetensors"), CONFIG, strict=False)
        trim_depformer(MODEL)
        quantize_model(MODEL, bits)
        CURRENT_BITS = bits

        mx.eval(MODEL.parameters())
        MODEL.warmup()

        # Re-apply the voice and persona that are actually in use, not the default.
        print(f"Re-applying voice {CURRENT_VOICE} after quantization change...")
        load_voice_prompt(CURRENT_VOICE, CURRENT_PROMPT)

    return {"status": "ok", "bits": bits, "scope": QUANT_SCOPE}


@app.get("/current-quant")
async def get_current_quant():
    """Get current quantization level."""
    # Check if model has quantized layers
    import mlx.nn as nn
    for name, module in MODEL.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            return {"bits": module.bits, "scope": QUANT_SCOPE}
    return {"bits": 0, "scope": QUANT_SCOPE}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle real-time audio streaming."""
    await websocket.accept()

    from moshi_mlx.utils import sampling

    # Fresh codec state for this session. rustymimi's .reset() does not clear
    # everything, so a reused tokenizer eventually dies mid-conversation (see
    # take_codecs).
    enc_tok, dec_tok = take_codecs()

    # Restore voice cache state for this connection
    restore_cache_state()
    cache_offset = MODEL.transformer_cache[0].self_attn.offset
    print(f"Client connected. Cache restored to offset: {cache_offset}", flush=True)

    text_sampler = sampling.Sampler(temp=TEXT_TEMP, top_k=TEXT_TOP_K)
    audio_sampler = sampling.Sampler(temp=AUDIO_TEMP, top_k=AUDIO_TOP_K)

    # --- session protocol state -------------------------------------------
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    # One writer owns the socket: audio bytes and control JSON must never
    # interleave, and a slow client must not stall the LM loop.
    send_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    # Queued context.inject jobs, applied at the next quiet boundary.
    pending_context: list[dict] = []
    # Facts that are already in this session's attention cache. Kept so they can
    # be re-applied after session.reset instead of silently disappearing - an
    # application's context is not the same thing as conversational state.
    applied_facts: list[dict] = []
    # None, or {"keep_context": bool} when a reset is waiting for a pause.
    reset_requested: list = [None]
    # Rolling window of (model_rms, mic_rms) used to detect a conversational
    # pause. Context injection mutates the attention cache, so it is only ever
    # done here rather than mid-turn.
    quiet = collections.deque(maxlen=max(1, INJECT_QUIET_FRAMES))
    last_out_rms = 0.0

    # Text token to seed the first live step with. The voice prompt conditioning
    # ends right after a silence frame, so the right token here is the text
    # padding code, not the "start of sequence" code.
    text_token = mx.array([[ZERO_TEXT_CODE]])

    # Track Moshi's previous audio output for full-duplex
    # Initialize with silence tokens for first frame
    prev_moshi_audio = mx.array(SILENCE_TOKENS.reshape(1, 8, 1), dtype=mx.int32)
    # The user stream is delayed by one frame (see the loop below), so the first
    # step has to feed the frame before the first real mic frame.
    prev_user_tokens = mx.array(SILENCE_TOKENS.reshape(1, 8, 1), dtype=mx.int32)

    # Pipeline: mic -> encode (CPU thread) -> LM (GPU) -> decode (CPU thread) -> ws
    #
    # Encoding and decoding are ~25ms each and independent of each other, so each
    # gets its own thread and overlaps with the LM step. That takes the per-frame
    # critical path from ~95ms (encode+LM+decode, serial) to ~45ms, comfortably
    # inside the 80ms budget. Dropping mic frames is then a rare fallback instead
    # of the normal case: at ~20% loss the model can no longer understand you and
    # just repeats its canned greeting.
    raw_q: queue.Queue = queue.Queue(maxsize=4)     # (frame, mic_rms)
    out_q: queue.Queue = queue.Queue(maxsize=3)     # (moshi_tokens, mic_rms)
    pcm_q: queue.Queue = queue.Queue(maxsize=6)     # (pcm_bytes, mic_rms, out_rms)
    tok_q: asyncio.Queue = asyncio.Queue(maxsize=3)  # (user_tokens, mic_rms)
    loop = asyncio.get_running_loop()
    STOP = object()

    async def writer_loop():
        """Sole owner of websocket.send_*, so frames never interleave."""
        while True:
            item = await send_q.get()
            if item is STOP:
                return
            kind, payload = item
            try:
                if kind == "bin":
                    await websocket.send_bytes(payload)
                else:
                    await websocket.send_text(payload)
            except Exception:
                return

    def send_event(event: dict):
        payload = json.dumps(event)
        try:
            send_q.put_nowait(("txt", payload))
        except asyncio.QueueFull:
            pass  # control events are advisory; never block audio for them

    def queue_audio(pcm_bytes: bytes):
        try:
            send_q.put_nowait(("bin", pcm_bytes))
        except asyncio.QueueFull:
            # Drop the oldest queued frame rather than the newest: keeping the
            # freshest audio is what keeps the conversation in sync.
            try:
                send_q.get_nowait()
                send_q.put_nowait(("bin", pcm_bytes))
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def handle_control(msg: dict):
        """Protocol messages from the application (never from the model)."""
        kind = msg.get("type")

        if kind == "context.inject":
            req_id = msg.get("id")
            raw = (msg.get("content") or "").strip()
            if not raw:
                send_event({"type": "context.rejected", "session_id": session_id,
                            "id": req_id, "reason": "empty content"})
                return
            # Frame the fact as supplied context, not as something said: the
            # model otherwise answers "that's what you said" or reads it aloud.
            content = wrap_prompt_text(f"{INJECT_PREFIX}{unwrap_prompt_text(raw)}")
            tokens = get_text_tokenizer().encode(content)
            if not tokens:
                send_event({"type": "context.rejected", "session_id": session_id,
                            "id": req_id, "reason": "could not tokenize content"})
                return
            if len(tokens) > MAX_INJECT_TOKENS:
                send_event({"type": "context.rejected", "session_id": session_id,
                            "id": req_id,
                            "reason": f"content is {len(tokens)} tokens including framing, "
                                      f"limit is {MAX_INJECT_TOKENS}"})
                return
            pending_context.append({"id": req_id, "content": raw, "tokens": tokens,
                                    "queued_at": time.time()})
            send_event({"type": "context.accepted", "session_id": session_id,
                        "id": req_id, "tokens": len(tokens), "queued": True,
                        "framed": content,
                        "note": "applied at the next conversational pause; "
                                "accepted is not the same as understood"})
            return

        if kind == "session.reset":
            # Clear the conversation. Application context is not conversational
            # state, so by default the facts the app injected are re-applied to
            # the fresh cache rather than silently forgotten. Pass
            # keep_context=false to drop them too.
            keep_context = msg.get("keep_context", True)
            if not isinstance(keep_context, bool):
                send_event({"type": "context.rejected", "session_id": session_id,
                            "id": msg.get("id"),
                            "reason": "keep_context must be a boolean"})
                return
            reset_requested[0] = {"keep_context": keep_context}
            send_event({"type": "session.reset.queued", "session_id": session_id,
                        "keep_context": keep_context,
                        "note": "applied at the next conversational pause"})
            return

        if kind == "session.start":
            send_event({"type": "session.started", "session_id": session_id,
                        "voice": CURRENT_VOICE, "sample_rate": SAMPLE_RATE,
                        "frame_size": FRAME_SIZE, "frame_rate": FRAME_RATE})
            return

        send_event({"type": "context.rejected", "session_id": session_id,
                    "id": msg.get("id"), "reason": f"unknown message type {kind!r}"})

    def put_tokens(item):
        # Called from the encoder thread; put_nowait on an asyncio.Queue is only
        # safe through call_soon_threadsafe.
        try:
            tok_q.put_nowait(item)
        except asyncio.QueueFull:
            dropped[0] += 1

    disconnected = asyncio.Event()
    dropped = [0]
    mic_capture = []

    def save_mic_capture():
        import wave
        samples = np.concatenate(mic_capture)
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        with wave.open("mic_received.wav", "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        print("Saved mic_received.wav", flush=True)

    async def read_frames():
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                data = message.get("bytes")
                text = message.get("text")
                if data is None and text is not None:
                    # Control plane: context.inject, session.reset, session.start
                    try:
                        msg = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict):
                        handle_control(msg)
                    continue
                if data is None:
                    continue
                if len(data) < FRAME_SIZE * 4:
                    continue
                frame = np.frombuffer(data[:FRAME_SIZE * 4], dtype=np.float32).copy()
                if len(mic_capture) < 125:
                    mic_capture.append(frame.copy())
                    if len(mic_capture) == 125:
                        save_mic_capture()
                mic_rms = float(np.sqrt(np.mean(frame ** 2)))
                while True:
                    try:
                        raw_q.put_nowait((frame, mic_rms))
                        break
                    except queue.Full:
                        try:
                            raw_q.get_nowait()
                            dropped[0] += 1
                        except queue.Empty:
                            pass
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()
        except Exception as e:
            print(f"Frame reader error: {e}")
            disconnected.set()
        finally:
            raw_q.put(STOP)

    def encoder_loop():
        nonlocal enc_tok
        frames = 0
        while True:
            item = raw_q.get()
            if item is STOP:
                loop.call_soon_threadsafe(put_tokens, STOP)
                return
            frame, mic_rms = item
            try:
                ut = enc_tok.encode_step(frame[None, None, :])
                loop.call_soon_threadsafe(put_tokens, (mx.array(ut.astype(np.int64)), mic_rms))
            except Exception as e:
                # Do not kill the conversation over a codec hiccup: swap in a
                # fresh encoder and keep going.
                print(f"Encoder error: {e} - rebuilding Mimi encoder", flush=True)
                try:
                    enc_tok, _ = _build_codecs()
                    frames = 0
                except Exception as e2:
                    print(f"Encoder rebuild failed: {e2}", flush=True)
                    loop.call_soon_threadsafe(put_tokens, STOP)
                    return
                continue
            frames += 1
            if frames >= CODEC_FRAME_LIMIT:
                # Prebuilt spare, so this is an instant pointer swap: no audio
                # gap, and we never reach the overflow at 4096.
                frames = 0
                try:
                    enc_tok, _ = take_codecs()
                    print(f"Rotated Mimi encoder after {CODEC_FRAME_LIMIT} frames", flush=True)
                except Exception as e:
                    print(f"Encoder swap failed: {e}", flush=True)

    def decoder_loop():
        nonlocal dec_tok
        frames = 0
        while True:
            item = out_q.get()
            if item is STOP:
                pcm_q.put(STOP)
                return
            audio_tokens, mic_rms = item
            try:
                out = dec_tok.decode_step(audio_tokens)
                frame = out[0, 0].astype(np.float32)
                pcm_q.put((frame.tobytes(), mic_rms, float(np.sqrt(np.mean(frame ** 2)))))
            except Exception as e:
                print(f"Decoder error: {e} - rebuilding Mimi decoder", flush=True)
                try:
                    _, dec_tok = _build_codecs()
                    frames = 0
                except Exception as e2:
                    print(f"Decoder rebuild failed: {e2}", flush=True)
                    pcm_q.put(STOP)
                    return
                continue
            frames += 1
            if frames >= CODEC_FRAME_LIMIT:
                frames = 0
                try:
                    _, dec_tok = take_codecs()
                    print(f"Rotated Mimi decoder after {CODEC_FRAME_LIMIT} frames", flush=True)
                except Exception as e:
                    print(f"Decoder swap failed: {e}", flush=True)

    reader_task = asyncio.create_task(read_frames())
    writer_task = asyncio.create_task(writer_loop())
    enc_thread = threading.Thread(target=encoder_loop, name="mimi-encode", daemon=True)
    dec_thread = threading.Thread(target=decoder_loop, name="mimi-decode", daemon=True)

    send_event({"type": "session.started", "session_id": session_id,
                "voice": CURRENT_VOICE, "quant_scope": QUANT_SCOPE,
                "sample_rate": SAMPLE_RATE, "frame_size": FRAME_SIZE,
                "frame_rate": FRAME_RATE})

    frame_count = 0
    total_gen_time = 0
    session_start = time.time()
    mic_levels = []
    out_levels = []
    text_seq = 0
    # Frames left in the post-injection silence window (see INJECT_MUTE_FRAMES).
    mute_frames_left = [0]

    def apply_context_job(job: dict, forced: bool = False) -> bool:
        """Put one queued update into the live cache. Caller holds MODEL_LOCK."""
        nonlocal text_token, prev_moshi_audio, prev_user_tokens
        t_inject = time.time()
        try:
            text_token = inject_context_tokens(
                job["tokens"], text_sampler, audio_sampler
            )
            # The model has "read" the update; resume as if it were silence
            # before it, so no speech is emitted from it.
            prev_moshi_audio = mx.array(SILENCE_TOKENS.reshape(1, 8, 1), dtype=mx.int32)
            quiet.clear()
            # The quiet-boundary gate cannot stop the model from *starting* the
            # injected sentence right after reading it ("His name is Biscuit.").
            # Discard the next second of generated frames and feed silence back.
            mute_frames_left[0] = INJECT_MUTE_FRAMES
            # Remember it so session.reset can restore it.
            applied_facts.append(job)
            # Oldest facts fall out of the replay set first: the cache itself
            # keeps them, this only bounds future restores.
            while (len(applied_facts) > 1
                   and sum(len(f["tokens"]) for f in applied_facts) > MAX_CONTEXT_TOKENS):
                evicted = applied_facts.pop(0)
                print(f"Context replay budget full, dropping {evicted['id']}", flush=True)
            send_event({
                "type": "context.applied",
                "session_id": session_id,
                "id": job["id"],
                "tokens": len(job["tokens"]),
                "offset": int(MODEL.transformer_cache[0].self_attn.offset),
                "latency_ms": round((time.time() - t_inject) * 1000, 1),
                "waited_ms": round((t_inject - job.get("queued_at", t_inject)) * 1000, 1),
                "forced": forced,
            })
            return True
        except Exception as e:
            print(f"Context injection failed: {e}", flush=True)
            send_event({"type": "context.rejected",
                        "session_id": session_id, "id": job["id"],
                        "reason": f"injection failed: {e}"})
            return False

    try:
        async with MODEL_LOCK:
            enc_thread.start()
            dec_thread.start()
            while not disconnected.is_set():
                # User tokens come from the encoder thread, so encoding has already
                # overlapped with the previous LM step.
                try:
                    item = await asyncio.wait_for(tok_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No mic frames means the loop is idle. Still honour the
                    # injection deadline: otherwise a client that stops sending
                    # audio can hold an update past PERSONAPLEX_INJECT_MAX_WAIT
                    # indefinitely, because the gate is only evaluated per frame.
                    if pending_context:
                        job = pending_context[0]
                        if (time.time() - job.get("queued_at", time.time())
                                >= INJECT_MAX_WAIT_S):
                            pending_context.pop(0)
                            apply_context_job(job, forced=True)
                    continue
                if item is STOP:
                    break
                user_tokens, mic_rms = item

                # ---- control actions ----------------------------------------
                # A reset replaces the cache outright, so it does NOT need a
                # pause. Gating it on silence was a deadlock: when the model
                # starts looping it never goes quiet, so the one action that
                # could rescue the conversation could never run.
                if reset_requested[0] is not None:
                    opts = reset_requested[0]
                    reset_requested[0] = None
                    keep_context = opts["keep_context"]
                    restore_cache_state()

                    # Replay the application's facts onto the fresh cache. Without
                    # this a reset silently destroys knowledge the app already
                    # supplied, which is the difference between a demo and an
                    # interface an application can rely on.
                    replayed = 0
                    if keep_context:
                        for fact in applied_facts:
                            try:
                                inject_context_tokens(
                                    fact["tokens"], text_sampler, audio_sampler
                                )
                                replayed += 1
                            except Exception as e:
                                print(f"Context replay failed: {e}", flush=True)
                    else:
                        applied_facts.clear()

                    text_token = mx.array([[ZERO_TEXT_CODE]])
                    prev_moshi_audio = mx.array(SILENCE_TOKENS.reshape(1, 8, 1), dtype=mx.int32)
                    prev_user_tokens = mx.array(SILENCE_TOKENS.reshape(1, 8, 1), dtype=mx.int32)
                    quiet.clear()
                    # A reset interrupts any post-injection muting: the cache is
                    # fresh, so there is nothing left to suppress.
                    mute_frames_left[0] = 0
                    send_event({"type": "session.reset", "session_id": session_id,
                                "offset": int(MODEL.transformer_cache[0].self_attn.offset),
                                "keep_context": keep_context})
                    if keep_context:
                        send_event({"type": "context.restored",
                                    "session_id": session_id,
                                    "facts": replayed,
                                    "offset": int(MODEL.transformer_cache[0].self_attn.offset),
                                    "note": "injected facts replayed onto the fresh cache"})

                # Injection appends to the cache, so it still prefers a pause -
                # but only for INJECT_MAX_WAIT_S. A model that talks without
                # stopping must not be able to hold an update hostage forever.
                if pending_context:
                    at_pause = (
                        len(quiet) >= max(1, INJECT_QUIET_FRAMES)
                        and all(o < INJECT_OUT_RMS and m < INJECT_MIC_RMS
                                for o, m in quiet)
                    )
                    job = pending_context[0]
                    waited = time.time() - job.get("queued_at", time.time())
                    if at_pause or waited >= INJECT_MAX_WAIT_S:
                        pending_context.pop(0)
                        apply_context_job(job, forced=not at_pause)

                # Full-duplex: combine Moshi's previous output + the user's
                # *previous* frame.
                #
                # NVIDIA's reference writes provided tokens at offset+delay, and
                # audio_delays is 1 for every user codebook, so the model was
                # trained to see the user's frame t-1 while generating frame t.
                # Feeding frame t instead (80ms early) makes replies drift off
                # topic; with the reference delay the same clip stays grounded
                # on what was actually said.
                all_audio_input = []
                for cb in range(8):
                    all_audio_input.append(prev_moshi_audio[:, cb, :])
                for cb in range(8):
                    all_audio_input.append(prev_user_tokens[:, cb, :])

                # Generate response
                start = time.time()
                next_text, next_audio = MODEL.sample(
                    text_token, all_audio_input, text_sampler, audio_sampler,
                )
                mx.eval(next_text, next_audio)
                total_gen_time += time.time() - start

                # Never speak injected content. The frame just generated is
                # conditioned on the update the model has only just read, so it
                # is often the injection itself continued as speech. Throw this
                # window away, feed silence back, and let the model start a
                # fresh utterance once the update is no longer "current".
                if mute_frames_left[0] > 0:
                    mute_frames_left[0] -= 1
                    # Discard frames the decoder finished before the update so
                    # they are not sent late, after the mute.
                    while True:
                        try:
                            stale = pcm_q.get_nowait()
                        except queue.Empty:
                            break
                        if stale is STOP:
                            # Leave the sentinel for the normal drain path.
                            try:
                                pcm_q.put_nowait(STOP)
                            except queue.Full:
                                pass
                            break
                    text_token = mx.array([[ZERO_TEXT_CODE]])
                    prev_moshi_audio = mx.array(
                        SILENCE_TOKENS.reshape(1, 8, 1), dtype=mx.int32
                    )
                    prev_user_tokens = user_tokens
                    frame_count += 1
                    queue_audio(np.zeros(FRAME_SIZE, dtype=np.float32).tobytes())
                    quiet.append((0.0, mic_rms))
                    if LOG_TEXT:
                        print("·", end="", flush=True)
                    continue

                # Save Moshi's output for next iteration (first 8 codebooks)
                prev_moshi_audio = next_audio[:, :8, :]
                prev_user_tokens = user_tokens
                text_token = next_text
                frame_count += 1

                if LOG_TEXT:
                    tid = int(next_text[0, 0])
                    if tid in (0, 3):
                        print("·", end="", flush=True)
                    else:
                        piece = get_text_tokenizer().id_to_piece(tid).replace("\u2581", " ")
                        print(piece, end="", flush=True)

                # Stream the text stream to the application. This is the model's
                # own transcript of what it is saying - it is never interpreted as
                # input here; applications decide what to do with it.
                if EMIT_TEXT_DELTAS:
                    tid = int(next_text[0, 0])
                    if tid not in (0, 3):
                        piece = get_text_tokenizer().id_to_piece(tid).replace("\u2581", " ")
                        if piece:
                            send_event({
                                "type": "generation.text_delta",
                                "session_id": session_id,
                                "seq": text_seq,
                                "frame": frame_count,
                                "token_id": tid,
                                "text": piece,
                            })
                            text_seq += 1

                # Hand the tokens to the decoder thread; it runs while we generate
                # the next frame. Convert to numpy here so the decoder thread never
                # touches an MLX array.
                out_q.put((np.array(next_audio[:, :8, :]).astype(np.uint32), mic_rms))

                # Send every decoded frame that is ready (usually the one the
                # decoder produced during this LM step).
                while True:
                    try:
                        payload = pcm_q.get_nowait()
                    except queue.Empty:
                        break
                    if payload is STOP:
                        break
                    pcm_bytes, mic_rms_f, out_rms = payload
                    mic_levels.append(mic_rms_f)
                    out_levels.append(out_rms)
                    last_out_rms = out_rms
                    queue_audio(pcm_bytes)

                # Track quietness for boundary detection: model output on this
                # frame, and the user's input that produced it.
                quiet.append((last_out_rms, mic_rms))

                if len(mic_levels) >= 13:
                    print(
                        f"1s AUDIO | mic_avg={np.mean(mic_levels):.4f} "
                        f"mic_peak={max(mic_levels):.4f} "
                        f"out_avg={np.mean(out_levels):.4f}",
                        flush=True
                    )
                    mic_levels.clear()
                    out_levels.clear()

                # Send stats every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - session_start
                    avg_latency = (total_gen_time / frame_count) * 1000
                    # measured against wall clock, not just generation time
                    rtf = (frame_count / FRAME_RATE) / elapsed if elapsed else 0.0
                    stats = {
                        "type": "stats",
                        "session_id": session_id,
                        "frames": frame_count,
                        "avg_latency_ms": round(avg_latency, 1),
                        "rtf": round(rtf, 2),
                    }
                    send_event(stats)

    except WebSocketDisconnect:
        print(f"Client disconnected after {frame_count} frames")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader_task.cancel()
        try:
            send_q.put_nowait(STOP)
        except asyncio.QueueFull:
            pass
        for q in (raw_q, out_q):
            try:
                q.put_nowait(STOP)
            except queue.Full:
                # wake the worker so it can see the sentinel
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(STOP)
                except queue.Full:
                    pass
        if LOG_TEXT:
            print(flush=True)
        print(
            f"Session done: {frame_count} frames, {dropped[0]} stale mic frames dropped",
            flush=True,
        )
        # Dropped mic frames are not just latency: each one is 80ms of missing
        # speech, so Mimi sees a chopped token stream and the model answers
        # something unrelated. Make that loud instead of a silent stat.
        sent_or_dropped = frame_count + dropped[0]
        if dropped[0] and dropped[0] > max(5, sent_or_dropped // 20):
            pct = 100 * dropped[0] / max(1, sent_or_dropped)
            print(
                f"  ! WARNING: {pct:.0f}% of mic frames were dropped - the machine "
                f"cannot keep up in real time.\n"
                f"    This garbles the model's audio tokens and is the usual cause "
                f"of replies that ignore what was said.\n"
                f"    Check `top` for other CPU hogs (browsers, CI runners, Xcode), "
                f"or run 4-bit (/set-quant).",
                flush=True,
            )


HTML_CLIENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PersonaPlex MLX Live</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #0a0a0a 100%);
            color: #fff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
        }

        h1 {
            color: #76B900;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: #888;
            margin-bottom: 1rem;
        }

        .badge {
            background: #76B900;
            color: #000;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }

        .container {
            max-width: 900px;
            width: 100%;
        }

        .config-panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 1rem;
            align-items: start;
        }

        .config-panel label {
            color: #aaa;
            font-size: 0.9rem;
        }

        select, textarea {
            width: 100%;
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #fff;
            font-family: inherit;
            font-size: 0.9rem;
        }

        select:focus, textarea:focus {
            outline: none;
            border-color: #76B900;
        }

        textarea {
            resize: vertical;
            min-height: 80px;
        }

        .visualizers {
            display: flex;
            justify-content: center;
            gap: 4rem;
            margin: 2rem 0;
        }

        .visualizer {
            text-align: center;
        }

        .visualizer label {
            display: block;
            margin-bottom: 1rem;
            color: #aaa;
            font-size: 0.9rem;
        }

        canvas {
            border-radius: 50%;
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
        }

        button {
            padding: 1rem 2.5rem;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }

        #startBtn {
            background: #76B900;
            color: #000;
        }

        #startBtn:hover {
            background: #8fd400;
            transform: translateY(-2px);
        }

        #startBtn:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
            transform: none;
        }

        #stopBtn {
            background: #ff4444;
            color: white;
        }

        #stopBtn:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }

        #muteBtn {
            background: #555;
            color: white;
        }

        #muteBtn:hover:not(:disabled) {
            background: #666;
        }

        #muteBtn.muted {
            background: #ff6b6b;
            color: white;
        }

        #muteBtn:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
        }

        .status {
            text-align: center;
            padding: 1rem;
            border-radius: 8px;
            background: rgba(255,255,255,0.05);
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }

        .status.connected {
            border: 2px solid #76B900;
            color: #76B900;
        }

        .status.disconnected {
            border: 2px solid #666;
            color: #888;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 2rem;
        }

        .stat-box {
            background: rgba(255,255,255,0.05);
            padding: 1.25rem;
            border-radius: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            color: #76B900;
            font-weight: bold;
        }

        .stat-label {
            font-size: 0.8rem;
            color: #888;
            margin-top: 0.25rem;
        }

        .transcript {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1.5rem;
            min-height: 4rem;
            max-height: 12rem;
            overflow-y: auto;
            font-size: 0.95rem;
            line-height: 1.5;
            color: #ddd;
        }

        .transcript .meta {
            color: #76B900;
            font-size: 0.8rem;
            display: block;
            margin-bottom: 0.25rem;
        }

        .transcript .notice {
            color: #ffb74d;
            font-size: 0.85rem;
            display: block;
            margin-top: 0.5rem;
        }

        footer {
            margin-top: auto;
            padding-top: 2rem;
            color: #555;
            font-size: 0.8rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>PersonaPlex MLX</h1>
    <p class="subtitle">Live Voice Conversation on Apple Silicon</p>
    <div class="badge">MLX Backend + Voice Conditioning</div>

    <div class="container">
        <div class="config-panel">
            <label for="voice">Voice:</label>
            <select id="voice">
                <optgroup label="Natural Voices">
                    <option value="NATF0" selected>Natural Female 1 (Kendra)</option>
                    <option value="NATF1">Natural Female 2</option>
                    <option value="NATF2">Natural Female 3</option>
                    <option value="NATF3">Natural Female 4</option>
                    <option value="NATM0">Natural Male 1</option>
                    <option value="NATM1">Natural Male 2</option>
                    <option value="NATM2">Natural Male 3</option>
                    <option value="NATM3">Natural Male 4</option>
                </optgroup>
                <optgroup label="Variety Voices">
                    <option value="VARF0">Variety Female 1</option>
                    <option value="VARF1">Variety Female 2</option>
                    <option value="VARF2">Variety Female 3</option>
                    <option value="VARF3">Variety Female 4</option>
                    <option value="VARF4">Variety Female 5</option>
                    <option value="VARM0">Variety Male 1</option>
                    <option value="VARM1">Variety Male 2</option>
                    <option value="VARM2">Variety Male 3</option>
                    <option value="VARM3">Variety Male 4</option>
                    <option value="VARM4">Variety Male 5</option>
                </optgroup>
            </select>

            <label for="prompt">Persona:</label>
            <textarea id="prompt">You are a friendly voice assistant on a call. Answer in one or two short sentences. If you did not catch what the user said, say so and ask them to repeat instead of guessing.</textarea>

            <label></label>
            <button id="applyVoiceBtn" style="padding: 0.5rem 1rem; background: #76B900; color: #000; border: none; border-radius: 6px; cursor: pointer; font-weight: 500;">Apply Voice Settings</button>

            <label for="quant">Quantization:</label>
            <select id="quant">
                <option value="4">4-bit (Fastest)</option>
                <option value="5">5-bit</option>
                <option value="6">6-bit</option>
                <option value="8">8-bit (Balanced)</option>
                <option value="0">None (Best Quality)</option>
            </select>

            <label for="injectText">Silent context:</label>
            <div>
                <textarea id="injectText" placeholder="Facts to give the model without speaking them, e.g. The customer's access code is 7429.
Plain text, one fact per update. It is treated as something the assistant knows, not as something you said."></textarea>
                <button id="injectBtn" style="margin-top: 0.5rem; padding: 0.5rem 1rem; background: #4a90d9; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 500;">Inject into live session</button>
                <button id="resetKeepBtn" style="margin-top: 0.5rem; padding: 0.5rem 1rem; background: #444; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 500;">Reset (keep context)</button>
                <button id="resetDropBtn" style="margin-top: 0.5rem; padding: 0.5rem 1rem; background: #5a2a2a; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 500;">Reset (drop context)</button>
            </div>
        </div>

        <div class="status disconnected" id="status">
            Click Start to begin conversation
        </div>

        <div class="visualizers">
            <div class="visualizer">
                <label>Your Voice</label>
                <canvas id="userCanvas" width="180" height="180"></canvas>
            </div>
            <div class="visualizer">
                <label>PersonaPlex</label>
                <canvas id="aiCanvas" width="180" height="180"></canvas>
            </div>
        </div>

        <div class="controls">
            <button id="startBtn">Start Conversation</button>
            <button id="muteBtn" disabled>🎤 Mute</button>
            <button id="stopBtn" disabled>Stop</button>
        </div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value" id="latency">--</div>
                <div class="stat-label">Latency (ms)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="rtf">--</div>
                <div class="stat-label">Realtime Factor</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="frames">0</div>
                <div class="stat-label">Frames</div>
            </div>
        </div>

        <div class="transcript" id="transcript">
            <span class="meta">Model text stream</span>
            <span id="transcriptText">—</span>
        </div>
    </div>

    <footer>
        Model: NVIDIA PersonaPlex 7B | Backend: MLX + rustymimi | Voice Conditioned
    </footer>

    <script>
        const TARGET_SAMPLE_RATE = 24000;
        const FRAME_SIZE = 1920;

        let ws = null;
        let audioContext = null;
        let playbackContext = null;
        let mediaStream = null;
        let userAnalyser = null;
        let aiAnalyser = null;
        let inputBuffer = [];
        let nativeSampleRate = 48000;

        // Audio playback queue
        let nextPlayTime = 0;
        const BUFFER_AHEAD = 0.15;
        let audioQueue = [];
        let isPlaying = false;
        let isMuted = false;

        const startBtn = document.getElementById('startBtn');
        const muteBtn = document.getElementById('muteBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');
        const voiceSelect = document.getElementById('voice');
        const promptInput = document.getElementById('prompt');
        const applyVoiceBtn = document.getElementById('applyVoiceBtn');
        const userCanvas = document.getElementById('userCanvas');
        const aiCanvas = document.getElementById('aiCanvas');
        const latencyEl = document.getElementById('latency');
        const rtfEl = document.getElementById('rtf');
        const framesEl = document.getElementById('frames');
        const quantSelect = document.getElementById('quant');
        const transcriptEl = document.getElementById('transcript');
        const transcriptText = document.getElementById('transcriptText');
        const injectText = document.getElementById('injectText');
        const injectBtn = document.getElementById('injectBtn');

        let sessionId = null;
        let injectSeq = 0;

        function addNotice(text) {
            const span = document.createElement('span');
            span.className = 'notice';
            span.textContent = text;
            transcriptEl.appendChild(span);
            transcriptEl.scrollTop = transcriptEl.scrollHeight;
        }

        // Send silent context into the running session. The server queues it and
        // applies it at the next pause; it is not spoken aloud.
        injectBtn.addEventListener('click', () => {
            const content = injectText.value.trim();
            if (!content) return;
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                addNotice('not connected');
                return;
            }
            const id = `inj-${++injectSeq}`;
            ws.send(JSON.stringify({ type: 'context.inject', id, content }));
            addNotice(`context ${id} sent (${content.length} chars)`);
            injectText.value = '';
        });

        // Clear the conversation. Keeping context is the default: application
        // facts are restored onto the fresh cache instead of being lost.
        function requestReset(keepContext) {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                addNotice('not connected');
                return;
            }
            ws.send(JSON.stringify({ type: 'session.reset', keep_context: keepContext }));
            addNotice(`session reset requested (${keepContext ? 'keep' : 'drop'} context)`);
        }
        document.getElementById('resetKeepBtn').addEventListener('click', () => requestReset(true));
        document.getElementById('resetDropBtn').addEventListener('click', () => requestReset(false));

        // Handle quantization change
        quantSelect.addEventListener('change', async () => {
            const bits = parseInt(quantSelect.value);
            quantSelect.disabled = true;
            status.textContent = `Reloading model with ${bits > 0 ? bits + '-bit' : 'no'} quantization...`;
            status.className = 'status';

            try {
                const resp = await fetch('/set-quant?bits=' + bits, { method: 'POST' });
                const data = await resp.json();
                if (data.error) {
                    status.textContent = 'Error: ' + data.error;
                } else {
                    status.textContent = `Model reloaded (${bits > 0 ? bits + '-bit' : 'full precision'})`;
                }
            } catch (e) {
                status.textContent = 'Error changing quantization';
            }
            quantSelect.disabled = false;
        });

        // Handle voice/persona change
        applyVoiceBtn.addEventListener('click', async () => {
            const voice = voiceSelect.value;
            const prompt = promptInput.value;
            applyVoiceBtn.disabled = true;
            voiceSelect.disabled = true;
            promptInput.disabled = true;
            status.textContent = `Applying voice: ${voiceSelect.options[voiceSelect.selectedIndex].text}...`;
            status.className = 'status';

            try {
                const resp = await fetch(`/set-voice?voice=${voice}&prompt=${encodeURIComponent(prompt)}`, { method: 'POST' });
                const data = await resp.json();
                if (data.error) {
                    status.textContent = 'Error: ' + data.error;
                } else {
                    status.textContent = `Voice applied: ${voiceSelect.options[voiceSelect.selectedIndex].text}`;
                }
            } catch (e) {
                status.textContent = 'Error changing voice';
            }
            applyVoiceBtn.disabled = false;
            voiceSelect.disabled = false;
            promptInput.disabled = false;
        });

        // Downsample the mic to 24 kHz for Mimi.
        //
        // This MUST low-pass first. Plain decimation folds everything above
        // 6 kHz back into the speech band, and on real speech that changes ~90%
        // of Mimi's tokens: the model then hears noise instead of words, so it
        // answers fluently but to nothing ("responding but not listening").
        // A 63-tap windowed-sinc FIR cuts the band before we decimate, and the
        // filter/position state persists across callbacks so there are no clicks
        // at chunk boundaries.
        function makeResampler(fromRate, toRate) {
            if (fromRate === toRate) return (input) => input;

            const ratio = fromRate / toRate;
            const cutoff = 0.45 * Math.min(fromRate, toRate);
            const N = 63;
            const M = (N - 1) / 2;
            const taps = new Float32Array(N);
            let sum = 0;
            for (let i = 0; i < N; i++) {
                const n = i - M;
                const fc = 2 * cutoff / fromRate;
                const sinc = n === 0 ? fc : Math.sin(Math.PI * fc * n) / (Math.PI * n);
                const win = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (N - 1));
                taps[i] = sinc * win;
                sum += taps[i];
            }
            for (let i = 0; i < N; i++) taps[i] /= sum;

            let history = new Float32Array(N - 1);
            let carry = new Float32Array(0);
            let pos = 0;

            function lowpass(input) {
                const pad = new Float32Array(N - 1 + input.length);
                pad.set(history, 0);
                pad.set(input, N - 1);
                const out = new Float32Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    let acc = 0;
                    for (let k = 0; k < N; k++) acc += pad[i + k] * taps[k];
                    out[i] = acc;
                }
                history = pad.slice(pad.length - (N - 1));
                return out;
            }

            return function (input) {
                const filtered = lowpass(input);
                const buf = new Float32Array(carry.length + filtered.length);
                buf.set(carry, 0);
                buf.set(filtered, carry.length);

                const out = [];
                let p = pos;
                while (p + 1 < buf.length) {
                    const i0 = Math.floor(p);
                    const frac = p - i0;
                    out.push(buf[i0] + frac * (buf[i0 + 1] - buf[i0]));
                    p += ratio;
                }
                const consumed = Math.floor(p);
                carry = buf.slice(consumed);
                pos = p - consumed;
                return Float32Array.from(out);
            };
        }

        function drawVisualizer(canvas, analyser, color) {
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            const centerX = width / 2;
            const centerY = height / 2;
            const maxRadius = width * 0.42;

            function draw() {
                requestAnimationFrame(draw);

                let intensity = 0;
                if (analyser) {
                    const dataArray = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(dataArray);
                    intensity = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255;
                }

                const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, maxRadius);
                gradient.addColorStop(0, '#1a1a2e');
                gradient.addColorStop(1, '#0a0a0a');
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, width, height);

                ctx.beginPath();
                ctx.arc(centerX, centerY, maxRadius, 0, Math.PI * 2);
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 3;
                ctx.stroke();

                const radius = maxRadius * (0.25 + 0.6 * intensity);
                const circleGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
                circleGradient.addColorStop(0, '#9acd32');
                circleGradient.addColorStop(1, color);
                ctx.beginPath();
                ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
                ctx.fillStyle = circleGradient;
                ctx.fill();

                ctx.beginPath();
                ctx.arc(centerX, centerY, maxRadius * 0.12, 0, Math.PI * 2);
                ctx.fillStyle = '#76B900';
                ctx.fill();
            }
            draw();
        }

        async function start() {
            try {
                status.textContent = 'Connecting...';
                status.className = 'status';

                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                    }
                });

                // Ask the browser for a 24 kHz context so its own (properly
                // band-limited) resampler converts the mic stream for us.
                // resampler() below is the fallback for browsers/devices that
                // ignore the requested rate.
                let gotTarget = true;
                try {
                    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
                    gotTarget = audioContext.sampleRate === TARGET_SAMPLE_RATE;
                } catch (e) {
                    gotTarget = false;
                }
                if (!gotTarget) {
                    if (audioContext) audioContext.close();
                    audioContext = new AudioContext();
                }
                nativeSampleRate = audioContext.sampleRate;
                if (nativeSampleRate !== TARGET_SAMPLE_RATE) {
                    console.warn(
                        `Mic running at ${nativeSampleRate} Hz; resampling to ` +
                        `${TARGET_SAMPLE_RATE} Hz with the in-page FIR.`
                    );
                }
                const resampler = makeResampler(nativeSampleRate, TARGET_SAMPLE_RATE);

                playbackContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });

                userAnalyser = audioContext.createAnalyser();
                userAnalyser.fftSize = 256;

                aiAnalyser = playbackContext.createAnalyser();
                aiAnalyser.fftSize = 256;

                const source = audioContext.createMediaStreamSource(mediaStream);
                source.connect(userAnalyser);

                // 2048 @ 24 kHz = 85 ms, i.e. roughly one 80 ms frame per
                // callback. (4096 would hand the socket two frames at a time.)
                const bufferSize = 2048;
                const scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);
                source.connect(scriptProcessor);
                // Don't connect to destination - that causes echo!
                // scriptProcessor needs a destination to work, use a silent gain node
                const silentGain = audioContext.createGain();
                silentGain.gain.value = 0;
                scriptProcessor.connect(silentGain);
                silentGain.connect(audioContext.destination);

                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    status.textContent = '● Connected - Speak now!';
                    status.className = 'status connected';
                    startBtn.disabled = true;
                    muteBtn.disabled = false;
                    stopBtn.disabled = false;
                };

                ws.onmessage = (event) => {
                    if (typeof event.data === 'string') {
                        let msg;
                        try { msg = JSON.parse(event.data); } catch (e) { return; }

                        if (msg.type === 'stats') {
                            latencyEl.textContent = msg.avg_latency_ms;
                            rtfEl.textContent = msg.rtf + 'x';
                            framesEl.textContent = msg.frames;
                        } else if (msg.type === 'generation.text_delta') {
                            // The model's own transcript. Text only: the speech
                            // arrives as binary audio frames below.
                            if (transcriptText.textContent === '—') transcriptText.textContent = '';
                            transcriptText.textContent += msg.text;
                            transcriptEl.scrollTop = transcriptEl.scrollHeight;
                        } else if (msg.type === 'session.started') {
                            sessionId = msg.session_id;
                            addNotice(`session ${msg.session_id} (voice ${msg.voice})`);
                        } else if (msg.type === 'context.accepted') {
                            addNotice(`context ${msg.id || ''} accepted — waiting for a pause`);
                        } else if (msg.type === 'context.applied') {
                            const how = msg.forced ? ', forced without a pause' : '';
                            addNotice(`context ${msg.id || ''} applied (${msg.tokens} tokens, cache offset ${msg.offset}${how})`);
                        } else if (msg.type === 'context.rejected') {
                            addNotice(`context ${msg.id || ''} rejected: ${msg.reason}`);
                        } else if (msg.type === 'session.reset.queued') {
                            addNotice(`session reset requested (${msg.keep_context ? 'keeping' : 'dropping'} injected context)`);
                        } else if (msg.type === 'context.restored') {
                            addNotice(`context restored: ${msg.facts} fact(s) replayed`);
                        } else if (msg.type === 'session.reset') {
                            addNotice('session reset');
                            transcriptText.textContent = '—';
                        }
                    } else {
                        const audioData = new Float32Array(event.data);
                        playAudio(audioData);
                    }
                };

                ws.onclose = () => {
                    status.textContent = '○ Disconnected';
                    status.className = 'status disconnected';
                    startBtn.disabled = false;
                    muteBtn.disabled = true;
                    muteBtn.classList.remove('muted');
                    muteBtn.textContent = '🎤 Mute';
                    isMuted = false;
                    stopBtn.disabled = true;
                };

                ws.onerror = (err) => {
                    console.error('WebSocket error:', err);
                    status.textContent = '✗ Connection error';
                };

                scriptProcessor.onaudioprocess = (e) => {
                    if (!ws || ws.readyState !== WebSocket.OPEN) return;

                    const inputData = e.inputBuffer.getChannelData(0);
                    const resampled = resampler(inputData);
                    inputBuffer.push(...resampled);

                    while (inputBuffer.length >= FRAME_SIZE) {
                        let frame = new Float32Array(inputBuffer.splice(0, FRAME_SIZE));
                        // Send silence when muted
                        if (isMuted) {
                            frame = new Float32Array(FRAME_SIZE);
                        }
                        ws.send(frame.buffer);
                    }
                };

                drawVisualizer(userCanvas, userAnalyser, '#4E8800');
                drawVisualizer(aiCanvas, aiAnalyser, '#4E8800');

            } catch (err) {
                console.error('Error starting:', err);
                status.textContent = '✗ Error: ' + err.message;
            }
        }

        function playAudio(audioData) {
            if (!playbackContext) return;

            audioQueue.push(audioData);

            if (!isPlaying && audioQueue.length >= 3) {
                isPlaying = true;
                schedulePlayback();
            } else if (isPlaying) {
                schedulePlayback();
            }
        }

        function schedulePlayback() {
            if (!playbackContext || audioQueue.length === 0) return;

            const currentTime = playbackContext.currentTime;

            if (nextPlayTime < currentTime) {
                nextPlayTime = currentTime + BUFFER_AHEAD;
            }

            while (audioQueue.length > 0) {
                const audioData = audioQueue.shift();

                const buffer = playbackContext.createBuffer(1, audioData.length, TARGET_SAMPLE_RATE);
                buffer.getChannelData(0).set(audioData);

                const source = playbackContext.createBufferSource();
                source.buffer = buffer;
                source.connect(aiAnalyser);
                aiAnalyser.connect(playbackContext.destination);
                source.start(nextPlayTime);

                nextPlayTime += buffer.duration;
            }
        }

        function stop() {
            if (ws) {
                ws.close();
                ws = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (playbackContext) {
                playbackContext.close();
                playbackContext = null;
            }
            inputBuffer = [];
            audioQueue = [];
            isPlaying = false;
            nextPlayTime = 0;

            status.textContent = '○ Disconnected';
            status.className = 'status disconnected';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        startBtn.addEventListener('click', start);
        muteBtn.addEventListener('click', () => {
            isMuted = !isMuted;
            if (isMuted) {
                muteBtn.textContent = '🔇 Unmute';
                muteBtn.classList.add('muted');
            } else {
                muteBtn.textContent = '🎤 Mute';
                muteBtn.classList.remove('muted');
            }
        });
        stopBtn.addEventListener('click', stop);

        drawVisualizer(userCanvas, null, '#4E8800');
        drawVisualizer(aiCanvas, null, '#4E8800');

        // Load current quant level on page load
        fetch('/current-quant').then(r => r.json()).then(data => {
            quantSelect.value = data.bits.toString();
        });
    </script>
</body>
</html>
"""


def main():
    """Console entry point for `uv run serve` / `personaplex`."""
    print("=" * 50)
    print("PersonaPlex MLX Live Server (Voice Conditioned)")
    print("=" * 50)
    print("Open http://localhost:8000 in your browser")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
