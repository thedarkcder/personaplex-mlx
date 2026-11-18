# PersonaPlex MLX

Real-time voice conversation AI running on Apple Silicon. A port of NVIDIA's PersonaPlex 7B to Apple's MLX framework.

![PersonaPlex MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-black?style=flat-square&logo=apple)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Features

- **Real-time voice conversation** - Full-duplex audio streaming at 24kHz
- **Voice conditioning** - 18 different voice personas (natural & variety)
- **Customizable personas** - Set system prompts for different AI personalities
- **Dynamic quantization** - Switch between 4/5/6/8-bit on the fly
- **Web UI** - Clean browser interface with audio visualization
- **~1.2x realtime** - Runs faster than realtime on M1/M2/M3 Macs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PersonaPlex MLX Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Browser (WebSocket)                                             │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              live_server_mlx.py                          │    │
│  │  FastAPI + WebSocket server                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Mimi Encode │───▶│   LM (7B)   │───▶│ Mimi Decode │         │
│  │   ~19ms     │    │   ~26ms     │    │   ~19ms     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                                     │
│                            ▼                                     │
│                     ┌─────────────┐                             │
│                     │  DepFormer  │                             │
│                     │  16 slices  │                             │
│                     │   ~12ms     │                             │
│                     └─────────────┘                             │
│                                                                  │
│  Total: ~65ms per 80ms frame = 1.23x realtime                   │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- ~8GB RAM (4-bit quantization)
- ~16GB disk for weights

## Installation

### 1. Clone the repository

```bash
git clone --recursive https://github.com/Acelogic/personaplex-mlx.git
cd personaplex-mlx
```

> **Note:** The `--recursive` flag clones the [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) submodule (PyTorch reference code).

### 2. Install dependencies with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv sync
```

### 3. Download weights

Download the PersonaPlex weights from HuggingFace:

```bash
uv run download-weights
```

Or manually download from [HuggingFace](https://huggingface.co/nvidia/PersonaPlex-7B) and place in `weights/`.

Required files:
- `weights/model.safetensors` (~16GB)
- `weights/tokenizer-e351c8d8-checkpoint125.safetensors` (~385MB)
- `weights/tokenizer_spm_32k_3.model` (~553KB)
- `weights/voices_cpu/*.pt` (voice prompts)

## Usage

### Start the server

```bash
uv run serve
```

Open http://localhost:8000 in your browser.

### Web UI Controls

- **Voice** - Select from 18 voice personas
- **Persona** - Customize the AI's system prompt
- **Quantization** - Switch between 4/5/6/8-bit (or full precision)
- **Silent context** - Inject facts into the running conversation without the model reading them aloud
- **Start/Stop** - Begin/end conversation
- **Mute** - Mute your microphone

## Live Session Protocol

The websocket at `/ws` carries two things: **binary frames** (audio) and
**JSON text frames** (control and events).

### Client → server

| Message | Description |
|---------|-------------|
| binary, 1920 × float32 | One 80 ms microphone frame at 24 kHz |
| `{"type":"context.inject","id":"...","content":"..."}` | Add facts to the live conversation without speaking them |
| `{"type":"session.reset","keep_context":true}` | Clear conversational state and restore the voice conditioning. `keep_context` defaults to `true`: injected facts are replayed onto the fresh cache (an application's context is not conversational state). Send `false` to drop them. |
| `{"type":"session.start"}` | Ask for session metadata |

### Server → client

| Message | Description |
|---------|-------------|
| binary, 1920 × float32 | `generation.audio_delta` — one 80 ms frame of speech |
| `{"type":"session.started", ...}` | Session id, voice, sample rate, frame size |
| `{"type":"generation.text_delta","seq":n,"frame":n,"token_id":n,"text":"..."}` | The model's own text stream, one token at a time |
| `{"type":"context.accepted","id":"...","tokens":n,"framed":"..."}` | Update queued; `framed` is the exact text put in the cache. **Not** the same as understood |
| `{"type":"context.applied","id":"...","offset":n,"latency_ms":n,"waited_ms":n,"forced":bool}` | Update is in the attention cache. `forced` means it could not wait for a pause |
| `{"type":"context.rejected","id":"...","reason":"..."}` | Update refused (empty, too long, unknown type) |
| `{"type":"session.reset.queued","keep_context":true}` | Reset accepted, waiting for a pause |
| `{"type":"session.reset","offset":n,"keep_context":true}` | Reset applied; conversation cleared |
| `{"type":"context.restored","facts":n,"offset":n}` | Injected facts replayed after a reset |
| `{"type":"stats", ...}` | Frames, average LM latency, realtime factor |

Both the persona prompt and injected content are wrapped in `[INST] … [/INST]`.
The reference server uses `<system> … <system>` instead, but with these
MLX-converted weights that form silences the model (observed: an entire session
at `out_avg ≈ 0.0002`), so it is opt-in via `PERSONAPLEX_PROMPT_FORMAT=system`.

**Injected content is plain text, one fact per update.** The model reads it in
its own text stream, so it treats it as something it knows, not as a question to
answer. Keep each update atomic:

| Injected content | Reply |
|---|---|
| `The customer's access code is 7429.` | *"Your access code is 7429."* (3/3 runs) |
| `The user's dog is a golden retriever called Biscuit.` | *"Oh, he's a golden retriever, huh? Loves to chew on bones"* — the fact was used, but the reply picked the wrong field |

One fact per update is the rule. When a caller needs reliable retrieval of a
specific field out of a multi-field sentence, it can include the question it
expects — `Question: what is the dog called? Answer: Biscuit.` — but that is the
caller's wording choice, not the protocol: the server passes text through
unchanged, and it is never required.

**Do not put provenance in the injected text.** Text that names its own source
(*"system note, not spoken by the user"*, *"source: system"*) does not fix
attribution and makes it worse: the model reads the phrase aloud — *"7429, source
system, not user"*. `PERSONAPLEX_INJECT_PREFIX` stays available for experiments
but defaults to empty. Mid-conversation the cache has no role channel at all, so
*"did I tell you that or did the system?"* gets a deflection or a wrong answer
however the text is phrased. Treat `context.applied` as the authoritative record
of what the application supplied; the model's self-report is not an audit trail.
(The one thing measured to change that answer was an injected question/answer
provenance pair, which is application content, not a server feature.)

`context.inject` is only applied when both streams have been quiet for a few
frames: injection advances the attention cache, so it must not run mid-utterance.
Queued updates therefore apply at the next conversational pause, which can be a
second or more later — watch for `context.applied`. If the model does not stop
talking, the update is applied anyway after `PERSONAPLEX_INJECT_MAX_WAIT`
(default 4 s) and reported with `forced: true`, so a looping model cannot hold
an update hostage forever. The deadline is also honoured when the client stops
sending audio: the loop is frame-driven, so a stalled mic stream must not freeze
a queued update indefinitely.

Limits: `PERSONAPLEX_MAX_INJECT_TOKENS` (default 64 tokens ≈ 5.1 s of context;
two or three short facts fit).
An update never produces speech of its own: after one is applied the server
discards the next `PERSONAPLEX_INJECT_MUTE_FRAMES` (default 12 ≈ 1 s) of
generated frames and streams silence instead. Without that window the model
often continues the injected sentence as if it had said it — it will speak the
fact aloud. Verify understanding by asking the model about the update
afterwards. Restored facts are capped by
`PERSONAPLEX_MAX_CONTEXT_TOKENS` (default 160); the oldest are dropped from the
replay set first.

`session.reset` does **not** wait for a pause: it replaces the cache outright,
so it is applied on the next frame even mid-sentence. (An earlier version waited
for silence, which deadlocked when the model looped and never went quiet.)
It keeps injected facts by default and
replays them onto the fresh cache, emitting `context.restored` with the count, so
an application can clear the conversation without losing the context it supplied.
Use `{"type":"session.reset","keep_context":false}` to forget them as well.
Changing voice or quantization is refused while a session is live: both rewrite
the shared KV cache, which would corrupt an active conversation.

## Record tool-calling examples

Open [Audio labeler](http://127.0.0.1:8000/training/) from the voice runner on
the same Mac. To open it without starting PersonaPlex, run
`.venv/bin/python tools/audio_labeler.py` and visit the
[standalone labeler](http://127.0.0.1:8765/).

1. Use **Space** or **Record** to start recording, then press **Space** or
   **Stop recording** to finish. Once the clip is ready, **Space** plays or
   pauses it. Choose **Record again** to replace the take. Select a marker
   before clicking the waveform. **Ready** marks when the required arguments
   are grounded in the audio. **Turn end** starts at the clip end and can be
   moved earlier to exclude trailing silence.
2. Enter the transcript and select the expected outcome. For an incomplete
   request, choose **Missing or unclear argument** and mark when the tool intent
   becomes clear. For a complete call, enter the expected arguments as JSON.
3. Select **Save example**. The WAV and its annotation are written under
   `tool_data/`, with clips in `audio/`, immutable schema snapshots in
   `catalogs/`, and one row per clip in `manifest.jsonl`.

The example catalog is available for practice. **Import JSON** accepts a list of
tools, an object with a `tools` array, or the JSON-RPC response from MCP
`tools/list`. Each tool definition needs a unique `name` and an object
`inputSchema`. For native tools, export the same fields from your app. The labeler
uses these definitions for annotation.

## Scripts

| Command | Description |
|---------|-------------|
| `uv run serve` | Start the web server |
| `uv run personaplex` | Same server, via the package entry point |
| `python tools/test_kv_pinning.py` | KV prompt-pinning tests (no model needed) |
| `python tools/voice_probe.py --inject ... --question ... [--reset keep\|drop] [--pad s]` | End-to-end check: inject a fact, optionally reset or push past the window, then ask about it out loud (uses macOS `say`) |
| `python tools/serve_side_port.py` | Run a second server on `PERSONAPLEX_PORT` for testing |
| `tools/pinning_ab.sh` | Pinned vs `PERSONAPLEX_PIN_PROMPT=0`: does the prompt survive a conversation longer than the window? |

Weights are downloaded from HuggingFace (`nvidia/personaplex-7b-v1`) as described
under Installation; there is no `download-weights` script.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Readiness probe: `200` when the model is loaded and voice-conditioned, `503` while starting. Cheap enough to poll during a live session; `?verify=1` adds the KV integrity check (not cheap) |
| `/ws` | WebSocket | Real-time audio streaming |
| `/voices` | GET | List available voices |
| `/v1/models` | GET | OpenAI-compatible model listing (compatibility stub; there is no `/v1/chat/completions` or `/v1/realtime`) |
| `/set-voice` | POST | Change voice and persona |
| `/set-quant` | POST | Change quantization level |
| `/current-quant` | GET | Get current quantization |

## Context window and prompt pinning

The model attends to a fixed window (`PERSONAPLEX_LM_CONTEXT`, default 3000
frames = 4 minutes at 12.5 fps). The voice + persona prompt is **pinned** in that
window for the whole session: the ring wraps around it, so the voice and role do
not drift on long calls. Only the conversation slides.

Injected facts are **not** pinned — they are written after the prompt and slide
with the conversation, so a fact lives for the rest of the window (~4 minutes
from injection by default). If a call needs a fact to outlive that, re-inject it
or start the session with more context.

This mirrors NVIDIA's `pin_kv_prefix` / attention-sink work (personaplex commit
`16419a2`) and is implemented in `kv_pinning.py` for the MLX caches. On startup
you should see:

```
Pinned 101 system prompt frames (window 3000 frames, sliding 2899)
```

Two details matter and are covered by `tools/test_kv_pinning.py`:

- the KV ring is sized to the *attention window*, not the RoPE limit, so the
  cache never hands attention stale frames from a previous cycle;
- pinned frames keep the RoPE positions they were encoded with, so they stay
  where they were while the ring advances around them.

### Checking it on a live server

`GET /health?verify=1` compares the prompt prefix in the live cache against a
snapshot taken at conditioning time. It is opt-in because it evaluates GPU
arrays and can stall a realtime loop; plain `/health` stays cheap.

```json
{"prompt_frames": 101, "pinned_frames": 101, "window_frames": 3000,
 "sliding_frames": 2899, "cache_offset": 512, "prompt_intact": true}
```

`PERSONAPLEX_PIN_PROMPT=0` turns pinning off but still takes the snapshot, so the
same check reports the difference. `tools/pinning_ab.sh` runs the measurement
end to end with a 200-frame window: after 60 s of conversation (300 frames) the
pinned server reports `prompt_intact: true`, the unpinned one `false`. That is
the whole claim — the prompt stays in the window — measured rather than inferred
from voice or persona behaviour, which is too noisy to be a test.

## Performance

Tested on Apple Silicon:

| Quantization | Model Time | Total Latency | RTF |
|--------------|------------|---------------|-----|
| 4-bit | ~26ms | ~65ms | 1.23x |
| 5-bit | ~30ms | ~70ms | 1.14x |
| 6-bit | ~35ms | ~75ms | 1.07x |
| 8-bit | ~45ms | ~85ms | 0.94x |
| None | ~80ms | ~120ms | 0.67x |

## Project Structure

```
personaplex-mlx/
├── live_server_mlx.py      # Main server (FastAPI + WebSocket)
├── kv_pinning.py           # Pins the system prompt in the KV window
├── convert_weights.py      # Weight conversion utility
├── test_pytorch_mlx_comparison.py  # Verification tests
├── pyproject.toml          # Dependencies and scripts
├── tools/                  # Probe, side-port runner, pinning tests/A-B
├── weights/                # Model weights (not in git)
│   ├── model.safetensors
│   ├── tokenizer*.safetensors
│   └── voices_cpu/*.pt
├── personaplex/            # Git submodule → NVIDIA/personaplex
└── personaplex_mlx/        # MLX model config
```

## Dependencies

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [moshi-mlx](https://github.com/kyutai-labs/moshi) - Base model implementation
- [rustymimi](https://github.com/kyutai-labs/moshi) - Neural audio codec
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## References

- [PersonaPlex Paper](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
- [PersonaPlex Model](https://huggingface.co/nvidia/personaplex-7b-v1)
- [Moshi/Mimi](https://github.com/kyutai-labs/moshi)

## License

MIT License - See [LICENSE](LICENSE) for details.

---

Built with MLX on Apple Silicon
