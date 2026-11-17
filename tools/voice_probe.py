"""Voice probe: inject a fact silently, then ask about it out loud.

Synthesises speech with macOS `say`, feeds it as mic frames at realtime pace, and
collects the model's text stream. This is the only honest way to test whether an
injected fact is *understood* rather than merely accepted.

The mic stream must never stop: the server's LM loop is driven by incoming
frames, so a client that goes quiet also stops the session from making progress
(including applying queued context).

Usage:
    python tools/voice_probe.py --inject "..." --question "What is my dog called?"
"""
import argparse, asyncio, json, subprocess, tempfile, time
from pathlib import Path

import numpy as np
import websockets

FRAME = 1920
RATE = 24000
DEFAULT_TICK = 0.08


def synth(text: str) -> np.ndarray:
    """macOS say -> mono float32 @ 24 kHz."""
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "q.aiff"
        raw = Path(td) / "q.raw"
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
             "-ar", str(RATE), "-ac", "1", "-f", "f32le", str(raw)],
            check=True,
        )
        return np.frombuffer(raw.read_bytes(), dtype=np.float32)


def to_frames(audio: np.ndarray):
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        yield audio[i:i + FRAME].astype(np.float32)


def silent(n_frames: int):
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(n_frames)]


class Pacer:
    """Sends one frame every 80 ms, so the server sees realtime microphone."""

    def __init__(self, ws, tick):
        self.ws = ws
        self.tick = tick
        self.q = asyncio.Queue(maxsize=2)   # bounded: feed() must pace with the sender
        self.sent = 0
        self.task = None

    async def _run(self):
        while True:
            frame = await self.q.get()
            if frame is None:
                return
            await self.ws.send(frame.tobytes())
            self.sent += 1
            await asyncio.sleep(self.tick)

    def start(self):
        self.task = asyncio.create_task(self._run())

    async def feed(self, frames, label=""):
        n = 0
        for f in frames:
            await self.q.put(f)   # backpressure == realtime pacing
            n += 1
        if label:
            print(f"  queued {n} frames of {label}", flush=True)

    def keepalive(self, n_frames: int):
        for f in silent(n_frames):
            try:
                self.q.put_nowait(f)
            except asyncio.QueueFull:
                pass

    async def stop(self):
        await self.q.put(None)
        if self.task:
            await self.task


class Probe:
    def __init__(self, url, tick, reset_settle, start_quiet=2.5):
        self.url = url
        self.tick = tick
        self.reset_settle = reset_settle
        # The greet is spoken on connect. Injecting or asking before it finishes
        # splices the question into the model's own unfinished sentence, which
        # reads as "not listening" but is really overlapping turns.
        self.start_quiet = start_quiet
        self.text = []
        self.events = []
        self.last_audio_rms = 0.0
        self.audio_frames = 0
        # Set when context.applied arrives: peak model output in the second
        # after it is what proves an injection was never spoken.
        self.inject_at = None
        self.post_inject_peak = 0.0
        self.MUTE_PROBE_WINDOW = 1.2

    async def recv_loop(self, ws):
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    pcm = np.frombuffer(msg, dtype=np.float32)
                    self.last_audio_rms = float(np.sqrt(np.mean(pcm ** 2))) if len(pcm) else 0.0
                    self.audio_frames += 1
                    if (self.inject_at is not None
                            and time.time() - self.inject_at <= self.MUTE_PROBE_WINDOW):
                        self.post_inject_peak = max(self.post_inject_peak, self.last_audio_rms)
                else:
                    ev = json.loads(msg)
                    self.events.append(ev)
                    if ev.get("type") == "generation.text_delta":
                        self.text.append(ev["text"])
                    elif ev.get("type") == "context.applied":
                        self.inject_at = time.time()
                        self.post_inject_peak = 0.0
                        detail = {k: v for k, v in ev.items() if k != "type"}
                        print(f"  [{ev.get('type')}] {detail}", flush=True)
                    elif ev.get("type") != "stats":
                        detail = {k: v for k, v in ev.items() if k != "type"}
                        print(f"  [{ev.get('type')}] {detail}", flush=True)
        except Exception as e:
            print(f"recv ended: {e}", flush=True)

    def saw(self, kind):
        return any(e.get("type") == kind for e in self.events)

    def count(self, kind):
        return sum(1 for e in self.events if e.get("type") == kind)

    async def wait_quiet(self, pacer, seconds):
        """Keep the mic stream alive until the model has been silent for a while."""
        need = max(1, int(seconds / self.tick))
        run = 0
        for _ in range(400):
            pacer.keepalive(1)
            await asyncio.sleep(self.tick)
            if self.last_audio_rms < 0.01:
                run += 1
                if run >= need:
                    return True
            else:
                run = 0
        return False

    async def run(self, inject, questions, pre_quiet, post_quiet,
                  reset_mode=None, pad=0.0):
        self.pad = pad
        transcript = ""
        async with websockets.connect(self.url, max_size=None) as ws:
            rx = asyncio.create_task(self.recv_loop(ws))
            pacer = Pacer(ws, self.tick)
            pacer.start()
            await ws.send(json.dumps({"type": "session.start"}))
            print("waiting for the greeting to finish...", flush=True)
            if not await self.wait_quiet(pacer, self.start_quiet):
                print(f"  !! model never went quiet for {self.start_quiet}s; "
                      f"results below may be two overlapping turns", flush=True)

            for idx, fact in enumerate(inject or []):
                # The server adds the <system> ... <system> tags itself.
                payload = fact
                print(f"injecting: {payload!r}", flush=True)
                applied_before = sum(1 for e in self.events if e.get("type") == "context.applied")
                await ws.send(json.dumps({"type": "context.inject", "id": f"probe-{idx+1}",
                                          "content": payload}))
                for _ in range(150):
                    applied_now = sum(1 for e in self.events if e.get("type") == "context.applied")
                    if applied_now > applied_before or self.saw("context.rejected"):
                        break
                    pacer.keepalive(1)
                    await asyncio.sleep(self.tick)
                if not sum(1 for e in self.events if e.get("type") == "context.applied") > applied_before:
                    print("  !! injection was never applied", flush=True)
                # Let the model sit with the new context, still streaming silence.
                print(f"  settling for {pre_quiet}s of silence...", flush=True)
                await self.wait_quiet(pacer, pre_quiet)
                print(f"  audio frames received since connect: {self.audio_frames}", flush=True)
                if self.inject_at is not None:
                    verdict = ("silent" if self.post_inject_peak < 0.01
                               else "SPOKEN - mute window failed")
                    print(f"  model output peak in the {self.MUTE_PROBE_WINDOW}s after "
                          f"injection: {self.post_inject_peak:.4f} -> {verdict}",
                          flush=True)

            if reset_mode:
                keep = reset_mode == "keep"
                print(f"resetting session (keep_context={keep})...", flush=True)
                before = self.count("session.reset")
                await ws.send(json.dumps({"type": "session.reset", "keep_context": keep}))
                for _ in range(400):
                    if self.count("session.reset") > before:
                        break
                    pacer.keepalive(1)
                    await asyncio.sleep(self.tick)
                if self.count("session.reset") == before:
                    print("  !! reset was never applied", flush=True)
                restored = [e for e in self.events if e.get("type") == "context.restored"]
                if restored:
                    print(f"  restored {restored[-1].get('facts')} fact(s), "
                          f"offset {restored[-1].get('offset')}", flush=True)
                else:
                    print("  no context.restored event", flush=True)
                # A reset makes the model speak again (greeting/filler). Let that
                # finish before asking, or the question overlaps it.
                print(f"  settling {self.reset_settle}s after reset...", flush=True)
                await pacer.feed(silent(int(self.reset_settle / self.tick)),
                                 "post-reset silence")
                await self.wait_quiet(pacer, pre_quiet)

            if self.pad > 0:
                # Burn real conversation frames: used to push past the sliding
                # window so pinning (or its absence) becomes observable.
                print(f"padding {self.pad}s of silence "
                      f"(~{int(self.pad / self.tick)} frames)...", flush=True)
                await pacer.feed(silent(int(self.pad / self.tick)), "pad silence")

            for q in questions:
                self.text.clear()
                self.audio_frames = 0
                print(f"asking: {q!r}", flush=True)
                await pacer.feed(to_frames(synth(q)), "question")
                await pacer.feed(silent(int(post_quiet / self.tick)), "post-silence")
                await asyncio.sleep(1.0)
                reply = "".join(self.text).strip()
                print(f"\n--- reply to {q!r} ---\n{reply or '(nothing)'}\n", flush=True)
                transcript += reply + "\n"

            rx.cancel()
            await pacer.stop()
            transcript = transcript.strip()
            print("\n=== MODEL TEXT STREAM ===", flush=True)
            print(transcript if transcript else "(nothing)", flush=True)
            return transcript


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8123/ws")
    ap.add_argument("--inject", action="append", default=None,
                    help="fact to inject silently; repeat for multiple updates")
    ap.add_argument("--question", action="append", required=True,
                    help="question to speak; repeat for a follow-up in the same session")
    ap.add_argument("--pre-quiet", type=float, default=1.5)
    ap.add_argument("--post-quiet", type=float, default=6.0)
    ap.add_argument("--reset", choices=["keep", "drop"], default=None,
                    help="send session.reset after injecting: keep or drop the injected facts")
    ap.add_argument("--reset-settle", type=float, default=3.5,
                    help="seconds of silence after a reset, so its greeting finishes")
    ap.add_argument("--start-quiet", type=float, default=2.5,
                    help="seconds of silence to wait for after connecting, so the "
                         "greeting has finished before anything is injected or asked")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="seconds of silence to stream before asking, to push the "
                         "session past the model's sliding context window")
    ap.add_argument("--frame-ms", type=float, default=DEFAULT_TICK * 1000,
                    help="frame pacing in ms; raise it if frames get dropped")
    a = ap.parse_args()
    await Probe(a.url, a.frame_ms / 1000, a.reset_settle, a.start_quiet).run(
        a.inject, a.question, a.pre_quiet, a.post_quiet, a.reset, a.pad)

if __name__ == "__main__":
    asyncio.run(main())
