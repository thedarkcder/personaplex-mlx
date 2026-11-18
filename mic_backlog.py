"""Bounded, time-sized backlog for incoming mic frames.

A mic frame is 80 ms of audio, but one LM step is ~26 ms at 4-bit, so the
pipeline has slack: if it stalls for a moment it can replay the frames that
arrived meanwhile and still stay ahead of realtime. That is exactly what an
applied ``context.inject`` is - ~0.7-1 s of cache work with no audio produced -
and the old 4-frame queue (320 ms) turned every update into dropped speech,
which the model hears as chopped audio and answers with something unrelated.

So the backlog is sized in *seconds* of audio, and overflow keeps the most
recent speech (drop-oldest): when a stall really is longer than the budget, the
newest audio is the part worth keeping.
"""
from __future__ import annotations

import os
import queue

# How much mic audio may be buffered while the LM is busy. 2.5 s covers a full
# injection plus the pause detection that has to precede it.
DEFAULT_BACKLOG_S = float(os.environ.get("PERSONAPLEX_MIC_BACKLOG_S", "2.5"))


def frames_for(seconds: float, frame_rate: float) -> int:
    """Frames in ``seconds`` of audio, never fewer than a couple of frames."""
    return max(2, int(round(seconds * frame_rate)))


class BoundedBacklog:
    """Thread-safe FIFO that drops its oldest item when full.

    ``dropped`` counts frames actually lost (overflow), which is the number that
    matters: anything else was buffered and replayed. ``peak`` is the deepest the
    queue got, i.e. how much of a stall the backlog absorbed.
    """

    def __init__(self, max_frames: int, frame_rate: float = 12.5, name: str = "mic"):
        self.max_frames = max_frames
        self.frame_rate = frame_rate
        self.name = name
        self.dropped = 0
        self.peak = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_frames)

    def put(self, item) -> None:
        """Enqueue, dropping the oldest item if the backlog is full."""
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:  # consumer raced us; count it, never block
                self.dropped += 1
        self.peak = max(self.peak, self._q.qsize())

    def put_nowait(self, item) -> None:
        self.put(item)

    def get(self):
        return self._q.get()

    def get_nowait(self):
        return self._q.get_nowait()

    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def seconds(self) -> float:
        return self.max_frames / self.frame_rate
