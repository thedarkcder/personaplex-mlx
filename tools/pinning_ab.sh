#!/usr/bin/env bash
# End-to-end check that the system prompt survives a conversation longer than
# the attention window. Run with the server up (PERSONAPLEX_LM_CONTEXT=200).
#
#   tools/pinning_ab.sh            # pinned (default)
#   PERSONAPLEX_PIN_PROMPT=0 tools/pinning_ab.sh   # control
#
# Prints the cache geometry before and after a 60s conversation, which is
# 300 frames > the 200-frame window.
set -u
cd "$(dirname "$0")/.."

PORT="${PERSONAPLEX_PORT:-8123}"
BASE="http://127.0.0.1:${PORT}"

for _ in $(seq 1 40); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" = "200" ] && break
    sleep 5
done

show() {
    curl -s "$BASE/health?verify=1" | .venv/bin/python -c '
import json, sys
h = json.load(sys.stdin)
keys = ["pinned_frames", "prompt_frames", "window_frames", "sliding_frames",
        "cache_offset", "prompt_intact"]
print("  " + "  ".join(f"{k}={h.get(k)}" for k in keys))
'
}

echo "before:"
show
echo "conversation (60s of silence, then a question):"
.venv/bin/python tools/voice_probe.py --pad 60 --frame-ms 200 \
    --question "What have we been talking about?" 2>&1 | tail -6
echo "after:"
show
