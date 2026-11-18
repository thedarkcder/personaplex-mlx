"""Serve the local audio annotation workspace without loading PersonaPlex."""

import argparse
import base64
import binascii
import fcntl
import hashlib
import io
import json
import os
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

SAMPLE_RATE = 24000
MAX_SECONDS = 60
MAX_WAV_BYTES = SAMPLE_RATE * MAX_SECONDS * 2 + 1024
MAX_BASE64_CHARS = (MAX_WAV_BYTES + 2) // 3 * 4
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "tool_data"
HTML_PATH = Path(__file__).with_suffix(".html")


DEMO_TOOLS = [
    {
        "name": "weather_forecast",
        "description": "Get the forecast for a city on a specified day.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "day": {"type": "string"},
            },
            "required": ["city", "day"],
        },
    },
    {
        "name": "next_departure",
        "description": "Find the next train departure from a station.",
        "inputSchema": {
            "type": "object",
            "properties": {"station": {"type": "string"}},
            "required": ["station"],
        },
    },
    {
        "name": "current_time",
        "description": "Read the current local time.",
        "inputSchema": {"type": "object", "additionalProperties": False},
    },
]


class CatalogImport(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    tools: list[dict] = Field(min_length=1)


class ClipSubmission(BaseModel):
    id: UUID
    wav_base64: str = Field(min_length=1, max_length=MAX_BASE64_CHARS)
    speaker_id: str = Field(min_length=1, max_length=100)
    transcript: str = Field(min_length=1, max_length=8000)
    catalog_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome: Literal["call", "underspecified", "no_call"]
    tool: str | None = None
    arguments: dict | None = None
    ready_s: float | None = Field(default=None, ge=0, le=MAX_SECONDS)
    intent_s: float | None = Field(default=None, ge=0, le=MAX_SECONDS)
    turn_end_s: float = Field(gt=0, le=MAX_SECONDS)


def catalog_record(name: str, tools: list[dict]) -> dict:
    names = set()
    for tool in tools:
        tool_name = tool.get("name")
        schema = tool.get("inputSchema")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("Each tool needs a name.")
        if tool_name in names:
            raise ValueError(f"Tool name appears twice: {tool_name}")
        names.add(tool_name)
        if not isinstance(schema, dict) or schema.get("type", "object") != "object":
            raise ValueError(f"{tool_name} needs an object inputSchema.")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            raise ValueError(f"{tool_name} needs a list of required argument names.")

    canonical = json.dumps(tools, sort_keys=True, ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {"id": digest, "name": name, "tools": tools}


def decode_wav(encoded: str) -> tuple[bytes, float]:
    try:
        data = base64.b64decode(encoded, validate=True)
        with wave.open(io.BytesIO(data), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
                1, 2, SAMPLE_RATE,
            ):
                raise ValueError("Use a 24 kHz mono WAV with 16-bit PCM.")
            frames = audio.getnframes()
            if not 0 < frames <= MAX_SECONDS * SAMPLE_RATE or len(data) > MAX_WAV_BYTES:
                raise ValueError(f"Record a clip up to {MAX_SECONDS} seconds long.")
            if len(audio.readframes(frames)) != frames * 2:
                raise ValueError("The audio clip is truncated.")
    except (wave.Error, EOFError, binascii.Error) as error:
        raise ValueError("The audio clip is not a valid WAV.") from error
    return data, frames / SAMPLE_RATE


def manifest_record(clip: ClipSubmission, catalog: dict, duration: float) -> dict:
    speaker_id = clip.speaker_id.strip()
    transcript = clip.transcript.strip()
    if not speaker_id or not transcript:
        raise ValueError("Enter a speaker ID and transcript.")
    if clip.turn_end_s > duration:
        raise ValueError("Turn end must be within the recording.")
    if clip.intent_s is not None and clip.intent_s > clip.turn_end_s:
        raise ValueError("Tool intent must be marked before or at turn end.")

    call = None
    tool_intent = None
    if clip.outcome != "no_call":
        selected = next((tool for tool in catalog["tools"] if tool["name"] == clip.tool), None)
        if selected is None:
            raise ValueError("Choose a tool from the selected catalog.")
        if clip.intent_s is not None:
            tool_intent = {"tool": clip.tool, "end_s": clip.intent_s}

    if clip.outcome == "call":
        if clip.ready_s is None or not 0 < clip.ready_s <= clip.turn_end_s:
            raise ValueError("Mark a ready time after the start and at or before turn end.")
        if clip.intent_s is not None and clip.intent_s > clip.ready_s:
            raise ValueError("Tool intent must be marked at or before readiness.")
        if clip.arguments is None:
            raise ValueError("Enter the expected arguments as a JSON object.")
        missing = set(selected["inputSchema"].get("required", [])) - clip.arguments.keys()
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(sorted(missing))}")
        call = {"tool": clip.tool, "arguments": clip.arguments, "ready_s": clip.ready_s}
    elif clip.outcome == "underspecified" and tool_intent is None:
        raise ValueError("Mark when the incomplete request first indicates a tool.")

    record = {
        "id": str(clip.id),
        "audio": f"audio/{clip.id}.wav",
        "speaker_id": speaker_id,
        "transcript": transcript,
        "catalog_id": catalog["id"],
        "turn_end_s": clip.turn_end_s,
        "outcome": clip.outcome,
        "call": call,
    }
    if tool_intent is not None:
        record["tool_intent"] = tool_intent
    return record


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(contents)
            temporary.close()
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


@contextmanager
def dataset_lock(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / ".write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def create_app(data_dir: Path) -> FastAPI:
    app = FastAPI(title="PersonaPlex Audio Labeler", docs_url=None, redoc_url=None)
    demo = catalog_record("Example tools", DEMO_TOOLS)
    catalog_dir = data_dir / "catalogs"
    manifest = data_dir / "manifest.jsonl"

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.client and request.client.host not in {"127.0.0.1", "::1"}:
            return JSONResponse({"detail": "Open the labeler locally on this Mac."}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=FileResponse)
    def index():
        return FileResponse(HTML_PATH, media_type="text/html")

    @app.get("/api/catalogs")
    def list_catalogs():
        catalogs = [demo]
        for path in sorted(catalog_dir.glob("*.json")):
            if path.stem != demo["id"]:
                catalogs.append(json.loads(path.read_bytes()))
        return catalogs

    @app.post("/api/catalogs", status_code=201)
    def import_catalog(selection: CatalogImport):
        try:
            catalog = catalog_record(selection.name, selection.tools)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        path = catalog_dir / f'{catalog["id"]}.json'
        with dataset_lock(data_dir):
            if path.exists():
                return json.loads(path.read_bytes())
            atomic_write(path, json.dumps(catalog, ensure_ascii=False).encode())
        return catalog

    @app.post("/api/clips", status_code=201)
    def save_clip(clip: ClipSubmission):
        catalog_path = catalog_dir / f"{clip.catalog_id}.json"
        if clip.catalog_id == demo["id"]:
            catalog = demo
        elif catalog_path.is_file():
            catalog = json.loads(catalog_path.read_bytes())
        else:
            raise HTTPException(422, "The selected tool catalog is unavailable.")

        try:
            data, duration = decode_wav(clip.wav_base64)
            record = manifest_record(clip, catalog, duration)
            line = json.dumps(record, ensure_ascii=False, allow_nan=False).encode()
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

        audio_path = data_dir / record["audio"]
        with dataset_lock(data_dir):
            existing = manifest.read_bytes() if manifest.exists() else b""
            for saved_line in existing.splitlines():
                saved = json.loads(saved_line)
                if saved["id"] != record["id"]:
                    continue
                if saved == record and audio_path.read_bytes() == data:
                    return saved
                raise HTTPException(409, "This recording was already saved with different labels.")
            if not catalog_path.exists():
                atomic_write(catalog_path, json.dumps(catalog, ensure_ascii=False).encode())
            atomic_write(audio_path, data)
            try:
                atomic_write(manifest, existing + line + b"\n")
            except OSError:
                audio_path.unlink(missing_ok=True)
                raise
        return record

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    uvicorn.run(create_app(args.data_dir.resolve()), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
