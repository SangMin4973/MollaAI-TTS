import json
import logging
import struct
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parent
else:
    ROOT_DIR = Path.cwd()
LOCAL_PACKAGE_DIR = ROOT_DIR / "Orpheus-TTS" / "orpheus_tts_pypi"

if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

from orpheus_tts import OrpheusModel


logger = logging.getLogger("molla.orpheus_ws")

MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_VOICE = "tara"
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_STOP_TOKEN_IDS = [128258]

app = FastAPI(title="molla-orpheus-tts")
engine: OrpheusModel | None = None
player_connections: set[WebSocket] = set()

PLAYER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MollaAI TTS Player</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: radial-gradient(circle at top, #17324d, #08111a 55%);
      color: #e6eef7;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    main {
      width: min(760px, calc(100vw - 32px));
      background: rgba(6, 16, 26, 0.88);
      border: 1px solid rgba(144, 202, 249, 0.18);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { margin: 0 0 16px; color: #9fb3c8; }
    #status {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #163047;
      color: #9be7ff;
      margin-bottom: 16px;
    }
    #log {
      white-space: pre-wrap;
      min-height: 220px;
      padding: 16px;
      border-radius: 14px;
      background: #050b12;
      border: 1px solid rgba(144, 202, 249, 0.12);
      overflow: auto;
    }
    audio { width: 100%; margin-top: 16px; }
  </style>
</head>
<body>
  <main>
    <h1>MollaAI TTS Player</h1>
    <p>Keep this page open. When the LLM client sends text to the server, the generated audio will play here.</p>
    <div id="status">Connecting...</div>
    <div id="log"></div>
    <audio id="player" controls autoplay></audio>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const playerEl = document.getElementById("player");
    let chunks = [];

    function log(message) {
      const line = `[${new Date().toLocaleTimeString()}] ${message}`;
      logEl.textContent = `${line}\n${logEl.textContent}`.trim();
    }

    function connect() {
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${protocol}://${location.host}/ws/player`);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        statusEl.textContent = "Connected";
        log("player connected");
      };

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          const payload = JSON.parse(event.data);
          if (payload.event === "start") {
            chunks = [];
            log(`start voice=${payload.voice} text_len=${payload.text_len}`);
          } else if (payload.event === "end") {
            const blob = new Blob(chunks, { type: "audio/wav" });
            playerEl.src = URL.createObjectURL(blob);
            playerEl.play().catch(() => {});
            log(`end elapsed_ms=${payload.elapsed_ms} chunks=${payload.chunk_count}`);
          } else if (payload.event === "error") {
            log(`error ${payload.detail}`);
          }
          return;
        }

        chunks.push(event.data);
      };

      ws.onclose = () => {
        statusEl.textContent = "Disconnected. Reconnecting...";
        log("player disconnected");
        setTimeout(connect, 1000);
      };

      ws.onerror = () => {
        statusEl.textContent = "Socket error";
      };
    }

    connect();
  </script>
</body>
</html>
"""


class SpeakRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    stop_token_ids: list[int] = DEFAULT_STOP_TOKEN_IDS
    sample_rate: int = DEFAULT_SAMPLE_RATE


def create_wav_header(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bits_per_sample: int = 16,
    channels: int = 1,
) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0xFFFFFFFF

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def parse_request(raw_message: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return {"text": raw_message}

    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object or plain text.")

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("`text` is required.")

    return {
        "text": text.strip(),
        "voice": payload.get("voice", DEFAULT_VOICE),
        "max_tokens": int(payload.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "temperature": float(payload.get("temperature", DEFAULT_TEMPERATURE)),
        "top_p": float(payload.get("top_p", DEFAULT_TOP_P)),
        "repetition_penalty": float(
            payload.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        ),
        "stop_token_ids": payload.get("stop_token_ids", DEFAULT_STOP_TOKEN_IDS),
        "sample_rate": int(payload.get("sample_rate", DEFAULT_SAMPLE_RATE)),
    }


def build_engine() -> OrpheusModel:
    started_at = time.monotonic()
    model = OrpheusModel(
        model_name=MODEL_NAME,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    )
    logger.info("orpheus_model_loaded elapsed_ms=%s", int((time.monotonic() - started_at) * 1000))
    return model


def warmup_model(model: OrpheusModel) -> None:
    started_at = time.monotonic()
    warmup_stream = model.generate_speech(
        prompt="Hello. This is a short warmup request.",
        voice=DEFAULT_VOICE,
        max_tokens=256,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
        stop_token_ids=DEFAULT_STOP_TOKEN_IDS,
        request_id="startup-warmup",
    )
    for _ in warmup_stream:
        break
    logger.info("orpheus_warmup_done elapsed_ms=%s", int((time.monotonic() - started_at) * 1000))


async def broadcast_text(payload: dict[str, Any]) -> None:
    dead_connections: list[WebSocket] = []
    message = json.dumps(payload, ensure_ascii=False)
    for websocket in player_connections:
        try:
            await websocket.send_text(message)
        except Exception:
            dead_connections.append(websocket)

    for websocket in dead_connections:
        player_connections.discard(websocket)


async def broadcast_bytes(chunk: bytes) -> None:
    dead_connections: list[WebSocket] = []
    for websocket in player_connections:
        try:
            await websocket.send_bytes(chunk)
        except Exception:
            dead_connections.append(websocket)

    for websocket in dead_connections:
        player_connections.discard(websocket)


@app.on_event("startup")
async def startup_event() -> None:
    global engine
    logging.basicConfig(level=logging.INFO)
    engine = build_engine()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "engine_loaded": engine is not None})


@app.get("/", response_class=HTMLResponse)
def player_page(request: Request) -> HTMLResponse:
    return HTMLResponse(PLAYER_HTML)


@app.post("/warmup")
def warmup() -> JSONResponse:
    if engine is None:
        return JSONResponse({"status": "error", "detail": "engine not loaded"}, status_code=500)
    warmup_model(engine)
    return JSONResponse({"status": "ok"})


@app.post("/speak")
async def speak(payload: SpeakRequest) -> JSONResponse:
    if engine is None:
        return JSONResponse({"status": "error", "detail": "engine not loaded"}, status_code=500)

    request_started_at = time.monotonic()
    await broadcast_text(
        {
            "event": "start",
            "voice": payload.voice,
            "sample_rate": payload.sample_rate,
            "text_len": len(payload.text),
        }
    )
    await broadcast_bytes(create_wav_header(sample_rate=payload.sample_rate))

    chunk_count = 0
    total_audio_bytes = 0
    stream = engine.generate_speech(
        prompt=payload.text,
        voice=payload.voice,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        repetition_penalty=payload.repetition_penalty,
        stop_token_ids=payload.stop_token_ids,
        request_id=f"http-{int(request_started_at * 1000)}",
    )
    for chunk in stream:
        chunk_count += 1
        total_audio_bytes += len(chunk)
        await broadcast_bytes(chunk)

    elapsed_ms = int((time.monotonic() - request_started_at) * 1000)
    await broadcast_text(
        {
            "event": "end",
            "chunk_count": chunk_count,
            "audio_bytes": total_audio_bytes,
            "elapsed_ms": elapsed_ms,
        }
    )
    return JSONResponse(
        {
            "status": "ok",
            "chunk_count": chunk_count,
            "audio_bytes": total_audio_bytes,
            "elapsed_ms": elapsed_ms,
        }
    )


@app.websocket("/ws/player")
async def player_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    player_connections.add(websocket)
    logger.info("player_connected connections=%s", len(player_connections))
    try:
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        player_connections.discard(websocket)
        logger.info("player_disconnected connections=%s", len(player_connections))


@app.websocket("/ws/tts")
async def tts_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    if engine is None:
        await websocket.send_text(
            json.dumps({"event": "error", "detail": "engine not loaded"}, ensure_ascii=False)
        )
        await websocket.close(code=1011)
        return

    try:
        while True:
            raw_message = await websocket.receive_text()
            request_started_at = time.monotonic()

            try:
                payload = parse_request(raw_message)
            except ValueError as exc:
                await websocket.send_text(
                    json.dumps({"event": "error", "detail": str(exc)}, ensure_ascii=False)
                )
                continue

            await websocket.send_text(
                json.dumps(
                    {
                        "event": "start",
                        "voice": payload["voice"],
                        "sample_rate": payload["sample_rate"],
                        "text_len": len(payload["text"]),
                    },
                    ensure_ascii=False,
                )
            )
            await websocket.send_bytes(create_wav_header(sample_rate=payload["sample_rate"]))

            first_chunk_sent = False
            chunk_count = 0
            total_audio_bytes = 0
            stream = engine.generate_speech(
                prompt=payload["text"],
                voice=payload["voice"],
                max_tokens=payload["max_tokens"],
                temperature=payload["temperature"],
                top_p=payload["top_p"],
                repetition_penalty=payload["repetition_penalty"],
                stop_token_ids=payload["stop_token_ids"],
                request_id=f"ws-{int(request_started_at * 1000)}",
            )
            for chunk in stream:
                chunk_count += 1
                total_audio_bytes += len(chunk)
                if not first_chunk_sent:
                    first_chunk_sent = True
                    logger.info(
                        "tts_first_chunk_sent elapsed_ms=%s text_len=%s",
                        int((time.monotonic() - request_started_at) * 1000),
                        len(payload["text"]),
                    )
                await websocket.send_bytes(chunk)

            await websocket.send_text(
                json.dumps(
                    {
                        "event": "end",
                        "chunk_count": chunk_count,
                        "audio_bytes": total_audio_bytes,
                        "elapsed_ms": int((time.monotonic() - request_started_at) * 1000),
                    },
                    ensure_ascii=False,
                )
            )
    except WebSocketDisconnect:
        logger.info("tts_websocket_disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("orpheus_ws_server:app", host="0.0.0.0", port=8000)
