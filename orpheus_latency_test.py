import argparse
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGE_DIR = ROOT_DIR / "Orpheus-TTS" / "orpheus_tts_pypi"

if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

from orpheus_tts import OrpheusModel


DEFAULT_PROMPT = (
    "Hey, thanks for calling today. I wanted to check in and see how your day is going so far."
)
DEFAULT_WARMUP_PROMPT = "Hello. This is a short warmup request."
DEFAULT_STOP_TOKEN_IDS = [128258]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Orpheus TTS startup and streaming latency."
    )
    parser.add_argument("--model-name", default="canopylabs/orpheus-tts-0.1-finetune-prod")
    parser.add_argument("--voice", default="tara")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-prompt", default=DEFAULT_WARMUP_PROMPT)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--stop-token-id",
        type=int,
        action="append",
        dest="stop_token_ids",
        help="Repeat this flag to pass multiple stop token ids. Default: 128258",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def format_ms(value: float) -> str:
    return f"{value * 1000:.1f}ms"


def measure_generation(
    model: OrpheusModel,
    *,
    prompt: str,
    voice: str,
    sample_rate: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stop_token_ids: list[int],
    request_id: str,
) -> dict[str, float | int | None | str]:
    chunk_count = 0
    total_bytes = 0
    first_chunk_latency = None

    started_at = time.monotonic()
    stream = model.generate_speech(
        prompt=prompt,
        voice=voice,
        request_id=request_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
    )
    for chunk in stream:
        if first_chunk_latency is None:
            first_chunk_latency = time.monotonic() - started_at
        chunk_count += 1
        total_bytes += len(chunk)

    total_time = time.monotonic() - started_at
    audio_duration = total_bytes / (sample_rate * 2) if total_bytes else 0.0
    realtime_factor = (total_time / audio_duration) if audio_duration else None

    return {
        "prompt": prompt,
        "chunk_count": chunk_count,
        "total_bytes": total_bytes,
        "first_chunk_latency": first_chunk_latency,
        "total_time": total_time,
        "audio_duration": audio_duration,
        "realtime_factor": realtime_factor,
    }


def print_report(title: str, stats: dict[str, float | int | None | str]) -> None:
    print(f"\n[{title}]")
    print(f"prompt_chars: {len(str(stats['prompt']))}")
    print(f"chunks: {stats['chunk_count']}")
    print(f"audio_bytes: {stats['total_bytes']}")
    print(
        "first_chunk_latency: "
        + (
            format_ms(float(stats["first_chunk_latency"]))
            if stats["first_chunk_latency"] is not None
            else "no audio"
        )
    )
    print(f"total_generation_time: {format_seconds(float(stats['total_time']))}")
    print(f"audio_duration: {format_seconds(float(stats['audio_duration']))}")
    if stats["realtime_factor"] is None:
        print("realtime_factor: n/a")
    else:
        print(f"realtime_factor: {float(stats['realtime_factor']):.3f}x")


def next_prompt(default_prompt: str) -> str | None:
    try:
        user_input = input("\nprompt> ").strip()
    except EOFError:
        return None
    except KeyboardInterrupt:
        return None

    if not user_input:
        return default_prompt
    if user_input.lower() in {"quit", "exit", ":q"}:
        return None
    return user_input


def main() -> None:
    args = parse_args()

    print("[setup]")
    print(f"cwd: {ROOT_DIR}")
    print(f"model_name: {args.model_name}")
    print(f"voice: {args.voice}")
    print(f"max_model_len: {args.max_model_len}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"stop_token_ids: {args.stop_token_ids or DEFAULT_STOP_TOKEN_IDS}")

    load_started_at = time.monotonic()
    model = OrpheusModel(
        model_name=args.model_name,
        max_model_len=args.max_model_len,
    )
    model_load_time = time.monotonic() - load_started_at
    print(f"model_load_time: {format_seconds(model_load_time)}")

    if not args.skip_warmup:
        warmup_stats = measure_generation(
            model,
            prompt=args.warmup_prompt,
            voice=args.voice,
            sample_rate=args.sample_rate,
            max_tokens=min(args.max_tokens, 256),
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=args.stop_token_ids or DEFAULT_STOP_TOKEN_IDS,
            request_id="warmup-001",
        )
        print_report("warmup", warmup_stats)

    print("\ninteractive mode: press Enter to reuse the default prompt, or type quit to exit.")
    request_index = 1
    while True:
        prompt = next_prompt(args.prompt)
        if prompt is None:
            print("\nexiting.")
            break

        benchmark_stats = measure_generation(
            model,
            prompt=prompt,
            voice=args.voice,
            sample_rate=args.sample_rate,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=args.stop_token_ids or DEFAULT_STOP_TOKEN_IDS,
            request_id=f"benchmark-{request_index:03d}",
        )
        print_report("benchmark", benchmark_stats)
        request_index += 1


if __name__ == "__main__":
    main()
