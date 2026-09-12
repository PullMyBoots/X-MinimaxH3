#!/usr/bin/env python3
"""Transcribe generated H3 audio and report dialogue timing adherence."""

from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


_UCONV = shutil.which("uconv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vad-method", choices=("pyannote", "silero"), default="pyannote")
    parser.add_argument("--align-model")
    parser.add_argument(
        "--whisperx-root",
        type=Path,
        help="optional WhisperX source checkout to import instead of site-packages",
    )
    args = parser.parse_args()
    if not args.contract.is_file():
        parser.error(f"contract does not exist: {args.contract}")
    missing = [str(path) for path in args.videos if not path.is_file()]
    if missing:
        parser.error(f"video does not exist: {missing[0]}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.whisperx_root is not None and not (
        args.whisperx_root / "whisperx" / "__init__.py"
    ).is_file():
        parser.error("--whisperx-root must contain whisperx/__init__.py")
    return args


@lru_cache(maxsize=1024)
def _normalize_han_variants(value: str) -> str:
    """Fold Traditional Chinese ASR spelling without a sample-specific map."""

    if _UCONV is None:
        return value
    completed = subprocess.run(
        [_UCONV, "-x", "Traditional-Simplified"],
        input=value,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def normalize_text(value: str) -> str:
    return "".join(
        character.lower()
        for character in _normalize_han_variants(value)
        if re.match(r"[\w\u3400-\u9fff]", character, re.UNICODE)
    )


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def expected_dialogue(document: dict) -> list[dict]:
    values = document.get("expected_dialogue")
    if values is None and isinstance(document.get("task"), dict):
        values = document["task"].get("expected_dialogue")
    if not isinstance(values, list):
        raise ValueError("contract must contain expected_dialogue")
    return values


def evaluate_segments(segments: list[dict], expected: list[dict]) -> dict:
    timed_words = [
        {
            **word,
            "segment_index": segment["index"],
        }
        for segment in segments
        for word in segment["words"]
        if word["start"] is not None and word["end"] is not None
    ]
    rows = []
    for item in expected:
        start, end = (float(value) for value in item["window_seconds"])
        words = [
            word
            for word in timed_words
            if start <= (float(word["start"]) + float(word["end"])) / 2 < end
        ]
        if words:
            observed_text = "".join(str(word["word"]) for word in words)
            segment_indices = list(dict.fromkeys(word["segment_index"] for word in words))
            observed_start = float(words[0]["start"])
            observed_end = float(words[-1]["end"])
        else:
            overlapping = [
                segment
                for segment in segments
                if start <= (float(segment["start"]) + float(segment["end"])) / 2 < end
            ]
            observed_text = "".join(segment["text"] for segment in overlapping)
            segment_indices = [segment["index"] for segment in overlapping]
            observed_start = None if not overlapping else float(overlapping[0]["start"])
            observed_end = None if not overlapping else float(overlapping[-1]["end"])
        observed = normalize_text(observed_text)
        target = normalize_text(str(item["text"]))
        distance = edit_distance(target, observed)
        rows.append(
            {
                "order": int(item["order"]),
                "expected_text": str(item["text"]),
                "expected_normalized": target,
                "window_seconds": [start, end],
                "observed_text": observed_text,
                "observed_normalized": observed,
                "segment_indices": segment_indices,
                "observed_start": observed_start,
                "observed_end": observed_end,
                "character_error_rate": distance / max(1, len(target)),
            }
        )
    expected_all = "".join(normalize_text(str(item["text"])) for item in expected)
    observed_all = normalize_text("".join(segment["text"] for segment in segments))
    return {
        "expected": rows,
        "full_transcript": "".join(segment["text"] for segment in segments),
        "full_transcript_normalized": observed_all,
        "expected_all_normalized": expected_all,
        "global_character_error_rate": edit_distance(expected_all, observed_all)
        / max(1, len(expected_all)),
        "unexpected_transcript_character_count": (
            len(observed_all) if not expected_all else 0
        ),
        "unexpected_speech_detected": bool(observed_all and not expected_all),
    }


def main() -> int:
    args = parse_args()
    if args.whisperx_root is not None:
        sys.path.insert(0, str(args.whisperx_root.resolve()))
    import whisperx

    document = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = expected_dialogue(document)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = whisperx.load_model(
        args.model,
        args.device,
        compute_type=args.compute_type,
        language=args.language,
        vad_method=args.vad_method,
        asr_options={
            "beam_size": args.beam_size,
            "condition_on_previous_text": False,
        },
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.language,
        device=args.device,
        model_name=args.align_model,
    )
    stem_counts: dict[str, int] = {}
    for video in args.videos:
        stem_counts[video.stem] = stem_counts.get(video.stem, 0) + 1
    for video in args.videos:
        audio = whisperx.load_audio(str(video))
        transcription = model.transcribe(
            audio,
            batch_size=args.batch_size,
            language=args.language,
            verbose=False,
        )
        aligned = whisperx.align(
            transcription["segments"],
            align_model,
            align_metadata,
            audio,
            args.device,
            return_char_alignments=False,
        )
        segments = []
        for index, segment in enumerate(aligned["segments"]):
            segments.append(
                {
                    "index": index,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": segment["text"],
                    "words": [
                        {
                            "start": (
                                None
                                if word.get("start") is None
                                else float(word["start"])
                            ),
                            "end": (
                                None
                                if word.get("end") is None
                                else float(word["end"])
                            ),
                            "word": word["word"],
                            "probability": (
                                None
                                if word.get("score") is None
                                else float(word["score"])
                            ),
                        }
                        for word in segment.get("words", [])
                    ],
                }
            )
        report = {
            "schema_version": 2,
            "video": str(video.resolve()),
            "contract": str(args.contract.resolve()),
            "model": args.model,
            "language_requested": args.language,
            "language_detected": transcription["language"],
            "language_probability": None,
            "device": args.device,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "batch_size": args.batch_size,
            "vad_method": args.vad_method,
            "alignment_model": args.align_model or "WhisperX default for zh",
            "text_normalization": (
                "unicode_word_lower_plus_icu_hant_to_hans"
                if _UCONV is not None
                else "unicode_word_lower_exact"
            ),
            "condition_on_previous_text": False,
            "whisperx_version": importlib.metadata.version("whisperx"),
            "whisperx_path": str(Path(whisperx.__file__).resolve()),
            "faster_whisper_version": importlib.metadata.version("faster-whisper"),
            "segments": segments,
            "evaluation": evaluate_segments(segments, expected),
        }
        output_stem = (
            video.stem
            if stem_counts[video.stem] == 1
            else f"{video.parent.parent.name}__{video.stem}"
        )
        output = args.output_dir / f"{output_stem}_whisper.json"
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(output)
        print(json.dumps(report["evaluation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
