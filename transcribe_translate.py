"""Transcribe and translate an audio call recording with basic speaker attribution.

Steps performed:
1. Load the audio file `call_recording.mp3`.
2. Detect the spoken language (restricted to Indian languages or English).
3. Transcribe and translate the audio into English using OpenAI Whisper via Hugging Face.
4. Run speaker diarization to differentiate customer from customer representative turns.

Prerequisites:
- Install dependencies (PyPI):
    pip install transformers torch librosa soundfile pyannote.audio
- Provide a valid Hugging Face access token for the diarization pipeline via the
  `HUGGINGFACE_TOKEN` environment variable if required by the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import torch
from pyannote.audio import Pipeline as DiarizationPipeline
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)


AUDIO_FILE = "call_recording.mp3"
SAMPLE_RATE = 16000
WHISPER_MODEL_ID = "openai/whisper-large-v3"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization"


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass
class TimedChunk:
    start: float
    end: float
    text: str


def load_audio_features(
    audio_path: str,
    processor: WhisperProcessor,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Load audio and convert to Whisper input features."""

    audio, _ = librosa.load(audio_path, sr=sample_rate)
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).input_features
    return inputs


def detect_language(
    input_features: torch.Tensor,
    model: WhisperForConditionalGeneration,
) -> str:
    """Detect the spoken language using Whisper's language head."""

    with torch.inference_mode():
        language_logits = model.detect_language(input_features)
    language_id = int(torch.argmax(language_logits[0]))
    detected_language = model.config.id_to_language.get(language_id, "unknown")
    return detected_language


def build_asr_pipeline(device: str) -> callable:
    """Create a Whisper translation pipeline."""

    device_index = 0 if device == "cuda" else -1
    dtype = torch.float16 if device == "cuda" else torch.float32

    return pipeline(
        task="automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        device=device_index,
        torch_dtype=dtype,
        generate_kwargs={"task": "translate"},
    )


def transcribe_and_translate(
    audio_path: str,
    asr_pipeline,
) -> Tuple[str, List[TimedChunk]]:
    """Run Whisper translation and return the full text and per-chunk segments."""

    result = asr_pipeline(
        audio_path,
        return_timestamps=True,
    )

    full_text = result["text"].strip()
    chunks = []
    for chunk in result.get("chunks", []):
        timestamp = chunk.get("timestamp")
        if not timestamp or timestamp[0] is None or timestamp[1] is None:
            continue
        chunks.append(
            TimedChunk(
                start=float(timestamp[0]),
                end=float(timestamp[1]),
                text=chunk.get("text", "").strip(),
            )
        )

    return full_text, chunks


def run_diarization(audio_path: str, hf_token: Optional[str]) -> List[SpeakerSegment]:
    """Apply speaker diarization to the audio file."""

    pipeline_kwargs = {}
    if hf_token:
        pipeline_kwargs["use_auth_token"] = hf_token

    diarization_pipeline = DiarizationPipeline.from_pretrained(
        DIARIZATION_MODEL_ID,
        **pipeline_kwargs,
    )

    diarization = diarization_pipeline(audio_path)

    segments: List[SpeakerSegment] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                start=float(segment.start),
                end=float(segment.end),
                speaker=speaker,
            )
        )
    return segments


def match_chunks_to_speakers(
    chunks: Iterable[TimedChunk],
    speakers: List[SpeakerSegment],
) -> List[Tuple[str, str]]:
    """Assign each transcription chunk to the most overlapping speaker."""

    assignments: List[Tuple[str, str]] = []

    for chunk in chunks:
        best_speaker = "Unknown"
        best_overlap = 0.0
        for seg in speakers:
            overlap = min(chunk.end, seg.end) - max(chunk.start, seg.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg.speaker
        assignments.append((best_speaker, chunk.text))

    return assignments


def map_speakers_to_roles(speaker_assignments: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Map diarization speaker labels to customer-facing roles."""

    role_order = ["Customer", "Customer Rep"]
    role_map: Dict[str, str] = {}
    additional_count = 1
    mapped: List[Tuple[str, str]] = []

    for speaker, text in speaker_assignments:
        if speaker not in role_map:
            role_index = len(role_map)
            if role_index < len(role_order):
                role_map[speaker] = role_order[role_index]
            else:
                role_map[speaker] = f"Additional Speaker {additional_count}"
                additional_count += 1
        mapped.append((role_map[speaker], text))

    return mapped


def merge_consecutive_turns(turns: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Combine consecutive turns with the same role."""

    merged: List[Tuple[str, str]] = []
    for role, text in turns:
        cleaned_text = text.strip()
        if not cleaned_text:
            continue
        if merged and merged[-1][0] == role:
            merged[-1] = (role, f"{merged[-1][1]} {cleaned_text}")
        else:
            merged.append((role, cleaned_text))
    return merged


def main() -> None:
    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(
            f"Audio file '{AUDIO_FILE}' not found in the current directory."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Whisper models...")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    whisper_model.to(device)

    print("Detecting spoken language...")
    input_features = load_audio_features(AUDIO_FILE, processor).to(device)
    detected_language = detect_language(input_features, whisper_model)

    print(f"Detected language: {detected_language}")

    print("Running English translation with Whisper...")
    asr_pipeline = build_asr_pipeline(device)
    full_translation, chunks = transcribe_and_translate(AUDIO_FILE, asr_pipeline)

    if not chunks:
        raise RuntimeError("No transcription chunks were produced; unable to map speakers.")

    print("Performing speaker diarization...")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    diarization_segments = run_diarization(AUDIO_FILE, hf_token)

    if not diarization_segments:
        raise RuntimeError("Speaker diarization returned no segments.")

    speaker_assignments = match_chunks_to_speakers(chunks, diarization_segments)
    role_turns = map_speakers_to_roles(speaker_assignments)
    merged_turns = merge_consecutive_turns(role_turns)

    print("\n==== Conversation (English Translation) ====")
    for role, text in merged_turns:
        print(f"[{role}] {text}")

    print("\n==== Full English Translation (Unsegmented) ====")
    print(full_translation)


if __name__ == "__main__":
    main()
