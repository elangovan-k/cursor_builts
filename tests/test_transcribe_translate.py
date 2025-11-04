"""Unit tests for the `transcribe_translate` module."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
import torch

from transcribe_translate import (
    SAMPLE_RATE,
    SpeakerSegment,
    TimedChunk,
    load_audio_features,
    map_speakers_to_roles,
    match_chunks_to_speakers,
    merge_consecutive_turns,
    transcribe_and_translate,
)


def test_load_audio_features(monkeypatch):
    """Ensure audio is loaded and processed into the expected tensor."""

    dummy_audio = np.random.rand(SAMPLE_RATE).astype(np.float32)

    def fake_load(path: str, sr: int):
        assert path == "dummy.mp3"
        assert sr == SAMPLE_RATE
        return dummy_audio, sr

    class DummyProcessor:
        def __call__(self, audio, sampling_rate, return_tensors):
            assert np.allclose(audio, dummy_audio)
            assert sampling_rate == SAMPLE_RATE
            assert return_tensors == "pt"

            class DummyOutput:
                input_features = torch.ones(1, 80, 5)

            return DummyOutput()

    monkeypatch.setattr("librosa.load", fake_load)

    features = load_audio_features("dummy.mp3", DummyProcessor())

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 80, 5)


def test_transcribe_and_translate_extracts_text_and_chunks():
    """Validate translation pipeline output handling."""

    class DummyPipeline:
        def __call__(self, audio_path: str, return_timestamps: bool):
            assert audio_path == "fake.mp3"
            assert return_timestamps is True
            return {
                "text": " Hello world! ",
                "chunks": [
                    {"timestamp": (0.0, 1.0), "text": "Hello"},
                    {"timestamp": (1.0, 2.0), "text": "world!"},
                ],
            }

    text, chunks = transcribe_and_translate("fake.mp3", DummyPipeline())

    assert text == "Hello world!"
    assert chunks == [
        TimedChunk(start=0.0, end=1.0, text="Hello"),
        TimedChunk(start=1.0, end=2.0, text="world!"),
    ]


def test_match_chunks_to_speakers_prefers_max_overlap():
    """Confirm diarization segments are matched based on highest overlap."""

    chunks = [
        TimedChunk(start=0.0, end=2.0, text="Hi"),
        TimedChunk(start=2.0, end=4.0, text="Hello"),
    ]
    speakers = [
        SpeakerSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
        SpeakerSegment(start=1.0, end=3.0, speaker="SPEAKER_01"),
        SpeakerSegment(start=3.0, end=5.0, speaker="SPEAKER_02"),
    ]

    assignments = match_chunks_to_speakers(chunks, speakers)

    assert assignments == [
        ("SPEAKER_01", "Hi"),
        ("SPEAKER_02", "Hello"),
    ]


def test_map_speakers_to_roles_handles_additional_speakers():
    """Ensure speaker labels map to Customer / Rep / Additional roles."""

    assignments: List[Tuple[str, str]] = [
        ("SPEAKER_00", "Hi"),
        ("SPEAKER_01", "Hello"),
        ("SPEAKER_02", "Greetings"),
    ]

    mapped = map_speakers_to_roles(assignments)

    assert mapped == [
        ("Customer", "Hi"),
        ("Customer Rep", "Hello"),
        ("Additional Speaker 1", "Greetings"),
    ]


def test_merge_consecutive_turns_combines_same_roles():
    """Verify consecutive turns from the same role are merged cleanly."""

    turns = [
        ("Customer", "Hello"),
        ("Customer", " there"),
        ("Customer Rep", "Hi"),
        ("Customer Rep", " how can I help?"),
        ("Customer", "Thanks"),
    ]

    merged = merge_consecutive_turns(turns)

    assert merged == [
        ("Customer", "Hello there"),
        ("Customer Rep", "Hi how can I help?"),
        ("Customer", "Thanks"),
    ]
