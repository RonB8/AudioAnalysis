"""
pytest configuration and shared fixtures for the Audio Quality QA Suite.

This module generates a set of synthetic WAV files into the ``test_data/``
directory before any test runs.  Each file is designed to exercise a specific
quality scenario so that the test suite can validate both detection of defects
and confirmation of clean audio.

Generated files
---------------
clean_speech.wav   — speech-like harmonic signal with natural pauses; passes all checks
clipped_audio.wav  — hard-clipped sine wave; fails the clipping check
long_silence.wav   — audio with a 1.5 s gap simulating packet loss; fails the silence check
low_snr_audio.wav  — weak signal buried in noise; fails the SNR check
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants used when synthesising test audio
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000          # Standard narrowband telephony rate (Hz)
TEST_DATA_DIR: Path = Path(__file__).parent.parent / "test_data"

# Seed for reproducibility — the same noise pattern is generated every run
_RNG_SEED: int = 42


# ---------------------------------------------------------------------------
# Private synthesis helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write *audio* to *path* as a 32-bit float WAV file."""
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="FLOAT")


def _generate_clean_speech(path: Path, rng: np.random.Generator) -> None:
    """Harmonic speech-like signal with brief inter-word pauses and very low noise.

    Structure: [active 0.4 s] [pause 0.1 s] [active 0.4 s] [pause 0.1 s] [active 0.4 s]
    Total duration: 1.4 s

    The pauses are shorter than MAX_SILENCE_DURATION_SEC (0.5 s), so the silence
    check passes.  Peak amplitude is 0.5 (well below the clipping threshold 0.99),
    and the signal-to-noise ratio is very high.
    """
    n_active = int(SAMPLE_RATE * 0.4)
    n_pause = int(SAMPLE_RATE * 0.1)
    t_active = np.linspace(0.0, 0.4, n_active, endpoint=False)

    # Three harmonic partials simulate a voiced telephone speech segment
    active_segment = (
        0.30 * np.sin(2 * np.pi * 300 * t_active)
        + 0.15 * np.sin(2 * np.pi * 600 * t_active)
        + 0.10 * np.sin(2 * np.pi * 900 * t_active)
        + 0.001 * rng.standard_normal(n_active)  # Barely audible noise floor
    )

    # Very quiet pause — noise level is below SILENCE_RMS_THRESHOLD (0.001)
    pause_segment = 0.0004 * rng.standard_normal(n_pause)

    audio = np.concatenate(
        [active_segment, pause_segment, active_segment, pause_segment, active_segment]
    )

    # Normalise to 0.5 full scale — no sample can ever reach the clipping threshold
    audio = audio / np.max(np.abs(audio)) * 0.5
    _write_wav(path, audio)


def _generate_clipped_audio(path: Path) -> None:
    """Sine wave amplified 2× then hard-clipped to ±1.0.

    Approximately 67 % of samples reside at the clipping rail (±1.0), which is
    far above the MAX_CLIPPING_RATIO (0.1 %) threshold.  Simulates an
    overdriven microphone pre-amplifier or a saturated codec stage.
    """
    n_samples = int(SAMPLE_RATE * 3.0)
    t = np.linspace(0.0, 3.0, n_samples, endpoint=False)

    audio = np.sin(2 * np.pi * 440 * t) * 2.0  # Exceeds ±1.0 full scale
    audio = np.clip(audio, -1.0, 1.0)           # Hard clip — flat-tops on waveform
    _write_wav(path, audio)


def _generate_long_silence_audio(path: Path) -> None:
    """Telephony audio with a 1.5 s silence gap simulating network packet loss.

    Structure: [active 1.0 s] [silence 1.5 s] [active 0.5 s]
    The 1.5 s gap exceeds MAX_SILENCE_DURATION_SEC (0.5 s), triggering failure.
    """
    t1 = np.linspace(0.0, 1.0, int(SAMPLE_RATE * 1.0), endpoint=False)
    t2 = np.linspace(0.0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)

    audio = np.concatenate(
        [
            0.4 * np.sin(2 * np.pi * 440 * t1),  # Active: 1.0 s
            np.zeros(int(SAMPLE_RATE * 1.5)),      # Packet-loss gap: 1.5 s
            0.4 * np.sin(2 * np.pi * 440 * t2),  # Active: 0.5 s
        ]
    )
    _write_wav(path, audio)


def _generate_low_snr_audio(path: Path, rng: np.random.Generator) -> None:
    """Barely perceptible 440 Hz tone buried under heavy white noise.

    Signal amplitude is 0.02; noise standard deviation is 0.50.
    The true SNR is ≈ –28 dB, far below the MIN_ACCEPTABLE_SNR_DB (20 dB) threshold.
    Simulates a call with excessive background noise or a codec bit-error floor.
    """
    n_samples = int(SAMPLE_RATE * 3.0)
    t = np.linspace(0.0, 3.0, n_samples, endpoint=False)

    weak_signal = 0.02 * np.sin(2 * np.pi * 440 * t)
    heavy_noise = 0.50 * rng.standard_normal(n_samples)
    audio = weak_signal + heavy_noise

    # Normalise to 0.9 — prevents accidental clipping without changing the SNR ratio
    audio = audio / np.max(np.abs(audio)) * 0.9
    _write_wav(path, audio)


# ---------------------------------------------------------------------------
# Session-scoped fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def generate_test_audio_files() -> None:  # type: ignore[return]
    """Generate all synthetic test audio files once per pytest session.

    The fixture is marked ``autouse=True`` so every test benefits from it
    automatically.  Files are (re)created on every run to guarantee that the
    test data matches the synthesis parameters defined in this module.

    Yields control back to pytest after the files are written, then does
    nothing on teardown (files are intentionally left on disk for inspection).
    """
    rng = np.random.default_rng(seed=_RNG_SEED)

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    _generate_clean_speech(TEST_DATA_DIR / "clean_speech.wav", rng)
    _generate_clipped_audio(TEST_DATA_DIR / "clipped_audio.wav")
    _generate_long_silence_audio(TEST_DATA_DIR / "long_silence.wav")
    _generate_low_snr_audio(TEST_DATA_DIR / "low_snr_audio.wav", rng)

    yield  # All tests run after this point
