"""
Audio Quality Analysis Module for VoIP and Telephony Testing.

Provides functions to detect common audio quality defects:
  - Clipping     : signal saturation / codec overdrive
  - Silence       : unexpected gaps simulating network packet loss
  - SNR           : signal-to-noise ratio estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Quality threshold defaults — all values are tunable per call-site
# ---------------------------------------------------------------------------

CLIPPING_THRESHOLD: float = 0.99
"""Absolute amplitude level (fraction of full scale) above which a sample is
considered clipped.  Float WAV full scale is ±1.0, so 0.99 catches hard clips
as well as signals riding very close to saturation."""

MAX_CLIPPING_RATIO: float = 0.001
"""Maximum acceptable fraction of clipped samples in the entire file (0.1 %)."""

MAX_SILENCE_DURATION_SEC: float = 0.5
"""Longest silence interval allowed before the check flags a failure (seconds).
In telephony, gaps longer than ~500 ms typically indicate packet loss."""

SILENCE_RMS_THRESHOLD: float = 0.001
"""Per-frame RMS level below which a frame is classified as silent."""

MIN_ACCEPTABLE_SNR_DB: float = 20.0
"""Minimum acceptable Signal-to-Noise Ratio in dB.  ITU-T P.800 recommends
at least 20 dB for acceptable telephony quality."""

NOISE_PERCENTILE: float = 10.0
"""Percentile of per-frame RMS values used as the background noise estimate.
The quietest 10 % of frames are assumed to represent the noise floor."""

FRAME_LENGTH: int = 2048
"""Analysis frame length in samples (used for RMS computation)."""

HOP_LENGTH: int = 512
"""Hop length in samples between consecutive analysis frames."""

MIN_SILENCE_REPORT_SEC: float = 0.05
"""Minimum duration of a silence interval to include in the report (50 ms).
Shorter dips are treated as normal amplitude variation, not as silence."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClippingResult:
    """Outcome of the clipping detection check."""

    clipped_sample_count: int
    total_sample_count: int
    clipping_ratio: float
    is_clipping: bool


@dataclass
class SilenceInterval:
    """A single contiguous silence segment."""

    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class SilenceResult:
    """Outcome of the silence detection check."""

    silence_intervals: List[SilenceInterval] = field(default_factory=list)
    longest_silence_sec: float = 0.0
    has_unexpected_silence: bool = False


@dataclass
class SNRResult:
    """Outcome of the SNR estimation."""

    snr_db: float
    signal_power: float
    noise_power: float
    is_acceptable: bool


@dataclass
class AudioQualityReport:
    """Aggregated quality report for a single audio file."""

    file_path: str
    sample_rate: int
    duration_sec: float
    clipping: ClippingResult
    silence: SilenceResult
    snr: SNRResult
    passes_all_checks: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_audio(
    file_path: str,
    target_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to *target_sample_rate*.

    Returns a mono float32 array normalised to ±1.0 and the actual sample rate.
    Resampling is performed by librosa when the file's native rate differs from
    *target_sample_rate*.

    Args:
        file_path: Path to the WAV (or any librosa-supported) audio file.
        target_sample_rate: Desired output sample rate in Hz.

    Returns:
        Tuple of (audio_samples, sample_rate).
    """
    audio, sample_rate = librosa.load(file_path, sr=target_sample_rate, mono=True)
    return audio, int(sample_rate)


def detect_clipping(
    audio: np.ndarray,
    threshold: float = CLIPPING_THRESHOLD,
    max_ratio: float = MAX_CLIPPING_RATIO,
) -> ClippingResult:
    """Detect amplitude clipping in *audio*.

    A sample is considered clipped when its absolute value is greater than or
    equal to *threshold* (expressed as a fraction of the full-scale value of
    1.0 for float audio).  Clipping manifests as flat-tops on the waveform and
    produces harmonic distortion that degrades intelligibility on VoIP calls.

    Args:
        audio: Mono audio samples in the range [-1.0, 1.0].
        threshold: Amplitude level at or above which a sample is clipped.
        max_ratio: Maximum acceptable fraction of clipped samples.

    Returns:
        ClippingResult with per-file statistics and a pass/fail flag.
    """
    total_samples = len(audio)

    if total_samples == 0:
        return ClippingResult(
            clipped_sample_count=0,
            total_sample_count=0,
            clipping_ratio=0.0,
            is_clipping=False,
        )

    clipped_mask = np.abs(audio) >= threshold
    clipped_count = int(np.sum(clipped_mask))
    clipping_ratio = clipped_count / total_samples

    return ClippingResult(
        clipped_sample_count=clipped_count,
        total_sample_count=total_samples,
        clipping_ratio=clipping_ratio,
        is_clipping=clipping_ratio > max_ratio,
    )


def detect_silence_intervals(
    audio: np.ndarray,
    sample_rate: int,
    max_silence_duration_sec: float = MAX_SILENCE_DURATION_SEC,
    rms_threshold: float = SILENCE_RMS_THRESHOLD,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> SilenceResult:
    """Identify contiguous silence intervals in *audio*.

    Uses short-time RMS energy to classify each analysis frame as silent or
    active.  Consecutive silent frames are merged into intervals.  Any interval
    exceeding *max_silence_duration_sec* is flagged as unexpected silence,
    simulating the effect of network packet loss in a VoIP stream.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate of *audio* in Hz.
        max_silence_duration_sec: Silence duration threshold that triggers failure.
        rms_threshold: Per-frame RMS value below which a frame is silent.
        frame_length: Number of samples per analysis frame.
        hop_length: Number of samples between consecutive frame starts.

    Returns:
        SilenceResult with detected intervals and a pass/fail flag.
    """
    if len(audio) == 0:
        return SilenceResult()

    frame_rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]  # shape: (n_frames,)

    is_silent_frame = frame_rms < rms_threshold

    intervals: List[SilenceInterval] = []
    in_silence = False
    silence_start_frame = 0

    for frame_idx, silent in enumerate(is_silent_frame):
        if silent and not in_silence:
            in_silence = True
            silence_start_frame = frame_idx

        elif not silent and in_silence:
            in_silence = False
            start_sec = float(
                librosa.frames_to_time(silence_start_frame, sr=sample_rate, hop_length=hop_length)
            )
            end_sec = float(
                librosa.frames_to_time(frame_idx, sr=sample_rate, hop_length=hop_length)
            )
            duration_sec = end_sec - start_sec
            if duration_sec >= MIN_SILENCE_REPORT_SEC:
                intervals.append(
                    SilenceInterval(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        duration_sec=duration_sec,
                    )
                )

    # Handle audio that ends while still in a silence segment
    if in_silence:
        start_sec = float(
            librosa.frames_to_time(silence_start_frame, sr=sample_rate, hop_length=hop_length)
        )
        end_sec = len(audio) / sample_rate
        duration_sec = end_sec - start_sec
        if duration_sec >= MIN_SILENCE_REPORT_SEC:
            intervals.append(
                SilenceInterval(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=duration_sec,
                )
            )

    longest = max((iv.duration_sec for iv in intervals), default=0.0)

    return SilenceResult(
        silence_intervals=intervals,
        longest_silence_sec=longest,
        has_unexpected_silence=longest > max_silence_duration_sec,
    )


def estimate_snr(
    audio: np.ndarray,
    sample_rate: int,
    noise_percentile: float = NOISE_PERCENTILE,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    min_snr_db: float = MIN_ACCEPTABLE_SNR_DB,
) -> SNRResult:
    """Estimate the Signal-to-Noise Ratio of *audio*.

    Strategy (percentile-based noise floor estimation):
      1. Compute short-time RMS for each frame.
      2. Treat the quietest *noise_percentile* % of frames as the noise floor.
      3. Compute overall RMS of the entire signal as the combined signal level.
      4. SNR (dB) = 20 * log10(overall_rms / noise_floor_rms).

    This approach is effective for speech-like signals that contain natural
    pauses, where the pause frames carry the noise floor information.

    Args:
        audio: Mono audio samples.
        sample_rate: Sample rate in Hz (unused internally; kept for API symmetry).
        noise_percentile: Percentile of frame RMS values used as noise estimate.
        frame_length: Analysis frame length in samples.
        hop_length: Hop length in samples.
        min_snr_db: Minimum SNR (dB) to pass the check.

    Returns:
        SNRResult with power estimates, SNR in dB, and a pass/fail flag.
    """
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-10:
        return SNRResult(
            snr_db=-float("inf"),
            signal_power=0.0,
            noise_power=0.0,
            is_acceptable=False,
        )

    frame_rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    noise_rms = float(np.percentile(frame_rms, noise_percentile))
    if noise_rms < 1e-10:
        noise_rms = 1e-10  # Guard against log10(0)

    signal_rms = float(np.sqrt(np.mean(audio ** 2)))
    if signal_rms < 1e-10:
        return SNRResult(
            snr_db=-float("inf"),
            signal_power=0.0,
            noise_power=noise_rms ** 2,
            is_acceptable=False,
        )

    snr_db = float(20.0 * np.log10(signal_rms / noise_rms))

    return SNRResult(
        snr_db=snr_db,
        signal_power=signal_rms ** 2,
        noise_power=noise_rms ** 2,
        is_acceptable=snr_db >= min_snr_db,
    )


def analyze_audio_file(
    file_path: str,
    target_sample_rate: int = 16000,
    clipping_threshold: float = CLIPPING_THRESHOLD,
    max_clipping_ratio: float = MAX_CLIPPING_RATIO,
    max_silence_duration_sec: float = MAX_SILENCE_DURATION_SEC,
    silence_rms_threshold: float = SILENCE_RMS_THRESHOLD,
    min_snr_db: float = MIN_ACCEPTABLE_SNR_DB,
) -> AudioQualityReport:
    """Run all quality checks on a single audio file.

    This is the primary entry point for the QA pipeline.  It loads the file,
    runs clipping, silence, and SNR analysis, and bundles the results into a
    single :class:`AudioQualityReport`.

    Args:
        file_path: Path to the audio file under test.
        target_sample_rate: Sample rate to resample to before analysis (Hz).
        clipping_threshold: See :func:`detect_clipping`.
        max_clipping_ratio: See :func:`detect_clipping`.
        max_silence_duration_sec: See :func:`detect_silence_intervals`.
        silence_rms_threshold: See :func:`detect_silence_intervals`.
        min_snr_db: See :func:`estimate_snr`.

    Returns:
        AudioQualityReport summarising all check results.
    """
    audio, sample_rate = load_audio(file_path, target_sample_rate)
    duration_sec = len(audio) / sample_rate

    clipping_result = detect_clipping(audio, clipping_threshold, max_clipping_ratio)
    silence_result = detect_silence_intervals(
        audio, sample_rate, max_silence_duration_sec, silence_rms_threshold
    )
    snr_result = estimate_snr(audio, sample_rate, min_snr_db=min_snr_db)

    passes_all = (
        not clipping_result.is_clipping
        and not silence_result.has_unexpected_silence
        and snr_result.is_acceptable
    )

    return AudioQualityReport(
        file_path=str(file_path),
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        clipping=clipping_result,
        silence=silence_result,
        snr=snr_result,
        passes_all_checks=passes_all,
    )
