"""
Automated Audio Quality Test Suite for VoIP / Telephony Validation.

Test organisation
-----------------
TestFileLoading         — verify all test files can be loaded and are non-empty
TestClippingDetection   — clipping check: clean audio passes, clipped audio fails
TestSilenceDetection    — silence check: clean audio passes, gapped audio fails
TestSNREstimation       — SNR check: clean audio passes, noisy audio fails
TestFullQualityReport   — end-to-end report for a known-good file
TestCleanAudioSuite     — parametrised: every "clean" file must pass all thresholds

All test data is created automatically by the session-scoped fixture in
conftest.py, so no manual file preparation is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.audio_analysis import (
    MAX_CLIPPING_RATIO,
    MAX_SILENCE_DURATION_SEC,
    MIN_ACCEPTABLE_SNR_DB,
    analyze_audio_file,
    detect_clipping,
    detect_silence_intervals,
    estimate_snr,
    load_audio,
)

# ---------------------------------------------------------------------------
# Paths — must match the files created by conftest.generate_test_audio_files
# ---------------------------------------------------------------------------

TEST_DATA_DIR: Path = Path(__file__).parent.parent / "test_data"

CLEAN_SPEECH: Path = TEST_DATA_DIR / "clean_speech.wav"
CLIPPED_AUDIO: Path = TEST_DATA_DIR / "clipped_audio.wav"
LONG_SILENCE: Path = TEST_DATA_DIR / "long_silence.wav"
LOW_SNR_AUDIO: Path = TEST_DATA_DIR / "low_snr_audio.wav"

# Files whose names appear in this list are expected to pass *all* quality checks.
# Add real-world "golden" files here as the test corpus grows.
CLEAN_AUDIO_FILENAMES: list[str] = [
    "clean_speech.wav",
    "A2.wav"
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load(path: Path):
    """Shorthand: load audio and return (samples, sample_rate)."""
    return load_audio(str(path))


# ---------------------------------------------------------------------------
# TestFileLoading
# ---------------------------------------------------------------------------


class TestFileLoading:
    """Basic smoke tests: every generated file must load cleanly."""

    @pytest.mark.parametrize(
        "audio_path",
        [CLEAN_SPEECH, CLIPPED_AUDIO, LONG_SILENCE, LOW_SNR_AUDIO],
        ids=["clean_speech", "clipped_audio", "long_silence", "low_snr_audio"],
    )
    def test_file_loads_without_error(self, audio_path: Path) -> None:
        """Audio file must load without raising any exception."""
        audio, sr = _load(audio_path)
        assert len(audio) > 0, f"Empty audio array returned for {audio_path.name}"

    @pytest.mark.parametrize(
        "audio_path",
        [CLEAN_SPEECH, CLIPPED_AUDIO, LONG_SILENCE, LOW_SNR_AUDIO],
        ids=["clean_speech", "clipped_audio", "long_silence", "low_snr_audio"],
    )
    def test_file_sample_rate_is_telephony_standard(self, audio_path: Path) -> None:
        """All files must be loaded at the 16 kHz narrowband telephony rate."""
        _, sr = _load(audio_path)
        assert sr == 16000, f"Expected 16000 Hz, got {sr} Hz for {audio_path.name}"

    def test_all_wav_files_present_in_test_data(self) -> None:
        """Confirm that the expected set of WAV files has been created."""
        expected = {"clean_speech.wav", "clipped_audio.wav", "long_silence.wav", "low_snr_audio.wav"}
        found = {p.name for p in TEST_DATA_DIR.glob("*.wav")}
        missing = expected - found
        assert not missing, f"Missing test data files: {missing}"


# ---------------------------------------------------------------------------
# TestClippingDetection
# ---------------------------------------------------------------------------


class TestClippingDetection:
    """Validate that the clipping detector correctly identifies saturation."""

    def test_clean_audio_has_no_clipping(self) -> None:
        """Clean speech (max amplitude 0.5) must not trigger the clipping flag."""
        audio, _ = _load(CLEAN_SPEECH)
        result = detect_clipping(audio)

        assert not result.is_clipping, (
            f"False-positive clipping detected in clean audio. "
            f"Clipping ratio: {result.clipping_ratio:.4%} "
            f"(threshold: {MAX_CLIPPING_RATIO:.4%})"
        )

    def test_clipped_audio_is_detected(self) -> None:
        """Hard-clipped audio must exceed the maximum allowed clipping ratio."""
        audio, _ = _load(CLIPPED_AUDIO)
        result = detect_clipping(audio)

        assert result.is_clipping, (
            f"Clipping not detected. Ratio: {result.clipping_ratio:.4%} "
            f"(must exceed {MAX_CLIPPING_RATIO:.4%})"
        )

    def test_clipping_ratio_is_in_valid_range(self) -> None:
        """Clipping ratio must always be a value between 0 and 1."""
        audio, _ = _load(CLIPPED_AUDIO)
        result = detect_clipping(audio)

        assert 0.0 <= result.clipping_ratio <= 1.0, (
            f"Clipping ratio {result.clipping_ratio} is outside [0, 1]"
        )

    def test_clipped_audio_reports_nonzero_clipped_samples(self) -> None:
        """Clipped file must report at least one clipped sample."""
        audio, _ = _load(CLIPPED_AUDIO)
        result = detect_clipping(audio)

        assert result.clipped_sample_count > 0

    def test_silence_audio_has_no_clipping(self) -> None:
        """Packet-loss simulation file uses 0.4 amplitude and must not clip."""
        audio, _ = _load(LONG_SILENCE)
        result = detect_clipping(audio)

        assert not result.is_clipping, (
            f"Unexpected clipping in long-silence file: {result.clipping_ratio:.4%}"
        )


# ---------------------------------------------------------------------------
# TestSilenceDetection
# ---------------------------------------------------------------------------


class TestSilenceDetection:
    """Validate that the silence detector catches network packet-loss gaps."""

    def test_clean_audio_has_no_unexpected_silence(self) -> None:
        """Brief natural pauses (0.1 s) in clean audio must not trigger the alarm."""
        audio, sr = _load(CLEAN_SPEECH)
        result = detect_silence_intervals(audio, sr)

        assert not result.has_unexpected_silence, (
            f"False-positive: unexpected silence in clean audio. "
            f"Longest gap: {result.longest_silence_sec:.3f}s "
            f"(limit: {MAX_SILENCE_DURATION_SEC}s)"
        )

    def test_long_silence_is_detected(self) -> None:
        """A 1.5 s packet-loss gap must exceed MAX_SILENCE_DURATION_SEC (0.5 s)."""
        audio, sr = _load(LONG_SILENCE)
        result = detect_silence_intervals(audio, sr)

        assert result.has_unexpected_silence, (
            f"Long silence not detected. "
            f"Longest gap found: {result.longest_silence_sec:.3f}s "
            f"(must exceed {MAX_SILENCE_DURATION_SEC}s)"
        )

    def test_detected_silence_duration_matches_expected(self) -> None:
        """The reported longest silence must be close to the injected 1.5 s gap."""
        audio, sr = _load(LONG_SILENCE)
        result = detect_silence_intervals(audio, sr)

        # Allow ±2 analysis frames of tolerance
        assert result.longest_silence_sec >= 1.3, (
            f"Detected silence duration {result.longest_silence_sec:.3f}s "
            f"is shorter than the expected ~1.5 s gap"
        )

    def test_silence_intervals_have_positive_duration(self) -> None:
        """Every reported silence interval must have a positive duration."""
        audio, sr = _load(LONG_SILENCE)
        result = detect_silence_intervals(audio, sr)

        for interval in result.silence_intervals:
            assert interval.duration_sec > 0, (
                f"Silence interval has non-positive duration: {interval}"
            )

    def test_clipped_audio_has_no_silence(self) -> None:
        """Continuous clipped sine wave must not contain any silence intervals."""
        audio, sr = _load(CLIPPED_AUDIO)
        result = detect_silence_intervals(audio, sr)

        assert not result.has_unexpected_silence, (
            f"False-positive silence in clipped audio: {result.longest_silence_sec:.3f}s"
        )


# ---------------------------------------------------------------------------
# TestSNREstimation
# ---------------------------------------------------------------------------


class TestSNREstimation:
    """Validate that the SNR estimator distinguishes clean from noisy signals."""

    def test_clean_audio_has_acceptable_snr(self) -> None:
        """Speech with a very low noise floor must report SNR >= MIN_ACCEPTABLE_SNR_DB."""
        audio, sr = _load(CLEAN_SPEECH)
        result = estimate_snr(audio, sr)

        assert result.is_acceptable, (
            f"SNR below threshold in clean audio: {result.snr_db:.2f} dB "
            f"(minimum required: {MIN_ACCEPTABLE_SNR_DB} dB)"
        )

    def test_low_snr_audio_is_rejected(self) -> None:
        """Heavily noise-dominated audio must report SNR < MIN_ACCEPTABLE_SNR_DB."""
        audio, sr = _load(LOW_SNR_AUDIO)
        result = estimate_snr(audio, sr)

        assert not result.is_acceptable, (
            f"Low-SNR audio passed unexpectedly: {result.snr_db:.2f} dB "
            f"(minimum: {MIN_ACCEPTABLE_SNR_DB} dB)"
        )

    def test_snr_returns_finite_value_for_valid_audio(self) -> None:
        """SNR estimate must be a finite (non-NaN, non-Inf) number for normal audio."""
        import math

        audio, sr = _load(CLEAN_SPEECH)
        result = estimate_snr(audio, sr)

        assert math.isfinite(result.snr_db), (
            f"SNR is not finite: {result.snr_db}"
        )

    def test_snr_powers_are_non_negative(self) -> None:
        """Signal and noise power estimates must be non-negative values."""
        audio, sr = _load(CLEAN_SPEECH)
        result = estimate_snr(audio, sr)

        assert result.signal_power >= 0, f"Negative signal power: {result.signal_power}"
        assert result.noise_power >= 0, f"Negative noise power: {result.noise_power}"


# ---------------------------------------------------------------------------
# TestFullQualityReport
# ---------------------------------------------------------------------------


class TestFullQualityReport:
    """End-to-end tests using the top-level analyze_audio_file() function."""

    def test_clean_audio_passes_full_report(self) -> None:
        """analyze_audio_file() must set passes_all_checks=True for clean speech."""
        report = analyze_audio_file(str(CLEAN_SPEECH))

        assert report.passes_all_checks, (
            f"Clean audio failed the full quality report:\n"
            f"  Clipping  : ratio={report.clipping.clipping_ratio:.4%}, "
            f"is_clipping={report.clipping.is_clipping}\n"
            f"  Silence   : longest={report.silence.longest_silence_sec:.3f}s, "
            f"unexpected={report.silence.has_unexpected_silence}\n"
            f"  SNR       : {report.snr.snr_db:.2f} dB, "
            f"acceptable={report.snr.is_acceptable}"
        )

    def test_report_contains_valid_duration(self) -> None:
        """The reported file duration must be a positive finite number."""
        report = analyze_audio_file(str(CLEAN_SPEECH))

        assert report.duration_sec > 0.0, (
            f"Non-positive duration in report: {report.duration_sec}"
        )

    def test_report_contains_correct_sample_rate(self) -> None:
        """The report must reflect the 16 kHz telephony sample rate."""
        report = analyze_audio_file(str(CLEAN_SPEECH))

        assert report.sample_rate == 16000, (
            f"Unexpected sample rate in report: {report.sample_rate}"
        )

    @pytest.mark.parametrize(
        "defective_file,expected_field",
        [
            (str(CLIPPED_AUDIO), "clipping"),
            (str(LONG_SILENCE), "silence"),
            (str(LOW_SNR_AUDIO), "snr"),
        ],
        ids=["clipped", "silent", "low_snr"],
    )
    def test_defective_files_fail_full_report(
        self, defective_file: str, expected_field: str
    ) -> None:
        """Each defective file must produce passes_all_checks=False."""
        report = analyze_audio_file(defective_file)

        assert not report.passes_all_checks, (
            f"Defective file incorrectly passed all quality checks: {defective_file}"
        )


# ---------------------------------------------------------------------------
# TestCleanAudioSuite  —  parametrised iteration over the clean-file corpus
# ---------------------------------------------------------------------------


class TestCleanAudioSuite:
    """Iterate through the CLEAN_AUDIO_FILENAMES corpus and assert full quality.

    Add filenames to CLEAN_AUDIO_FILENAMES at the top of this module to extend
    the test corpus with additional golden reference recordings.
    """

    @pytest.mark.parametrize("filename", CLEAN_AUDIO_FILENAMES)
    def test_no_clipping(self, filename: str) -> None:
        """Clean corpus file must not contain clipping artifacts."""
        audio, _ = _load(TEST_DATA_DIR / filename)
        result = detect_clipping(audio)

        assert not result.is_clipping, (
            f"[{filename}] Unexpected clipping: {result.clipping_ratio:.4%}"
        )

    @pytest.mark.parametrize("filename", CLEAN_AUDIO_FILENAMES)
    def test_no_unexpected_silence(self, filename: str) -> None:
        """Clean corpus file must not contain long silence gaps."""
        audio, sr = _load(TEST_DATA_DIR / filename)
        result = detect_silence_intervals(audio, sr)

        assert not result.has_unexpected_silence, (
            f"[{filename}] Unexpected silence: {result.longest_silence_sec:.3f}s"
        )

    @pytest.mark.parametrize("filename", CLEAN_AUDIO_FILENAMES)
    def test_acceptable_snr(self, filename: str) -> None:
        """Clean corpus file must meet the minimum SNR threshold."""
        audio, sr = _load(TEST_DATA_DIR / filename)
        result = estimate_snr(audio, sr)

        assert result.is_acceptable, (
            f"[{filename}] SNR too low: {result.snr_db:.2f} dB "
            f"(required >= {MIN_ACCEPTABLE_SNR_DB} dB)"
        )

    @pytest.mark.parametrize("filename", CLEAN_AUDIO_FILENAMES)
    def test_full_quality_report_passes(self, filename: str) -> None:
        """Clean corpus file must pass every check in the consolidated report."""
        report = analyze_audio_file(str(TEST_DATA_DIR / filename))

        assert report.passes_all_checks, (
            f"[{filename}] Quality report failed:\n"
            f"  Clipping: {report.clipping.clipping_ratio:.4%} clipped\n"
            f"  Silence : {report.silence.longest_silence_sec:.3f}s longest gap\n"
            f"  SNR     : {report.snr.snr_db:.2f} dB"
        )
