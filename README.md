# Automated Audio Quality QA Suite

A Python-based test framework for validating audio quality in VoIP and telephony
environments.  It detects the three most common defects found in telephony
recordings: **amplitude clipping**, **unexpected silence gaps** (packet loss),
and **low signal-to-noise ratio**.

---

## Directory Structure

```
audio-qa-suite/
├── .github/
│   └── workflows/
│       └── audio-qa.yml          # GitHub Actions CI/CD pipeline
├── src/
│   ├── __init__.py
│   └── audio_analysis.py         # Core analysis module (clipping, silence, SNR)
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures + synthetic test-data generator
│   └── test_audio_quality.py     # Full automated test suite
├── test_data/                    # Auto-populated by conftest on first pytest run
│   └── .gitkeep
├── reports/                      # HTML test reports land here
│   └── .gitkeep
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Prerequisites

- Python 3.10 or later
- Git (to clone the repository)
- On Linux / CI: `libsndfile1` system library (`sudo apt-get install libsndfile1`)
- On macOS: `brew install libsndfile`
- On Windows: no additional system libraries needed (wheels are bundled)

---

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:RonB8/AudioAnalysis.git
cd audio-qa-suite
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows (Command Prompt)**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Test Suite

### Basic run (console output)

```bash
pytest
```

### With a detailed HTML report

```bash
pytest --html=reports/test-report.html --self-contained-html
```

Open `reports/test-report.html` in any browser to review the results.

### Target a single test class

```bash
pytest tests/test_audio_quality.py::TestClippingDetection -v
```

### Stop on first failure

```bash
pytest -x
```

---

## Test Data

The session-scoped fixture in `tests/conftest.py` automatically generates four
synthetic WAV files in `test_data/` before any test runs:

| File | Contents | Expected outcome |
|------|----------|-----------------|
| `clean_speech.wav` | Harmonic signal + pauses + tiny noise | Passes all checks |
| `clipped_audio.wav` | Hard-clipped 440 Hz sine (2× overdrive) | Fails **clipping** check |
| `long_silence.wav` | 1.5 s silence gap (packet-loss simulation) | Fails **silence** check |
| `low_snr_audio.wav` | Weak signal buried in heavy white noise | Fails **SNR** check |

To test your own recordings, drop any WAV file into `test_data/` and add its
filename to `CLEAN_AUDIO_FILENAMES` in `tests/test_audio_quality.py` if it is
expected to pass all quality thresholds.

---

## Quality Thresholds

All defaults are defined as module-level constants in `src/audio_analysis.py`
and can be overridden per call:

| Constant | Default | Description |
|----------|---------|-------------|
| `CLIPPING_THRESHOLD` | `0.99` | Fraction of full scale above which a sample is clipped |
| `MAX_CLIPPING_RATIO` | `0.001` | Max acceptable fraction of clipped samples (0.1 %) |
| `MAX_SILENCE_DURATION_SEC` | `0.5` | Longest acceptable silence interval (seconds) |
| `SILENCE_RMS_THRESHOLD` | `0.001` | Per-frame RMS below which a frame is silent |
| `MIN_ACCEPTABLE_SNR_DB` | `20.0` | Minimum SNR in dB (ITU-T P.800 recommendation) |
| `NOISE_PERCENTILE` | `10.0` | Percentile of frame RMS used as noise floor estimate |

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/audio-qa.yml`) runs on every
push and pull request across Python 3.10, 3.11, and 3.12.  It:

1. Installs the `libsndfile1` system library
2. Creates a virtual environment and installs `requirements.txt`
3. Runs `pytest` with an HTML report
4. Uploads the report as a downloadable build artifact (retained for 30 days)

No pre-built audio files are committed — the synthetic test data is generated
at the start of each CI run by the session-scoped pytest fixture.

---

## Public API Reference

```python
from src.audio_analysis import (
    load_audio,               # Load WAV → (np.ndarray, int sample_rate)
    detect_clipping,          # → ClippingResult
    detect_silence_intervals, # → SilenceResult
    estimate_snr,             # → SNRResult
    analyze_audio_file,       # → AudioQualityReport  (full pipeline)
)
```

Each function accepts keyword arguments to override the default thresholds,
making them suitable for embedding in larger QA pipelines with different
quality profiles (e.g., wideband HD Voice vs. narrowband PSTN).
