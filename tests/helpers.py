import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SR = 32000
N_SAMPLES = 1600


def write_mono_wav(path: Path, n_samples: int = N_SAMPLES, sr: int = SR) -> None:
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    waveform = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), waveform, sr)


def write_multichannel_wav(
    path: Path,
    n_channels: int = 4,
    n_samples: int = N_SAMPLES,
    sr: int = SR,
) -> None:
    channels = []
    for ch in range(n_channels):
        t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
        freq = 440 * (ch + 1)
        channels.append((0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    data = np.stack(channels, axis=1)
    sf.write(str(path), data, sr)


def make_waveform_dataset(
    root: Path,
    soundscape_name: str = "scene01",
    labels=None,
    include_oracle: bool = True,
) -> dict:
    if labels is None:
        labels = ["Speech", "Clapping"]

    soundscape_dir = root / "soundscape"
    oracle_dir = root / "oracle_target"
    soundscape_dir.mkdir(parents=True, exist_ok=True)
    if include_oracle:
        oracle_dir.mkdir(parents=True, exist_ok=True)

    write_multichannel_wav(soundscape_dir / f"{soundscape_name}.wav")
    if include_oracle:
        for idx, label in enumerate(labels):
            write_mono_wav(oracle_dir / f"{soundscape_name}_{idx}_{label}.wav")

    return {
        "soundscape_dir": str(soundscape_dir),
        "oracle_target_dir": str(oracle_dir) if include_oracle else None,
        "sr": SR,
    }


def make_mock_synthesize_output(
    labels,
    n_channels: int = 4,
    n_samples: int = N_SAMPLES,
    doas=None,
):
    if doas is None:
        doas = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    fg_events = []
    for idx, label in enumerate(labels):
        fg_events.append(
            {
                "metadata": {"label": label},
                "event_position": [doas[idx]],
                "waveform_dry": np.zeros((1, n_samples), dtype=np.float32),
            }
        )

    mixture = np.zeros((n_channels, n_samples), dtype=np.float32)
    return {
        "mixture": mixture,
        "fg_events": fg_events,
    }
