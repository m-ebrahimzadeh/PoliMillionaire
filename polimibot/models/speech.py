"""Speech transcription wrapper. The only file that imports `whisper`."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TranscriberSpec:
    """Everything needed to load a Whisper model."""
    model_name: str = "base"        # "tiny", "base", "small", "medium", "large"
    device: str = "cpu"             # "cpu" or "cuda"
    language: str = "en"            # force English; None lets Whisper auto-detect


@dataclass
class TranscriptionResult:
    text: str
    elapsed_seconds: float


class SpeechTranscriber:
    """Thin wrapper over OpenAI Whisper for WAV-bytes → text.

    Usage:
        spec = TranscriberSpec(model_name="base", device="cuda")
        transcriber = SpeechTranscriber.load(spec)
        text = transcriber.transcribe(wav_bytes).text
    """

    def __init__(self, model, spec: TranscriberSpec) -> None:
        self._model = model
        self.spec = spec

    @property
    def name(self) -> str:
        return f"whisper-{self.spec.model_name}"

    @classmethod
    def load(cls, spec: TranscriberSpec | None = None) -> "SpeechTranscriber":
        """Load a Whisper model. Slow on first call; cache the result."""
        import whisper  # noqa: PLC0415  — intentional lazy import

        if spec is None:
            spec = TranscriberSpec()

        print(f"Loading Whisper ({spec.model_name}) on {spec.device}…")
        t0 = time.monotonic()
        model = whisper.load_model(spec.model_name, device=spec.device)
        print(f"Whisper loaded in {time.monotonic() - t0:.1f}s")
        return cls(model, spec)

    def transcribe(self, wav_bytes: bytes) -> TranscriptionResult:
        """Transcribe WAV bytes to text.

        Converts raw bytes to a float32 numpy array in-memory (no disk I/O),
        then runs Whisper inference.  Handles any sample rate — Whisper
        resamples to 16 kHz internally when given a numpy array.
        """
        import whisper  # noqa: PLC0415

        audio_np = self._wav_bytes_to_numpy(wav_bytes)

        t0 = time.monotonic()
        result = self._model.transcribe(
            audio_np,
            language=self.spec.language,
            fp16=(self.spec.device != "cpu"),
        )
        elapsed = time.monotonic() - t0

        return TranscriptionResult(
            text=result["text"].strip(),
            elapsed_seconds=round(elapsed, 4),
        )

    # ── private ────────────────────────────────────────────────────────────

    @staticmethod
    def _wav_bytes_to_numpy(wav_bytes: bytes) -> np.ndarray:
        """WAV bytes → float32 numpy array normalised to [-1, 1].

        Mirrors the in-memory pattern from the 11_NLP_Speech tutorial's
        get_audio() helper, which uses scipy.io.wavfile.read on a BytesIO
        buffer to avoid writing to disk.
        """
        from scipy.io.wavfile import read as wav_read  # noqa: PLC0415

        sr, audio = wav_read(io.BytesIO(wav_bytes))

        # Normalise integer PCM formats to float32 in [-1, 1].
        # Whisper's own load_audio() does the same normalisation.
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Stereo → mono by averaging channels
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        return audio
