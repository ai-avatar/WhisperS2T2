import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import VADBaseClass

logger = logging.getLogger(__name__)


class SileroVAD(VADBaseClass):
    """
    Silero VAD adapter that matches WhisperS2T's expected VAD output format:
      np.ndarray of shape (N, 3) where each row is [speech_prob, start_time_s, end_time_s]

    Under the hood we use Silero's `get_speech_timestamps(...)` (segment-level).
    We then project segments onto a fixed frame grid of `frame_size` seconds and
    emit hard probabilities (0/1). This preserves the existing SpeechSegmenter logic.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sampling_rate: int = 16000,
        vad_sampling_rate: Optional[int] = 8000,
        frame_size: float = 0.02,
        vad_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(sampling_rate=sampling_rate)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.frame_size = float(frame_size)
        self.vad_kwargs: Dict[str, Any] = dict(vad_kwargs or {})
        # Optional speed knob: run Silero VAD at 8kHz while keeping the main pipeline at 16kHz.
        # Segments are returned in seconds, so downstream alignment stays consistent.
        self.vad_sampling_rate = int(vad_sampling_rate or sampling_rate)

        # Lazy-import (keeps package usable without silero installed until needed)
        from silero_vad import get_speech_timestamps, load_silero_vad

        self._get_speech_timestamps = get_speech_timestamps
        self.model = load_silero_vad()

        # Best-effort device placement across Silero package variants
        try:
            self.model.to(self.device)
        except Exception:
            pass

        try:
            self.model.eval()
        except Exception:
            pass

    def update_params(self, params: Dict[str, Any] = {}):
        for key, value in params.items():
            setattr(self, key, value)

        # Allow updating vad_kwargs at runtime
        if "vad_kwargs" in params and params["vad_kwargs"] is not None:
            self.vad_kwargs = dict(params["vad_kwargs"])

    def _call_silero(self, wav_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Call Silero get_speech_timestamps with compatibility shims across versions.
        """
        # Some Silero variants keep an internal state; reset per file for correctness.
        try:
            self.model.reset_states()
        except Exception:
            pass

        return self._get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=self.vad_sampling_rate,
            return_seconds=True,
            **self.vad_kwargs,
        )

    def _segments_to_frame_probs(
        self, segments_s: List[Tuple[float, float]], audio_duration_s: float
    ) -> np.ndarray:
        if audio_duration_s <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        n_frames = int(np.ceil(audio_duration_s / self.frame_size))
        probs = np.zeros((n_frames,), dtype=np.float32)

        for start_s, end_s in segments_s:
            if end_s <= 0 or start_s >= audio_duration_s:
                continue
            start_s = max(0.0, float(start_s))
            end_s = min(audio_duration_s, float(end_s))
            if end_s <= start_s:
                continue

            start_idx = int(np.floor(start_s / self.frame_size))
            end_idx = int(np.ceil(end_s / self.frame_size)) - 1
            start_idx = max(0, min(n_frames - 1, start_idx))
            end_idx = max(0, min(n_frames - 1, end_idx))
            probs[start_idx : end_idx + 1] = 1.0

        # Vectorize frame grid construction (big win for long audio vs Python loop)
        start_times = (np.arange(n_frames, dtype=np.float32) * np.float32(self.frame_size)).astype(
            np.float32, copy=False
        )
        end_times = np.minimum(start_times + np.float32(self.frame_size), np.float32(audio_duration_s)).astype(
            np.float32, copy=False
        )
        return np.stack([probs, start_times, end_times], axis=1).astype(np.float32, copy=False)

    @torch.inference_mode()
    def __call__(self, audio_signal: np.ndarray, batch_size: int = 4):  # noqa: ARG002
        audio_duration_s = len(audio_signal) / float(self.sampling_rate)

        audio_vad = audio_signal
        if self.vad_sampling_rate != self.sampling_rate:
            if self.sampling_rate == 16000 and self.vad_sampling_rate == 8000:
                # Fast decimation; good enough for VAD in practice and ~2x fewer model steps.
                audio_vad = audio_signal[::2]
            else:
                raise ValueError(
                    f"Unsupported vad_sampling_rate={self.vad_sampling_rate} for input sampling_rate={self.sampling_rate}. "
                    "Supported: 16000->8000."
                )

        wav = torch.from_numpy(audio_vad.astype(np.float32, copy=False))
        if self.device != "cpu":
            try:
                wav = wav.to(self.device)
            except Exception:
                logger.debug("SileroVAD: could not move audio tensor to device=%s", self.device)

        segments = self._call_silero(wav)
        segments_s: List[Tuple[float, float]] = [(float(s["start"]), float(s["end"])) for s in segments]

        return self._segments_to_frame_probs(segments_s, audio_duration_s)


