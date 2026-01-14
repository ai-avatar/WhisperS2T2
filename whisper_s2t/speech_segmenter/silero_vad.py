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
        frame_size: float = 0.02,
        chunk_size: Optional[float] = 30.0,
        margin_size: float = 1.0,
        merge_gap: float = 0.05,
        debug_timing: bool = False,
        vad_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(sampling_rate=sampling_rate)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.frame_size = float(frame_size)
        self.chunk_size = chunk_size  # seconds; if None => process full audio in one pass
        self.margin_size = float(margin_size)
        self.merge_gap = float(merge_gap)
        self.debug_timing = bool(debug_timing)
        self.vad_kwargs: Dict[str, Any] = dict(vad_kwargs or {})

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

    @staticmethod
    def _merge_segments(segments_s: List[Tuple[float, float]], merge_gap: float) -> List[Tuple[float, float]]:
        if not segments_s:
            return []

        segments_s = sorted(((float(s), float(e)) for s, e in segments_s), key=lambda x: x[0])

        merged: List[Tuple[float, float]] = []
        cur_s, cur_e = segments_s[0]
        for s, e in segments_s[1:]:
            if s <= (cur_e + merge_gap):
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _iter_chunks(self, audio_signal: np.ndarray):
        """
        Yield (chunk_signal, chunk_start_sample, is_first, is_last).
        Uses overlap based on margin_size to avoid cutting speech at boundaries.
        """
        if self.chunk_size is None:
            yield audio_signal, 0, True, True
            return

        chunk_len = int(float(self.chunk_size) * self.sampling_rate)
        if chunk_len <= 0 or chunk_len >= len(audio_signal):
            yield audio_signal, 0, True, True
            return

        margin_len = int(float(self.margin_size) * self.sampling_rate)
        stride = max(1, chunk_len - 2 * margin_len)

        start = 0
        first = True
        while start < len(audio_signal):
            end = min(len(audio_signal), start + chunk_len)
            chunk = audio_signal[start:end]
            last = end >= len(audio_signal)
            yield chunk, start, first, last
            first = False
            if last:
                break
            start += stride

    def _call_silero(self, wav_tensor: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Call Silero get_speech_timestamps with compatibility shims across versions.
        We always try to get seconds out; if not supported, we convert from samples.
        """
        # Newer API: supports return_seconds=True
        try:
            return self._get_speech_timestamps(
                wav_tensor,
                self.model,
                sampling_rate=self.sampling_rate,
                return_seconds=True,
                **self.vad_kwargs,
            )
        except TypeError:
            pass

        # Older API: return samples (dicts with 'start'/'end' in samples)
        ts = self._get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=self.sampling_rate,
            **self.vad_kwargs,
        )
        # Convert samples -> seconds
        out: List[Dict[str, Any]] = []
        for seg in ts:
            out.append(
                {
                    "start": float(seg["start"]) / float(self.sampling_rate),
                    "end": float(seg["end"]) / float(self.sampling_rate),
                }
            )
        return out

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

        vad_times = np.zeros((n_frames, 3), dtype=np.float32)
        for i in range(n_frames):
            s_time = i * self.frame_size
            e_time = min(audio_duration_s, (i + 1) * self.frame_size)
            vad_times[i] = (probs[i], s_time, e_time)

        return vad_times

    @torch.no_grad()
    def __call__(self, audio_signal: np.ndarray, batch_size: int = 16):
        audio_duration_s = len(audio_signal) / float(self.sampling_rate)

        all_segments_s: List[Tuple[float, float]] = []

        import time
        t0 = time.time() if self.debug_timing else None

        # Process chunks in small batches (primarily reduces Python overhead on many chunks).
        chunk_batch = []
        for chunk_signal, chunk_start_sample, is_first, is_last in self._iter_chunks(audio_signal):
            chunk_batch.append((chunk_signal, chunk_start_sample, is_first, is_last))
            if len(chunk_batch) < max(1, int(batch_size)):
                continue

            all_segments_s.extend(self._process_chunk_batch(chunk_batch))
            chunk_batch = []

        if chunk_batch:
            all_segments_s.extend(self._process_chunk_batch(chunk_batch))

        all_segments_s = self._merge_segments(all_segments_s, merge_gap=self.merge_gap)

        if self.debug_timing and t0 is not None:
            print(f"SileroVAD time: {time.time() - t0} seconds")

        return self._segments_to_frame_probs(all_segments_s, audio_duration_s)

    def _process_chunk_batch(
        self, chunk_batch: List[Tuple[np.ndarray, int, bool, bool]]
    ) -> List[Tuple[float, float]]:
        """
        Run Silero VAD per chunk, trim margins to avoid duplicates, and shift to global timeline.
        """
        out: List[Tuple[float, float]] = []
        for chunk_signal, chunk_start_sample, is_first, is_last in chunk_batch:
            chunk_duration_s = len(chunk_signal) / float(self.sampling_rate)
            if chunk_duration_s <= 0:
                continue

            wav = torch.from_numpy(chunk_signal.astype(np.float32, copy=False))
            if self.device != "cpu":
                try:
                    wav = wav.to(self.device)
                except Exception:
                    logger.debug("SileroVAD: could not move chunk tensor to device=%s", self.device)

            segments = self._call_silero(wav)
            segments_s: List[Tuple[float, float]] = [(float(s["start"]), float(s["end"])) for s in segments]

            # Trim margins so overlaps don't duplicate segments between chunks.
            left_keep = 0.0 if is_first else float(self.margin_size)
            right_keep = float(chunk_duration_s) if is_last else float(chunk_duration_s - self.margin_size)

            if right_keep < left_keep:
                left_keep, right_keep = 0.0, float(chunk_duration_s)

            chunk_start_s = float(chunk_start_sample) / float(self.sampling_rate)
            for s, e in segments_s:
                s = max(float(s), left_keep)
                e = min(float(e), right_keep)
                if e > s:
                    out.append((s + chunk_start_s, e + chunk_start_s))

        return out


