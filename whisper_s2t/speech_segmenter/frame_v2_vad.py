import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import VADBaseClass

logger = logging.getLogger(__name__)


class FrameV2VAD(VADBaseClass):
    """
    Frame-wise VAD powered by NVIDIA NeMo's Frame-VAD Multilingual MarbleNet v2.0.

    Model card:
      https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0

    Output format matches WhisperS2T VAD expectations:
      np.ndarray of shape (N, 3): [speech_prob, start_time_s, end_time_s]
    """

    MODEL_NAME = "nvidia/frame_vad_multilingual_marblenet_v2.0"

    def __init__(
        self,
        device: Optional[str] = None,
        chunk_size: float = 15.0,
        margin_size: float = 1.0,
        frame_size: float = 0.02,
        batch_size: int = 4,
        sampling_rate: int = 16000,
        model_name: str = MODEL_NAME,
        nemo_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(sampling_rate=sampling_rate)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.batch_size = int(batch_size)
        self.frame_size = float(frame_size)
        self.chunk_size = float(chunk_size)
        self.margin_size = float(margin_size)
        self.model_name = str(model_name)
        self.nemo_kwargs: Dict[str, Any] = dict(nemo_kwargs or {})

        # The published model expects 16kHz mono audio and 20ms frame shift.
        if self.sampling_rate != 16000:
            raise ValueError("FrameV2VAD currently supports sampling_rate=16000 only.")
        if abs(self.frame_size - 0.02) > 1e-9:
            raise ValueError("FrameV2VAD currently supports frame_size=0.02 only.")

        self._init_params()
        self._load_model()

    def _init_params(self):
        self.signal_chunk_len = int(self.chunk_size * self.sampling_rate)
        self.signal_stride = int(self.signal_chunk_len - 2 * int(self.margin_size * self.sampling_rate))

        self.margin_logit_len = int(self.margin_size / self.frame_size)
        self.signal_to_logit_len = int(self.frame_size * self.sampling_rate)  # 320 @16kHz and 20ms

    def _load_model(self):
        # Lazy import so the rest of the package works without NeMo installed.
        try:
            import nemo.collections.asr as nemo_asr
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "FrameV2VAD requires NVIDIA NeMo. Install with e.g. `pip install -U nemo_toolkit['asr']`."
            ) from e

        # NeMo versions differ in checkpoint strictness; some expect extra keys
        # (e.g. "loss.weight") that aren't always present in published artifacts.
        try:
            self.vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
                model_name=self.model_name, strict=False, **self.nemo_kwargs
            )
        except TypeError:
            # Older NeMo doesn't accept strict=...
            self.vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
                model_name=self.model_name, **self.nemo_kwargs
            )
        self.vad_model = self.vad_model.to(self.device)
        self.vad_model.eval()

    def update_params(self, params: Dict[str, Any] = {}):
        for key, value in params.items():
            setattr(self, key, value)
        self._init_params()

    def _prepare_input_batch(self, audio_signal: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        input_signal: List[np.ndarray] = []
        input_signal_length: List[int] = []

        for s_idx in range(0, len(audio_signal), self.signal_stride):
            sig = audio_signal[s_idx : s_idx + self.signal_chunk_len]
            sig_len = int(len(sig))
            input_signal.append(sig)
            input_signal_length.append(sig_len)

            if sig_len < self.signal_chunk_len:
                input_signal[-1] = np.pad(input_signal[-1], (0, self.signal_chunk_len - sig_len))
                break

        return input_signal, input_signal_length

    @staticmethod
    def _extract_logits(outputs: Any) -> torch.Tensor:
        # NeMo models typically return a Tensor, or a tuple/list with the first item being logits.
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (tuple, list)) and outputs:
            if isinstance(outputs[0], torch.Tensor):
                return outputs[0]
        if isinstance(outputs, dict):
            for key in ("logits", "outputs", "output"):
                v = outputs.get(key)
                if isinstance(v, torch.Tensor):
                    return v
        raise TypeError(f"Unexpected model output type: {type(outputs)}")

    @staticmethod
    def _logits_to_speech_probs(logits: torch.Tensor) -> torch.Tensor:
        # Expected: [B, T, C] (C=2) or [B, T]
        if logits.ndim == 3:
            if logits.shape[-1] == 2:
                return torch.softmax(logits, dim=-1)[..., 1]
            if logits.shape[-1] == 1:
                return torch.sigmoid(logits[..., 0])
            return torch.softmax(logits, dim=-1)[..., -1]
        if logits.ndim == 2:
            return torch.sigmoid(logits)
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    def _forward(self, input_signal: List[np.ndarray], input_signal_length: List[int]) -> np.ndarray:
        all_probs: List[torch.Tensor] = []

        autocast_ctx = (
            torch.cuda.amp.autocast() if (self.device != "cpu" and torch.cuda.is_available()) else torch.autocast("cpu")
        )
        # torch.autocast("cpu") is a no-op in many environments, but keeps the context uniform.
        with torch.inference_mode(), autocast_ctx:
            for s_idx in range(0, len(input_signal), self.batch_size):
                batch = input_signal[s_idx : s_idx + self.batch_size]
                batch_lens = input_signal_length[s_idx : s_idx + self.batch_size]

                # Make tensors efficiently; ensure writable contiguous buffers for torch.from_numpy.
                batch_np = [np.asarray(x, dtype=np.float32, order="C") for x in batch]
                batch_pt = torch.from_numpy(np.stack(batch_np, axis=0)).to(self.device)
                lens_pt = torch.tensor(batch_lens, dtype=torch.long, device=self.device)

                outputs = self.vad_model(input_signal=batch_pt, input_signal_length=lens_pt)
                logits = self._extract_logits(outputs)
                probs = self._logits_to_speech_probs(logits)  # [B, T]

                for p, ln in zip(probs, lens_pt):
                    # Map from samples to expected frame count (20ms hop).
                    t = int(ln.item() / self.signal_to_logit_len)
                    all_probs.append(p[:t].detach().to("cpu"))

        if len(all_probs) > 1 and self.margin_logit_len > 0:
            all_probs[0] = all_probs[0][:-self.margin_logit_len]
            all_probs[-1] = all_probs[-1][self.margin_logit_len:]
            for i in range(1, len(all_probs) - 1):
                all_probs[i] = all_probs[i][self.margin_logit_len:-self.margin_logit_len]

        if not all_probs:
            return np.zeros((0,), dtype=np.float32)

        return torch.cat(all_probs, dim=0).numpy().astype(np.float32, copy=False)

    def _probs_to_vad_times(self, speech_probs: np.ndarray, audio_duration_s: float) -> np.ndarray:
        n_frames = int(len(speech_probs))
        if n_frames == 0 or audio_duration_s <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        start_times = (np.arange(n_frames, dtype=np.float32) * np.float32(self.frame_size)).astype(
            np.float32, copy=False
        )
        end_times = np.minimum(start_times + np.float32(self.frame_size), np.float32(audio_duration_s)).astype(
            np.float32, copy=False
        )
        return np.stack([speech_probs.astype(np.float32, copy=False), start_times, end_times], axis=1).astype(
            np.float32, copy=False
        )

    def __call__(self, audio_signal: np.ndarray, batch_size: int = 4):  # noqa: ARG002
        audio_duration_s = len(audio_signal) / float(self.sampling_rate)

        input_signal, input_signal_length = self._prepare_input_batch(audio_signal)
        speech_probs = self._forward(input_signal, input_signal_length)
        return self._probs_to_vad_times(speech_probs, audio_duration_s)


