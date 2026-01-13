import numpy as np

from . import VADBaseClass


class TenVAD(VADBaseClass):
    """
    Wrapper around TEN VAD (https://huggingface.co/TEN-framework/ten-vad) that conforms
    to WhisperS2T's VADBaseClass contract:

    __call__(audio_signal) -> np.ndarray of shape [T, 3] where each row is:
      [speech_prob, frame_start_time, frame_end_time]
    """

    def __init__(
        self,
        device=None,  # kept for drop-in compatibility; TEN VAD doesn't use it
        hop_size: int = 256,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self.device = device
        self.hop_size = int(hop_size)
        self.threshold = float(threshold)
        self._init_vad()

    def _init_vad(self):
        try:
            # ten-vad Python API exposes TenVad (ctypes wrapper around native libs)
            # Ref: https://huggingface.co/TEN-framework/ten-vad/blob/main/include/ten_vad.py
            from ten_vad import TenVad  # type: ignore
        except Exception as e:
            raise ImportError(
                "TEN VAD is not installed. Install it with: "
                "pip install 'ten-vad @ git+https://github.com/TEN-framework/ten-vad.git'"
            ) from e

        if self.hop_size <= 0:
            raise ValueError("hop_size must be > 0")
        if self.sampling_rate != 16000:
            # TEN VAD is primarily designed for 16kHz; keep explicit to avoid subtle bugs.
            raise ValueError("TEN VAD wrapper currently supports sampling_rate=16000 only.")

        self._ten_vad = TenVad(hop_size=self.hop_size, threshold=self.threshold)
        self._hop_duration = self.hop_size / float(self.sampling_rate)

    def update_params(self, params={}):
        needs_reinit = False
        for key, value in params.items():
            if not hasattr(self, key):
                setattr(self, key, value)
                continue

            if key in {"hop_size", "threshold", "sampling_rate"}:
                needs_reinit = True
            setattr(self, key, value)

        # Keep types sane
        self.hop_size = int(self.hop_size)
        self.threshold = float(self.threshold)
        self.sampling_rate = int(self.sampling_rate)

        if needs_reinit:
            self._init_vad()

    def _float_to_int16(self, audio_signal: np.ndarray) -> np.ndarray:
        # audio_signal is expected float32 in [-1, 1]
        x = np.asarray(audio_signal, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        x = (x * 32768.0).astype(np.int16)
        return x

    def __call__(self, audio_signal, batch_size=4):
        # batch_size kept for signature compatibility; not used by TEN VAD.
        x16 = self._float_to_int16(audio_signal)
        audio_duration = len(x16) / float(self.sampling_rate)

        if len(x16) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        n_hops = (len(x16) + self.hop_size - 1) // self.hop_size
        vad_times = np.zeros((n_hops, 3), dtype=np.float32)

        for idx in range(n_hops):
            s = idx * self.hop_size
            e = min((idx + 1) * self.hop_size, len(x16))

            # pad last frame to hop_size for TEN VAD
            frame = x16[s:e]
            if len(frame) < self.hop_size:
                padded = np.zeros((self.hop_size,), dtype=np.int16)
                padded[: len(frame)] = frame
                frame = padded

            prob, _flags = self._ten_vad.process(frame)

            s_time = idx * self._hop_duration
            if s_time >= audio_duration:
                break
            e_time = min(audio_duration, (idx + 1) * self._hop_duration)

            vad_times[idx, 0] = float(prob)
            vad_times[idx, 1] = float(s_time)
            vad_times[idx, 2] = float(e_time)

        # If we broke early (should be rare), trim trailing zeros
        valid = vad_times[:, 2] > 0
        return vad_times[valid]


