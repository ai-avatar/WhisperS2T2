import os
from typing import Dict, Optional, Tuple

import numpy as np

from . import VADBaseClass


class TenVADOnnx(VADBaseClass):
    """
    TEN VAD ONNX wrapper (GPU-capable when using onnxruntime-gpu + CUDAExecutionProvider).

    Contract: __call__(audio_signal) -> np.ndarray [T, 3] rows:
      [speech_prob, start_time, end_time]

    Notes:
    - This wrapper expects `ten-vad.onnx` to exist next to this file by default.
    - Uses runtime inspection to handle stateful vs stateless models.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        hop_size: int = 256,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        providers: Optional[list] = None,
        audio_input_name: Optional[str] = None,
        prob_output_name: Optional[str] = None,
    ):
        super().__init__(sampling_rate=sampling_rate)
        self.device = device
        self.hop_size = int(hop_size)
        self.threshold = float(threshold)
        self.audio_input_name = audio_input_name
        self.prob_output_name = prob_output_name

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "ten-vad.onnx")
        self.model_path = model_path

        if self.sampling_rate != 16000:
            raise ValueError("TenVADOnnx currently supports sampling_rate=16000 only.")
        if self.hop_size <= 0:
            raise ValueError("hop_size must be > 0")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TEN VAD ONNX model not found at: {self.model_path}")

        self._providers = providers
        self._session = None
        self._state_inputs: Dict[str, np.ndarray] = {}
        self._default_inputs: Dict[str, np.ndarray] = {}
        self._input_metas = []
        self._audio_input_meta = None
        self._init_session()

        self._hop_duration = self.hop_size / float(self.sampling_rate)

    def _init_session(self):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise ImportError(
                "onnxruntime is required for TenVADOnnx. For GPU, install onnxruntime-gpu."
            ) from e

        if self._providers is None:
            # Prefer CUDA when caller asked for cuda-like device; always keep CPU fallback.
            dev = (self.device or "").lower()
            if "cuda" in dev or dev.startswith("gpu"):
                self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        self._session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=self._providers)

        # Resolve audio input name if not provided.
        inputs = self._session.get_inputs()
        self._input_metas = inputs
        if self.audio_input_name is None:
            # Heuristic: pick the input that most likely represents a hop-sized audio frame.
            # We strongly prefer a tensor input whose shape explicitly includes hop_size (256),
            # otherwise fall back to dynamic 1D/2D tensors.
            def score(inp) -> Tuple[int, int, int]:
                t = (inp.type or "").lower()
                shape = inp.shape or []
                rank = len(shape)
                hop_match = 1 if any(isinstance(d, int) and d == self.hop_size for d in shape) else 0
                dtype_match = 1 if (("int16" in t) or ("float" in t) or ("int32" in t)) else 0
                # Prefer lower-rank inputs for raw audio
                rank_score = 0
                if rank == 1:
                    rank_score = 2
                elif rank == 2:
                    rank_score = 1
                return (hop_match, dtype_match, rank_score)

            candidates = [i for i in inputs if "tensor" in (i.type or "").lower()]
            if candidates:
                candidates.sort(key=score, reverse=True)
                self.audio_input_name = candidates[0].name
            elif len(inputs):
                self.audio_input_name = inputs[0].name

        for i in inputs:
            if i.name == self.audio_input_name:
                self._audio_input_meta = i
                break

        # Initialize non-audio inputs with defaults (zeros, or derived scalars such as sr/threshold).
        self._state_inputs = {}
        self._default_inputs = {}
        for i in inputs:
            if i.name == self.audio_input_name:
                continue
            self._default_inputs[i.name] = self._make_default_input(i)
            # Treat all non-audio tensors as "state" that can be updated if model outputs provide same-named tensors.
            self._state_inputs[i.name] = self._default_inputs[i.name]

        # Resolve prob output name if not provided.
        outputs = self._session.get_outputs()
        if self.prob_output_name is None:
            # heuristic: pick first float output
            for o in outputs:
                if "float" in (o.type or "").lower():
                    self.prob_output_name = o.name
                    break
            if self.prob_output_name is None and len(outputs):
                self.prob_output_name = outputs[0].name

    def _make_default_input(self, meta) -> np.ndarray:
        """
        Create a best-effort default value for a non-audio ONNX input.
        This prevents ORT _validate_input() errors on models that require extra inputs
        (e.g., recurrent state, cache, config scalars).
        """
        name = (meta.name or "").lower()
        t = (meta.type or "").lower()
        shape = meta.shape or []

        # Resolve dynamic dims to 1 (best-effort). If rank is unknown, treat as scalar.
        if len(shape) == 0:
            np_shape = ()
        else:
            dims = []
            for d in shape:
                if isinstance(d, int) and d > 0:
                    dims.append(d)
                else:
                    dims.append(1)
            np_shape = tuple(dims)

        # Choose dtype based on tensor type
        if "int64" in t:
            dtype = np.int64
        elif "int32" in t:
            dtype = np.int32
        elif "int16" in t:
            dtype = np.int16
        elif "float16" in t:
            dtype = np.float16
        else:
            dtype = np.float32

        # Heuristics for common scalar config inputs
        if np_shape == () or np_shape == (1,):
            if any(k in name for k in ["sr", "sample_rate", "sampling_rate", "rate"]):
                return np.array(self.sampling_rate, dtype=dtype).reshape(np_shape)
            if any(k in name for k in ["thresh", "threshold"]):
                return np.array(self.threshold, dtype=np.float32).astype(dtype).reshape(np_shape)

        # Default: zeros
        return np.zeros(np_shape, dtype=dtype)

    def _prepare_audio_input(self, frame_i16: np.ndarray) -> np.ndarray:
        """
        Adapt a hop frame into the ONNX model's expected dtype/shape.
        """
        meta = self._audio_input_meta
        if meta is None:
            return frame_i16

        t = (meta.type or "").lower()
        x: np.ndarray
        if "int16" in t:
            x = frame_i16.astype(np.int16, copy=False)
        elif "int32" in t:
            x = frame_i16.astype(np.int32)
        elif "float" in t:
            # Convert to float32 in [-1, 1] (common audio convention)
            x = (frame_i16.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
        else:
            x = frame_i16

        shape = meta.shape or []
        # Common cases:
        # - (hop,)               -> keep 1D
        # - (1, hop)             -> add batch dim
        # - (1, 1, hop)          -> add 2 dims
        # - (hop, 1) / (1, hop, 1) -> reshape accordingly
        rank = len(shape)
        if rank == 1:
            return x.reshape((self.hop_size,))
        if rank == 2:
            s0, s1 = shape
            if s0 == 1 and (s1 == self.hop_size or s1 is None or s1 == "?" or s1 == -1):
                return x.reshape((1, self.hop_size))
            if s1 == 1 and (s0 == self.hop_size or s0 is None or s0 == "?" or s0 == -1):
                return x.reshape((self.hop_size, 1))
            # fallback: batch it
            return x.reshape((1, self.hop_size))
        if rank == 3:
            # Prefer (1, 1, hop)
            return x.reshape((1, 1, self.hop_size))
        # fallback: keep 1D
        return x.reshape((self.hop_size,))

    def update_params(self, params={}):
        needs_reinit = False
        for key, value in params.items():
            if key in {"device", "model_path", "hop_size", "threshold", "sampling_rate", "providers", "audio_input_name", "prob_output_name"}:
                needs_reinit = True
            setattr(self, key, value)

        self.hop_size = int(self.hop_size)
        self.sampling_rate = int(self.sampling_rate)
        self.threshold = float(self.threshold)

        if needs_reinit:
            self._init_session()
            self._hop_duration = self.hop_size / float(self.sampling_rate)

    def _float_to_int16(self, audio_signal: np.ndarray) -> np.ndarray:
        x = np.asarray(audio_signal, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        return (x * 32768.0).astype(np.int16)

    def _run_frame(self, frame_i16: np.ndarray) -> float:
        assert self._session is not None
        # Always feed ALL required inputs to avoid ORT _validate_input errors.
        feed: Dict[str, np.ndarray] = {}
        for meta in self._input_metas:
            if meta.name == self.audio_input_name:
                feed[meta.name] = self._prepare_audio_input(frame_i16)
            else:
                v = self._state_inputs.get(meta.name, None)
                if v is None:
                    v = self._default_inputs.get(meta.name, None)
                if v is None:
                    v = self._make_default_input(meta)
                feed[meta.name] = v

        out_names = None  # all outputs
        outs = self._session.run(out_names, feed)

        # Map outputs by name for state updates / prob extraction
        out_meta = self._session.get_outputs()
        out_by_name = {m.name: v for m, v in zip(out_meta, outs)}

        # Update any state inputs if model outputs match input names.
        for k in list(self._state_inputs.keys()):
            if k in out_by_name:
                self._state_inputs[k] = out_by_name[k]

        prob = out_by_name.get(self.prob_output_name)
        if prob is None:
            # fallback: first output
            prob = outs[0] if len(outs) else 0.0

        # scalarize
        if isinstance(prob, np.ndarray):
            return float(np.asarray(prob).reshape(-1)[0])
        return float(prob)

    def __call__(self, audio_signal, batch_size=4):
        # batch_size kept for signature compatibility; not used.
        x16 = self._float_to_int16(audio_signal)
        audio_duration = len(x16) / float(self.sampling_rate)
        if len(x16) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        n_hops = (len(x16) + self.hop_size - 1) // self.hop_size
        vad_times = np.zeros((n_hops, 3), dtype=np.float32)

        for idx in range(n_hops):
            s = idx * self.hop_size
            e = min((idx + 1) * self.hop_size, len(x16))

            frame = x16[s:e]
            if len(frame) < self.hop_size:
                padded = np.zeros((self.hop_size,), dtype=np.int16)
                padded[: len(frame)] = frame
                frame = padded

            prob = self._run_frame(frame)

            s_time = idx * self._hop_duration
            if s_time >= audio_duration:
                break
            e_time = min(audio_duration, (idx + 1) * self._hop_duration)

            vad_times[idx, 0] = float(prob)
            vad_times[idx, 1] = float(s_time)
            vad_times[idx, 2] = float(e_time)

        valid = vad_times[:, 2] > 0
        return vad_times[valid]


