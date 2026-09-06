import os
from typing import Tuple
import numpy as np
import onnxruntime as ort

from config import settings
from models.download import ensure_model_exists


class ONNXInferenceEngine:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        if not os.path.exists(self.model_path):
            self.model_path = ensure_model_exists()

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        selected_providers = [p for p in providers if p in available_providers]

        self.session = ort.InferenceSession(self.model_path, providers=selected_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, tile: np.ndarray) -> np.ndarray:
        if tile.ndim == 3:
            if tile.shape[2] in [1, 3, 4] and tile.shape[0] not in [1, 3, 4]:
                tile = np.transpose(tile, (2, 0, 1))
            tile = np.expand_dims(tile, axis=0)

        tile_fp32 = tile.astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: tile_fp32})
        out = outputs[0]

        if out.ndim == 4 and out.shape[0] == 1:
            out = out[0]
        return out

    def infer_with_uncertainty(
        self,
        tile: np.ndarray,
        num_passes: int = None,
        noise_std: float = 0.015
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_passes = num_passes or settings.MC_DROPOUT_PASSES

        if tile.ndim == 3:
            if tile.shape[2] in [1, 3, 4] and tile.shape[0] not in [1, 3, 4]:
                tile = np.transpose(tile, (2, 0, 1))
            tile = np.expand_dims(tile, axis=0)

        passes = []
        for i in range(num_passes):
            if i == 0:
                perturbed = tile.astype(np.float32)
            else:
                noise = np.random.normal(0.0, noise_std, tile.shape).astype(np.float32)
                perturbed = np.clip(tile.astype(np.float32) + noise, 0.0, 1.0)

            out = self.infer(perturbed)
            passes.append(out)

        stacked = np.stack(passes, axis=0)
        mean_output = np.mean(stacked, axis=0)
        uncertainty_map = np.std(stacked, axis=0)

        return mean_output, uncertainty_map
