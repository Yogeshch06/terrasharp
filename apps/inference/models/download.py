import os
import logging
from huggingface_hub import hf_hub_download

from config import settings

logger = logging.getLogger("terrasharp.download")


def create_fallback_onnx_model(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        import torch
        import torch.nn as nn

        class FallbackUpsampler(nn.Module):
            def __init__(self, in_chans=4, out_chans=4, scale=4):
                super().__init__()
                self.conv1 = nn.Conv2d(in_chans, 32, 3, 1, 1)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(32, out_chans * (scale ** 2), 3, 1, 1)
                self.pixel_shuffle = nn.PixelShuffle(scale)

            def forward(self, x):
                feat = self.relu(self.conv1(x))
                out = self.pixel_shuffle(self.conv2(feat))
                return out

        model = FallbackUpsampler(in_chans=4, out_chans=4, scale=settings.UPSCALE_FACTOR)
        model.eval()
        dummy_input = torch.randn(1, 4, 64, 64, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"}
            }
        )
        logger.warning("WARNING: using untrained fallback model.")
    except Exception as e:
        logger.error(f"Failed to generate fallback ONNX model with PyTorch: {e}")


def ensure_model_exists():
    model_path = os.path.abspath(settings.MODEL_PATH)
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if settings.HF_MODEL_REPO:
        try:
            logger.info(f"Attempting to download model from Hugging Face: {settings.HF_MODEL_REPO}")
            downloaded = hf_hub_download(
                repo_id=settings.HF_MODEL_REPO,
                filename=settings.HF_MODEL_FILENAME,
                token=settings.HF_TOKEN,
                local_dir=os.path.dirname(model_path)
            )
            if downloaded != model_path and os.path.exists(downloaded):
                if os.path.exists(model_path):
                    os.remove(model_path)
                os.rename(downloaded, model_path)
            if os.path.exists(model_path):
                logger.info(f"Successfully downloaded model to {model_path}")
                return model_path
        except Exception as exc:
            logger.warning(f"Could not download model from Hugging Face ({exc}). Falling back to local generation.")

    create_fallback_onnx_model(model_path)
    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_model_exists()
