import os
import argparse
import yaml
import torch
import onnx

from model_arch import SwinIRLight


def parse_args():
    parser = argparse.ArgumentParser(description="Export TerraSharp SwinIR-Light to ONNX")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX filename")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    normalized_path = path.replace("/", os.sep)
    repo_relative_prefix = os.path.join("packages", "model-training")
    if normalized_path == repo_relative_prefix or normalized_path.startswith(repo_relative_prefix + os.sep):
        return os.path.abspath(os.path.join(repo_root, normalized_path))
    return os.path.abspath(path)


def export():
    args = parse_args()
    config = load_config(args.config)

    m_cfg = config.get("model", {})
    e_cfg = config.get("export", {})

    checkpoint_path = resolve_path(args.checkpoint or e_cfg.get("checkpoint_path", "packages/model-training/checkpoints/best.pth"))
    output_path = args.output or e_cfg.get("onnx_path", "terrasharp_swinir_sentinel2.onnx")
    opset_version = args.opset or e_cfg.get("opset_version", 17)

    model = SwinIRLight(
        in_chans=m_cfg.get("in_chans", 4),
        out_chans=m_cfg.get("out_chans", 4),
        embed_dim=m_cfg.get("embed_dim", 64),
        depths=m_cfg.get("depths", [2, 2]),
        num_heads=m_cfg.get("num_heads", [4, 4]),
        window_size=m_cfg.get("window_size", 8),
        mlp_ratio=m_cfg.get("mlp_ratio", 2.0),
        upscale=m_cfg.get("upscale", 4)
    )

    print(f"Resolved checkpoint path: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found; refusing to export untrained weights: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if checkpoint_epoch is not None:
        print(f"Loaded checkpoint successfully from {checkpoint_path} (epoch {checkpoint_epoch})")
    else:
        print(f"Loaded checkpoint successfully from {checkpoint_path}")

    model.eval()

    dummy_input = torch.randn(1, 4, 256, 256, dtype=torch.float32)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Exporting model to {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully exported and verified ONNX model: {output_path}")


if __name__ == "__main__":
    export()
