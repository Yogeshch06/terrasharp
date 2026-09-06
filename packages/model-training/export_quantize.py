import os
import argparse
import yaml
from onnxruntime.quantization import quantize_dynamic, QuantType


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic INT8 Quantization for TerraSharp ONNX Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Input FP32 ONNX path")
    parser.add_argument("--output", type=str, default=None, help="Output INT8 ONNX path")
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def quantize():
    args = parse_args()
    config = load_config(args.config)
    e_cfg = config.get("export", {})

    input_path = args.input or e_cfg.get("onnx_path", "terrasharp_swinir_sentinel2.onnx")
    output_path = args.output or e_cfg.get("quant_path", "terrasharp_swinir_sentinel2_int8.onnx")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input ONNX model {input_path} not found. Run export_onnx.py first.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Quantizing {input_path} -> {output_path} (INT8 dynamic)...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original size: {orig_size:.2f} MB | Quantized size: {quant_size:.2f} MB")
    print(f"Successfully quantized model saved to: {output_path}")


if __name__ == "__main__":
    quantize()
