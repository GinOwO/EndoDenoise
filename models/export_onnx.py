import torch
import argparse
from train import DnCNN

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--onnx_path", type=str, required=True)
parser.add_argument("--input_size", type=int, nargs=2, default=[1, 256])
args = parser.parse_args()

model = DnCNN()
model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, args.input_size[0], args.input_size[1])
torch.onnx.export(
    model,
    dummy_input,  # type: ignore
    args.onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {2: "height", 3: "width"},
        "output": {2: "height", 3: "width"},
    },
    opset_version=11,
)

print(f"Exported to {args.onnx_path}")
