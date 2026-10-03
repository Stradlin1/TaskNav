from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.lane.val import get_lane_head


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "lane" / "train" / "weights" / "best.pt"
REQUIRED_OPSET = 11


class LaneONNXWrapper(torch.nn.Module):
    """Expose stable, named classification and offset outputs for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images)
        if isinstance(output, dict):
            cls_logits = output.get("cls", output.get("cls_logits"))
            offset = output.get("offset", output.get("lane_offset"))
        elif isinstance(output, (tuple, list)) and len(output) >= 2:
            cls_logits, offset = output[:2]
        else:
            raise RuntimeError(f"Unexpected lane model output type: {type(output)}")
        if cls_logits is None or offset is None:
            keys = list(output) if isinstance(output, dict) else None
            raise RuntimeError(f"Lane model output does not contain cls/offset tensors; keys={keys}")
        return cls_logits, offset


def parse_args():
    parser = argparse.ArgumentParser(description="Export TaskNav LaneRobot to ONNX and verify ONNX Runtime parity.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Input Ultralytics .pt checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path. Defaults to the checkpoint path with an .onnx suffix.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument(
        "--opset",
        type=int,
        default=REQUIRED_OPSET,
        choices=[REQUIRED_OPSET],
        help=f"ONNX opset version. TaskNav deployment is fixed to opset {REQUIRED_OPSET}.",
    )
    parser.add_argument("--device", default="auto", help="Export device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Store weights in an ONNX sidecar. The default writes one self-contained ONNX file.",
    )
    parser.add_argument("--simplify", action="store_true", help="Run onnxsim after export (single-file mode only).")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument(
        "--verify-runtime",
        dest="verify_runtime",
        action="store_true",
        help="Require ONNX Runtime numerical parity (default).",
    )
    verify_group.add_argument(
        "--no-verify-runtime",
        dest="verify_runtime",
        action="store_false",
        help="Skip ONNX Runtime numerical parity verification.",
    )
    parser.set_defaults(verify_runtime=True)
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for ONNX Runtime parity.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for ONNX Runtime parity.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def validate_pytorch_contract(model, cls_logits, offset):
    """Check output shapes against runtime head metadata instead of fixed constants."""
    head = get_lane_head(model)
    batch = int(cls_logits.shape[0])
    expected_cls = (batch, head.x_grids + 1, head.row_anchors, head.num_lanes)
    expected_offset = (batch, 1, head.row_anchors, head.num_lanes)
    if tuple(cls_logits.shape) != expected_cls:
        raise RuntimeError(f"cls_logits shape mismatch: expected {expected_cls}, got {tuple(cls_logits.shape)}")
    if tuple(offset.shape) != expected_offset:
        raise RuntimeError(f"offset shape mismatch: expected {expected_offset}, got {tuple(offset.shape)}")
    return {
        "x_grids": int(head.x_grids),
        "row_anchors": int(head.row_anchors),
        "num_lanes": int(head.num_lanes),
        "cls_shape": expected_cls,
        "offset_shape": expected_offset,
    }


def simplify_onnx(output: Path):
    try:
        import onnx
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError("--simplify requires both onnx and onnxsim in the export environment.") from exc
    model = onnx.load(str(output))
    simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError("onnxsim validation failed.")
    onnx.save(simplified, str(output))


def check_onnx(output: Path, expect_external_data: bool):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX validation requires the onnx package.") from exc

    onnx.checker.check_model(str(output))
    model = onnx.load(str(output), load_external_data=False)

    main_opset = next(
        (
            int(item.version)
            for item in model.opset_import
            if item.domain in ("", "ai.onnx")
        ),
        None,
    )
    if main_opset != REQUIRED_OPSET:
        raise RuntimeError(
            f"ONNX opset mismatch: expected {REQUIRED_OPSET}, got {main_opset}."
        )

    external_locations = set()
    for initializer in model.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            metadata = {item.key: item.value for item in initializer.external_data}
            if metadata.get("location"):
                external_locations.add(metadata["location"])
    if expect_external_data and not external_locations:
        raise RuntimeError("--external-data was requested, but the exported model has no external initializers.")
    if not expect_external_data and external_locations:
        raise RuntimeError(f"Single-file export unexpectedly references external data: {sorted(external_locations)}")
    for location in external_locations:
        sidecar = output.parent / location
        if not sidecar.is_file():
            raise FileNotFoundError(f"ONNX external-data sidecar is missing: {sidecar}")
    return sorted(external_locations)


def verify_onnx_runtime(wrapper, output: Path, imgsz: int, device: torch.device, atol: float, rtol: float):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Runtime verification requires onnxruntime.") from exc

    # Compare ONNX Runtime CPU against PyTorch CPU.
    # This avoids treating normal CUDA-vs-CPU kernel differences as export errors.
    wrapper_cpu = wrapper.to("cpu").eval()

    generator = torch.Generator(device="cpu").manual_seed(0)
    verify_input = torch.rand((1, 3, imgsz, imgsz), generator=generator)

    with torch.no_grad():
        pytorch_outputs = tuple(t.detach().numpy() for t in wrapper_cpu(verify_input))

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    if output_names != ["cls_logits", "offset"]:
        raise RuntimeError(f"Unexpected ONNX output names/order: {output_names}")

    runtime_outputs = session.run(
        output_names,
        {input_name: verify_input.numpy()},
    )

    errors = {}
    for name, expected, actual in zip(output_names, pytorch_outputs, runtime_outputs):
        if expected.shape != actual.shape:
            raise RuntimeError(f"ONNX Runtime {name} shape mismatch: PyTorch {expected.shape}, ONNX {actual.shape}")
        if not np.isfinite(actual).all():
            raise RuntimeError(f"ONNX Runtime {name} contains NaN or Inf.")
        difference = np.abs(expected - actual)
        errors[name] = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean())}
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=f"ONNX mismatch in {name}")
    return errors


def main():
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    output = (args.output or weights.with_suffix(".onnx")).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    if args.imgsz < 1:
        raise ValueError(f"--imgsz must be positive, got {args.imgsz}")
    if args.simplify and args.external_data:
        raise ValueError("--simplify and --external-data cannot be combined; simplify a single-file model instead.")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    yolo = YOLO(str(weights))
    model = yolo.model.to(device).eval()
    wrapper = LaneONNXWrapper(model).to(device).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        cls_logits, offset = wrapper(dummy)
    contract = validate_pytorch_contract(model, cls_logits, offset)

    print(f"Checkpoint: {weights}")
    print(f"Device: {device}")
    print(f"Protocol: x_grids={contract['x_grids']}, rows={contract['row_anchors']}, tasks={contract['num_lanes']}")
    print(f"PyTorch outputs: cls={contract['cls_shape']}, offset={contract['offset_shape']}")

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["cls_logits", "offset"],
        dynamic_axes=None,
        dynamo=False,
        external_data=args.external_data,
    )
    if args.simplify:
        simplify_onnx(output)
    sidecars = check_onnx(output, expect_external_data=args.external_data)

    print(f"ONNX: {output} ({output.stat().st_size} bytes)")
    print(f"ONNX opset: {REQUIRED_OPSET}")
    if sidecars:
        print(f"External data: {[str(output.parent / path) for path in sidecars]}")
    else:
        print("External data: none (single-file ONNX)")

    if args.verify_runtime:
        errors = verify_onnx_runtime(wrapper, output, args.imgsz, device, args.atol, args.rtol)
        for name, values in errors.items():
            print(f"ONNX Runtime {name}: max_abs={values['max_abs']:.8g}, mean_abs={values['mean_abs']:.8g}")
        print(f"ONNX Runtime parity: PASSED (rtol={args.rtol:g}, atol={args.atol:g})")
    else:
        print("ONNX Runtime parity: SKIPPED")


if __name__ == "__main__":
    main()
