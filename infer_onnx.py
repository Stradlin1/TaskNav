from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_SOURCE = Path("/home/xhm/Desktop/TaskNav/test")

X_GRIDS = 160
ROW_ANCHORS = 56
NUM_TASKS = 4

Y_START = 1.0
Y_END = 0.3333333333

EXIST_THR = 0.5

POST_SMOOTH = True
POLY_DEGREE = 2
POLY_BLEND = 0.5

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 255, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="TaskNav LaneRobot ONNX inference and visualization."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to exported TaskNav .onnx model.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Image file or image directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <source>_onnx_pred",
    )
    parser.add_argument(
        "--exist-thr",
        type=float,
        default=EXIST_THR,
        help="No-lane probability threshold.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--no-post-smooth",
        action="store_true",
        help="Disable polynomial visualization smoothing.",
    )
    return parser.parse_args()


def collect_images(source: Path):
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file: {source}")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")

    files = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    if not files:
        raise FileNotFoundError(f"No images found in: {source}")

    return files


def create_session(model_path: Path, provider: str):
    available = ort.get_available_providers()

    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider requested, but it is unavailable. "
                f"Available providers: {available}"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]

    else:
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    return session


def infer_imgsz(session):
    inp = session.get_inputs()[0]
    shape = inp.shape

    if (
        len(shape) == 4
        and isinstance(shape[2], int)
        and isinstance(shape[3], int)
        and shape[2] == shape[3]
    ):
        return int(shape[2])

    return 640


def preprocess(image_path: Path, imgsz: int):
    """
    Match LaneRobotDataset:
      EXIF transpose
      RGB
      direct resize to imgsz x imgsz
      float32 / 255
      CHW
    """
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    original = image.copy()

    resized = image.resize(
        (imgsz, imgsz),
        Image.Resampling.BILINEAR,
    )

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.ascontiguousarray(arr[None])

    return original, arr


def softmax(x: np.ndarray, axis: int):
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)

    e = np.exp(x)

    return e / np.sum(e, axis=axis, keepdims=True)


def poly_smooth_1d(xs, valid, degree=2, blend=0.5):
    xs = xs.astype(np.float32).copy()
    valid = valid.astype(bool)

    if valid.sum() < degree + 1:
        return xs

    rows = np.arange(xs.shape[0], dtype=np.float32)

    try:
        coef = np.polyfit(
            rows[valid],
            xs[valid],
            degree,
        )

        fit = np.polyval(coef, rows).astype(np.float32)

        xs[valid] = (
            (1.0 - blend) * xs[valid]
            + blend * fit[valid]
        )

    except Exception:
        pass

    return xs


def decode(
    cls_logits: np.ndarray,
    offset: np.ndarray,
    exist_thr: float,
    post_smooth: bool,
):
    """
    Current TaskNav V2 decode contract:

      class = argmax(cls_logits[:X_GRIDS])
      x = class + signed_offset
      valid = P(no_lane) < exist_thr

    Inputs:
      cls_logits [1, 161, 56, 4]
      offset     [1,   1, 56, 4]

    Returns:
      pred_x     [56, 4]
      no_lane_p  [56, 4]
    """

    if cls_logits.shape != (
        1,
        X_GRIDS + 1,
        ROW_ANCHORS,
        NUM_TASKS,
    ):
        raise RuntimeError(
            "Unexpected cls_logits shape: "
            f"{cls_logits.shape}, expected "
            f"(1, {X_GRIDS + 1}, {ROW_ANCHORS}, {NUM_TASKS})"
        )

    if offset.shape != (
        1,
        1,
        ROW_ANCHORS,
        NUM_TASKS,
    ):
        raise RuntimeError(
            "Unexpected offset shape: "
            f"{offset.shape}, expected "
            f"(1, 1, {ROW_ANCHORS}, {NUM_TASKS})"
        )

    visible_logits = cls_logits[:, :X_GRIDS, :, :]

    cls_id = np.argmax(
        visible_logits,
        axis=1,
    ).astype(np.float32)

    off = np.clip(
        offset[:, 0, :, :],
        -0.5,
        0.5,
    )

    pred_x = cls_id + off

    probs = softmax(
        cls_logits,
        axis=1,
    )

    no_lane_p = probs[:, X_GRIDS, :, :]

    valid = no_lane_p < exist_thr

    pred_x = pred_x[0]
    valid = valid[0]
    no_lane_p = no_lane_p[0]

    pred_x[~valid] = -1.0

    if post_smooth:
        for task in range(NUM_TASKS):
            task_valid = pred_x[:, task] >= 0

            pred_x[:, task] = poly_smooth_1d(
                pred_x[:, task],
                task_valid,
                degree=POLY_DEGREE,
                blend=POLY_BLEND,
            )

            pred_x[~task_valid, task] = -1.0

    return pred_x, no_lane_p


def draw_prediction(
    image: Image.Image,
    pred_x: np.ndarray,
):
    vis = image.copy()

    draw = ImageDraw.Draw(vis)

    width, height = vis.size

    ys = np.linspace(
        Y_START,
        Y_END,
        ROW_ANCHORS,
        dtype=np.float32,
    )

    radius = max(2, round(min(width, height) / 250))
    line_width = max(2, round(min(width, height) / 300))

    for task in range(NUM_TASKS):
        color = COLORS[task % len(COLORS)]

        points = []

        for row in range(ROW_ANCHORS):
            x_grid = float(pred_x[row, task])

            if x_grid < 0:
                continue

            x_grid = float(
                np.clip(
                    x_grid,
                    0.0,
                    X_GRIDS - 1.0,
                )
            )

            x_norm = x_grid / (X_GRIDS - 1.0)

            x = int(
                round(
                    x_norm * (width - 1)
                )
            )

            y = int(
                round(
                    ys[row] * (height - 1)
                )
            )

            points.append((x, y))

            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ),
                fill=color,
            )

        if len(points) >= 2:
            draw.line(
                points,
                fill=color,
                width=line_width,
            )

        draw.text(
            (10, 10 + task * 22),
            f"task_{task}: {len(points)} rows",
            fill=color,
        )

    return vis


def run_one(
    session,
    image_path: Path,
    output_path: Path,
    imgsz: int,
    exist_thr: float,
    post_smooth: bool,
):
    original, inp = preprocess(
        image_path,
        imgsz,
    )

    input_name = session.get_inputs()[0].name

    output_names = [
        item.name
        for item in session.get_outputs()
    ]

    outputs = session.run(
        output_names,
        {
            input_name: inp,
        },
    )

    if output_names == [
        "cls_logits",
        "offset",
    ]:
        cls_logits, offset = outputs

    elif len(outputs) == 2:
        cls_logits, offset = outputs

    else:
        raise RuntimeError(
            f"Expected 2 outputs, got: {output_names}"
        )

    pred_x, no_lane_p = decode(
        cls_logits,
        offset,
        exist_thr=exist_thr,
        post_smooth=post_smooth,
    )

    vis = draw_prediction(
        original,
        pred_x,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vis.save(output_path)

    visible_counts = [
        int(
            np.sum(
                pred_x[:, task] >= 0
            )
        )
        for task in range(NUM_TASKS)
    ]

    return visible_counts


def main():
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    if args.output is None:
        if source.is_dir():
            output_dir = source.parent / (
                source.name + "_onnx_pred"
            )
        else:
            output_dir = source.parent / (
                source.stem + "_onnx_pred"
            )
    else:
        output_dir = (
            args.output
            .expanduser()
            .resolve()
        )

    images = collect_images(source)

    print(
        f"ONNX Runtime: {ort.__version__}"
    )

    print(
        "Available providers:",
        ort.get_available_providers(),
    )

    session = create_session(
        model_path,
        args.provider,
    )

    print(
        "Session providers:",
        session.get_providers(),
    )

    print(
        "Input:",
        session.get_inputs()[0].name,
        session.get_inputs()[0].shape,
    )

    print(
        "Outputs:",
        [
            (x.name, x.shape)
            for x in session.get_outputs()
        ],
    )

    imgsz = infer_imgsz(session)

    print(
        f"Input size: {imgsz}x{imgsz}"
    )

    print(
        f"Images: {len(images)}"
    )

    print(
        f"Output: {output_dir}"
    )

    post_smooth = not args.no_post_smooth

    for i, image_path in enumerate(
        images,
        start=1,
    ):
        if source.is_dir():
            relative = image_path.relative_to(source)
            output_path = (
                output_dir
                / relative.parent
                / f"{relative.stem}_pred.jpg"
            )
        else:
            output_path = (
                output_dir
                / f"{image_path.stem}_pred.jpg"
            )

        counts = run_one(
            session=session,
            image_path=image_path,
            output_path=output_path,
            imgsz=imgsz,
            exist_thr=args.exist_thr,
            post_smooth=post_smooth,
        )

        print(
            f"[{i:04d}/{len(images):04d}] "
            f"{image_path.name} "
            f"visible_rows={counts}"
        )

    print()
    print(
        f"Finished. Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
