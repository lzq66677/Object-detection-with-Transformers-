"""Simple RT-DETR-R18 inference script.

This script builds a lightweight RT-DETR R18 model via ``torch.hub`` and
runs it on a local image.  The default behaviour downloads the
pre-trained COCO weights that ship with the upstream repository, but
custom checkpoints can also be loaded through ``--weights``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    import torchvision.transforms.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "torch and torchvision are required for this script. "
        "Please install them before running the demo."
    ) from exc


COCO_CLASSES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def load_model(device: torch.device, weights: Path | None = None) -> torch.nn.Module:
    """Create an RT-DETR R18 model and optionally load custom weights."""

    try:
        model = torch.hub.load(
            "lyuwenyu/RT-DETR",
            "rt_detr_r18vd",
            pretrained=weights is None,
            source="github",
        )
    except Exception as exc:  # pragma: no cover - hub requires network access
        raise RuntimeError(
            "Failed to load RT-DETR R18 from torch.hub. "
            "Please ensure you have internet access or provide ``--weights``."
        ) from exc
    model.eval().to(device)

    if weights is not None:
        checkpoint = torch.load(weights, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)

    return model


def preprocess(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert the PIL image into a tensor batch expected by RT-DETR."""

    tensor = F.to_tensor(image).unsqueeze(0)
    return tensor.to(device)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert centre-based boxes into corner based boxes."""

    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack((x0, y0, x1, y1), dim=-1)


def postprocess(
    outputs: Dict[str, torch.Tensor],
    image_size: Tuple[int, int],
    score_threshold: float,
) -> List[Dict[str, object]]:
    """Turn raw RT-DETR predictions into human readable detections."""

    pred_logits = outputs["pred_logits"].sigmoid()[0]
    pred_boxes = outputs["pred_boxes"][0]

    scores, labels = pred_logits.max(-1)
    keep = scores > score_threshold

    scores = scores[keep]
    labels = labels[keep]
    boxes = pred_boxes[keep]

    boxes = box_cxcywh_to_xyxy(boxes)
    img_w, img_h = image_size
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)

    detections: List[Dict[str, object]] = []
    for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        detections.append(
            {
                "label": COCO_CLASSES[label],
                "score": float(score),
                "bbox": [float(x) for x in box],
            }
        )

    detections.sort(key=lambda det: det["score"], reverse=True)
    return detections


def draw_detections(
    image: Image.Image,
    detections: Iterable[Dict[str, object]],
    score_threshold: float,
) -> Image.Image:
    """Overlay bounding boxes and labels on the image."""

    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for det in detections:
        if det["score"] < score_threshold:
            continue

        x0, y0, x1, y1 = det["bbox"]
        label = det["label"]
        score = det["score"]
        caption = f"{label}: {score:.2f}"

        draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
        text_size = draw.textbbox((x0, y0), caption, font=font)
        if text_size is not None:
            _, _, tw, th = text_size
            draw.rectangle((x0, y0 - th, x0 + tw, y0), fill="red")
            draw.text((x0, y0 - th), caption, fill="white", font=font)
        else:
            draw.text((x0, y0), caption, fill="red", font=font)

    return drawn


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    image_path = args.image.expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    weights_path = args.weights.expanduser() if args.weights else None
    model = load_model(device=device, weights=weights_path)

    inputs = preprocess(image, device)
    with torch.inference_mode():
        outputs = model(inputs)

    detections = postprocess(outputs, image.size, args.threshold)

    if args.save_json:
        json_path = Path(args.save_json).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(detections, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Detection metadata saved to {json_path}")

    if args.save_image:
        annotated = draw_detections(image, detections, args.threshold)
        output_path = Path(args.save_image).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path)
        print(f"Annotated image saved to {output_path}")

    print("Top detections:")
    for det in detections[: args.top_k]:
        print(
            f"  - {det['label']}: {det['score']:.3f}"
            f" @ [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RT-DETR R18 quick inference demo")
    parser.add_argument("--image", type=Path, required=True, help="Path to the local image file")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional path to a custom checkpoint. When omitted, the pretrained RT-DETR R18 weights are fetched via torch.hub.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the inference on (default: cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Score threshold used for filtering detections.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of detections to print to stdout.",
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=Path("rt_detr_result.jpg"),
        help="Where to store the annotated output image.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=Path("rt_detr_result.json"),
        help="Where to store detection metadata as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
