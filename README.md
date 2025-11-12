# Object-detection-with-Transformers-
This repository includes some models with some innovations which is about Rtdetr

## Quick RT-DETR-R18 demo

The repository now ships with a minimal inference script that downloads the
pre-trained **RT-DETR R18** checkpoint and runs it on a local image.

1. Install the required runtime packages (PyTorch, torchvision, Pillow):

   ```bash
   pip install torch torchvision pillow
   ```

2. Run inference on an image:

   ```bash
   python scripts/test_rt_detr.py --image path/to/your/image.jpg --device cpu
   ```

   The script fetches the upstream weights via `torch.hub` on the first run,
   creates an annotated image (`rt_detr_result.jpg`), and stores the detection
   metadata as JSON (`rt_detr_result.json`).

To use an offline checkpoint, supply the `--weights` argument with a path to a
compatible RT-DETR R18 `.pth` file.
