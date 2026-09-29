#!/usr/bin/env python
"""TextGaze single-image inference demo.

Runs gaze target estimation for one head bounding box in one image, optionally
conditioned on a textual scene cue (the core of TextGaze), and saves a heatmap
overlay.

Example:
    python inference_demo.py \
        --image assets/qualitative/sample_1_00002049_1_gt_annotation.jpg \
        --bbox 0.60 0.33 0.66 0.43 \
        --text "Several students at computers; the foreground student gazes at the woman on the right." \
        --ckpt <gf_checkpoint>.pt \
        --arch gazelle_dinov2_vitl14 \
        --out overlay.jpg
"""
import argparse

import numpy as np
import torch
from PIL import Image

from gazefuze.model import get_gazelle_model


def parse_args():
    ap = argparse.ArgumentParser(description="TextGaze single-image demo")
    ap.add_argument("--image", type=str, required=True, help="path to the scene image")
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="head bbox, normalized to [0,1] (x1 y1 x2 y2)")
    ap.add_argument("--text", type=str, default="",
                    help="textual scene cue (Qwen2.5-VL style caption); empty = no cue")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt path")
    ap.add_argument("--arch", type=str, default="gazelle_dinov2_vitl14",
                    help="model arch (append _inout for in/out-of-frame models)")
    ap.add_argument("--out", type=str, default="overlay.jpg", help="output overlay path")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model, transform = get_gazelle_model(args.arch)
    model.load_gazelle_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()

    pil = Image.open(args.image).convert("RGB")
    pred = model.forward({
        "images": transform(pil).unsqueeze(0).to(device),
        "bboxes": [[args.bbox]],                 # [img][person][4], normalized
        "texts": [args.text],                    # flattened, one per person
        "num_ppl_per_img": [1],
    })

    hm = pred["heatmap"][0][0]                   # 64x64 tensor for head #0
    peak_idx = hm.flatten().argmax().item()
    peak_y, peak_x = np.unravel_index(peak_idx, hm.shape)
    W, H = pil.size
    print(f"peak={hm.max():.3f} at ({peak_x / hm.shape[1]:.3f}, {peak_y / hm.shape[0]:.3f}) "
          f"-> pixel ({int(peak_x / hm.shape[1] * W)}, {int(peak_y / hm.shape[0] * H)})")
    if "inout" in pred:
        print(f"in/out prob: {pred['inout'][0][0].item():.3f}")

    # overlay: upsample heatmap to image size, blend in jet-like colormap
    hm_up = torch.nn.functional.interpolate(
        hm[None, None], size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    hm_up = (hm_up - hm_up.min()) / (hm_up.max() - hm_up.min() + 1e-8)
    cmap = (np.stack([hm_up * 2, np.clip(hm_up * 2 - 1, 0, 1),
                      np.clip(hm_up * 4 - 3, 0, 1)], axis=-1) * 255).astype(np.uint8)
    overlay = (np.asarray(pil) * 0.55 + cmap * 0.45).astype(np.uint8)
    Image.fromarray(overlay).save(args.out, quality=90)
    print(f"saved overlay -> {args.out}")


if __name__ == "__main__":
    main()
