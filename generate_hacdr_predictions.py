# generate_hacdr_combined_masks.py
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mmseg.apis import init_segmentor, inference_segmentor


def build_color_mask(pred_mask, palette):
    """
    Convert a 2D class-index mask into an RGB color mask.
    pred_mask: H x W array of class IDs
    palette: list of [R, G, B] colors
    """
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(palette):
        color_mask[pred_mask == class_id] = color

    return color_mask


def main():
    parser = argparse.ArgumentParser(description="Generate combined prediction masks with HACDR-Net")
    parser.add_argument("--config", type=str, required=True, help="Path to config .py")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--img_dir", type=str, required=True, help="Folder containing test images")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder to save combined masks")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"],
        help="Allowed image extensions"
    )
    parser.add_argument(
        "--save_color",
        action="store_true",
        help="Also save a color visualization of the combined mask"
    )

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # Always use the fixed palette to ensure correct colour mapping.
    # model.PALETTE from checkpoint metadata has incorrect ordering
    # and should not be used.
    palette = [
        [0, 0, 0],        # 0 = background
        [255, 255, 0],    # 1 = EX  (Hard Exudates)      → yellow
        [255, 0, 0],      # 2 = MA  (Microaneurysms)     → red
        [0, 0, 255],      # 3 = SE  (Soft Exudates)      → blue
        [0, 255, 0],      # 4 = HE  (Hemorrhages)        → green
    ]

    # Collect image files
    image_paths = []
    for ext in args.exts:
        image_paths.extend(sorted(img_dir.glob(f"*{ext}")))
        image_paths.extend(sorted(img_dir.glob(f"*{ext.upper()}")))

    # Remove duplicates
    seen = set()
    unique_paths = []
    for p in image_paths:
        if str(p) not in seen:
            unique_paths.append(p)
            seen.add(str(p))

    if not unique_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    print(f"Found {len(unique_paths)} images")

    for img_path in unique_paths:
        print(f"Processing {img_path.name}")

        result = inference_segmentor(model, str(img_path))
        pred_mask = result[0].astype(np.uint8)   # class-index mask

        stem = img_path.stem

        # Save combined class-index mask
        combined_mask_path = out_dir / f"{stem}_combined_pred.png"
        cv2.imwrite(str(combined_mask_path), pred_mask)

        # Optional color visualization
        if args.save_color:
            color_mask = build_color_mask(pred_mask, palette)
            color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            color_mask_path = out_dir / f"{stem}_combined_color.png"
            cv2.imwrite(str(color_mask_path), color_mask_bgr)

    print(f"\nDone. Combined masks saved to: {out_dir}")


if __name__ == "__main__":
    main()