import os
import numpy as np
from PIL import Image, ImageOps
import torch
from omegaconf import OmegaConf

import gradio as gr
import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import cv2
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline
import argparse


# If StableHair is in another file, change this import accordingly:
from infer_full import StableHair




def get_square_crop_from_mask(img: Image.Image,
                              mask: Image.Image,
                              expand_ratio: float = 1.2,
                              min_side: int = 256):
    """
    Use the mask to find a good square crop region.

    - Finds the bounding box where mask > 0.
    - Expands it a bit (expand_ratio).
    - Makes it square.
    - If crop goes out of image bounds, pads the image (expand) and updates coordinates.

    Returns:
        img_padded: possibly padded RGB image
        mask_padded: padded L mask
        crop_box: (x0, y0, x1, y1) in padded coordinates (square)
        pads: (pad_left, pad_top, pad_right, pad_bottom)
    """
    W, H = img.size
    mask_np = np.array(mask)

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        # Fallback: center square crop on the full image
        side = min(W, H)
        x0 = (W - side) // 2
        y0 = (H - side) // 2
        x1 = x0 + side
        y1 = y0 + side
        return img, mask, (x0, y0, x1, y1), (0, 0, 0, 0)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    bw = x_max - x_min
    bh = y_max - y_min

    side = int(max(bw, bh) * expand_ratio)
    side = max(side, min_side)

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = x0 + side
    y1 = y0 + side

    # Compute padding if crop goes out of bounds
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    if pad_left or pad_top or pad_right or pad_bottom:
        img_padded = ImageOps.expand(
            img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
        )
        mask_padded = ImageOps.expand(
            mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
        )
    else:
        img_padded = img
        mask_padded = mask

    # Shift crop box into padded coordinates
    x0 += pad_left
    y0 += pad_top
    x1 += pad_left
    y1 += pad_top

    return img_padded, mask_padded, (x0, y0, x1, y1), (pad_left, pad_top, pad_right, pad_bottom)


def numpy_to_pil_from_sample(sample):
    """
    Convert the StableHair 'sample' output into a PIL image.
    sample is expected to be float in [0, 1], shape (H, W, 3).
    """
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()

    sample = np.array(sample)
    if sample.ndim == 4:
        # assume (N, H, W, C) – take first
        sample = sample[0]
    sample = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(sample)


def run_batch_stable_hair(
        base_dir="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Liza_stable",
        ref_image="/scratch/hl106/zx_workspace/cto/VcEdit/ref_images/buzz_cut.png",
        config_path="./configs/hair_transfer.yaml",
        output_subdir="stable_hair_buzz_cut"):

    images_dir = os.path.join(base_dir, "images")
    alpha_dir = os.path.join(base_dir, "alpha")
    out_dir = os.path.join(base_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Optional temp dir for debug crops (not strictly needed, but handy)
    tmp_source_dir = os.path.join(out_dir, "_crops")
    os.makedirs(tmp_source_dir, exist_ok=True)

    # Initialize StableHair model
    model = StableHair(config=config_path, weight_dtype=torch.float16)
    kwargs = OmegaConf.to_container(model.config.inference_kwargs)

    # Force the reference image to the one you want
    kwargs["reference_image"] = ref_image

    image_names = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
    )

    print(f"Found {len(image_names)} frames.")

    for idx, fname in enumerate(image_names):
        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(alpha_dir, fname)
        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found for {fname}, skipping.")
            continue

        print(f"[{idx+1}/{len(image_names)}] Processing {fname}...")

        # Load original frame and mask
        img_full = Image.open(img_path).convert("RGB")
        mask_full = Image.open(mask_path).convert("L")
        W, H = img_full.size

        # 1) Use mask to get a square crop (with possible padding)
        img_pad, mask_pad, crop_box, pads = get_square_crop_from_mask(
            img_full, mask_full, expand_ratio=1.2, min_side=256
        )
        x0, y0, x1, y1 = crop_box
        crop_img = img_pad.crop(crop_box)
        crop_mask = mask_pad.crop(crop_box)
        crop_w, crop_h = crop_img.size

        # Make sure crop is perfectly square
        if crop_w != crop_h:
            side = max(crop_w, crop_h)
            crop_img = crop_img.resize((side, side), Image.BICUBIC)
            crop_mask = crop_mask.resize((side, side), Image.NEAREST)
            crop_w = crop_h = side

        # 2) Save crop to a temp file and run StableHair on it
        tmp_source_path = os.path.join(tmp_source_dir, fname)
        crop_img.save(tmp_source_path)

        kwargs["source_image"] = tmp_source_path
        # You can tweak the seed per frame if you want temporal variety:
        # kwargs["random_seed"] = kwargs.get("random_seed", 42) + idx

        _, sample, _, _ = model.Hair_Transfer(**kwargs)
        gen_patch = numpy_to_pil_from_sample(sample)

        # Resize generated patch back to crop size
        gen_patch = gen_patch.resize((crop_w, crop_h), Image.BICUBIC)

        # 3) Paste the generated patch back into the padded image
        result_pad = img_pad.copy()

        # Use the (cropped) mask as alpha so only the masked region is replaced
        alpha = crop_mask.point(lambda v: 255 if v > 0 else 0)
        result_pad.paste(gen_patch, box=(x0, y0), mask=alpha)

        # 4) Unpad back to original resolution
        pad_left, pad_top, pad_right, pad_bottom = pads
        final_img = result_pad.crop(
            (pad_left, pad_top, pad_left + W, pad_top + H)
        )

        # 5) Save result
        out_path = os.path.join(out_dir, fname)
        final_img.save(out_path)
        print(f"  -> saved to {out_path}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Base directory containing the results.",
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        help="Path to the reference image.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/hair_transfer.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="stable_hair_buzz_cut",
        help="Name of the output subdirectory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_batch_stable_hair(
        base_dir=args.base_dir,
        ref_image=args.ref_image,
        config_path=args.config_path,
        output_subdir=args.output_subdir,
    )
