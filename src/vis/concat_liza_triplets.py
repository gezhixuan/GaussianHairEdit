#!/usr/bin/env python

import os
from PIL import Image

# ---- CONFIGURE THESE PATHS ----
BASE_DIR = "/scratch/hl106/zx_workspace/cto/VcEdit/gs_data"

INPUT_DIRS = [
    os.path.join(BASE_DIR, "Liza_new/images"),
    os.path.join(BASE_DIR, "Liza_llm_buzzcut/images"),
    os.path.join(BASE_DIR, "Liza_gemeni/images"),
]

OUTPUT_DIR = os.path.join(BASE_DIR, "Liza_concat/images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Background color for the big canvas (R, G, B, A)
BACKGROUND_COLOR = (0, 0, 0, 255)  # black; change if you want


def concat_triplet(img_paths, out_path, bg_color=BACKGROUND_COLOR):
    """
    Concatenate three images horizontally on a larger background
    without resizing the originals.
    """
    if len(img_paths) != 3:
        raise ValueError(f"Expected 3 image paths, got {len(img_paths)}")

    # Load as RGBA to handle PNG with alpha
    images = [Image.open(p).convert("RGBA") for p in img_paths]

    widths = [im.width for im in images]
    heights = [im.height for im in images]

    total_width = sum(widths)
    max_height = max(heights)

    # Create big canvas
    canvas = Image.new("RGBA", (total_width, max_height), bg_color)

    # Paste images side-by-side, vertically centered
    x_offset = 0
    for im in images:
        y_offset = (max_height - im.height) // 2
        canvas.paste(im, (x_offset, y_offset), im)
        x_offset += im.width

    canvas.save(out_path)


def main():
    # Use the filenames from the first directory as reference
    ref_dir = INPUT_DIRS[0]
    all_files = sorted(
        f for f in os.listdir(ref_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for fname in all_files:
        paths = [os.path.join(d, fname) for d in INPUT_DIRS]

        # Check all three exist
        if not all(os.path.exists(p) for p in paths):
            print(f"[WARN] Skipping {fname}: not found in all dirs")
            continue

        out_path = os.path.join(OUTPUT_DIR, fname)

        try:
            concat_triplet(paths, out_path)
            print(f"[OK] {fname} -> {out_path}")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")


if __name__ == "__main__":
    main()
