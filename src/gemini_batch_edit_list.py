#!/usr/bin/env python3
import os
import argparse
from glob import glob

from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

# Set this to a list of filenames to restrict editing, e.g.:
# selected_list = [
#     "000280.png",
#     "000392.png",
#     "000400.png",
#     "000465.png",
#     "000506.png",
#     "000513.png",
#     "000538.png",
# ]
#
# If selected_list is None, ALL images in the directory will be edited.
selected_list = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-edit images with Gemini and overwrite them in-place."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/Liza_gemeni/images",
        help="Directory containing images to edit (default: Liza_gemeni/images)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="Image extension to process (e.g. png, jpg). Default: png",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=80,
        help="Maximum number of Gemini requests (to limit billing).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY", ""),
        help="Google Gemini API key. Defaults to $GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-image",
        # default="gemini-2.5-pro",
        help="Gemini image model name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="buzz with no braids",
        help='Hairstyle description to apply, e.g. "long natural curls". '
             'Default: "buzz with no braids".',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.api_key:
        raise RuntimeError(
            "No API key provided. Set --api-key or export GOOGLE_API_KEY."
        )

    client = genai.Client(api_key=args.api_key)

    # Build the instruction from the prompt
    text_input = (
        f"Using the provided image, please change the hairstyle to {args.prompt}."
    )
    print(text_input)

    pattern = os.path.join(args.image_dir, f"*.{args.ext}")
    image_paths = sorted(glob(pattern))

    if not image_paths:
        raise RuntimeError(f"No *.{args.ext} images found in {args.image_dir}")

    # Respect max_requests so you don’t blow through billing
    to_process = image_paths[: args.max_requests]

    print(
        f"Found {len(image_paths)} images, processing {len(to_process)} "
        f"(max_requests={args.max_requests})"
    )

    for img_path in tqdm(to_process, desc="Editing images", unit="img"):
        filename = os.path.basename(img_path)

        # If selected_list is not None, only process the ones in the list.
        if selected_list is not None and filename not in selected_list:
            continue

        try:
            # Open the original image
            image_input = Image.open(img_path)

            # Call Gemini with text + image
            response = client.models.generate_content(
                model=args.model,
                contents=[text_input, image_input],
            )

            # Find the first image in the response and overwrite the original
            edited_image = None
            for part in response.parts:
                # Some parts might be text (safety messages, etc.)
                if getattr(part, "inline_data", None) is not None:
                    edited_image = part.as_image()
                    break

            if edited_image is None:
                print(f"\n[WARN] No image returned for {img_path}, skipping.")
                continue

            # Overwrite the original file
            edited_image.save(img_path)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_path}: {e}")


if __name__ == "__main__":
    main()
