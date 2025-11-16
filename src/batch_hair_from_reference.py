#!/usr/bin/env python3
import os
import io
import argparse
from glob import glob

from google import genai
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-edit images with Gemini by matching the hairstyle in a "
            "single reference image. Overwrites images in-place."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing images to edit.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="Image extension to process (e.g. png, jpg). Default: png",
    )
    parser.add_argument(
        "--ref-image",
        type=str,
        required=True,
        help="Path to the reference hairstyle image.",
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
        help="Gemini image model name. Default: gemini-2.5-flash-image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- API key check ---
    if not args.api_key:
        raise RuntimeError(
            "No API key provided. Set --api-key or export GOOGLE_API_KEY."
        )

    client = genai.Client(api_key=args.api_key)

    # --- Load reference hairstyle image once ---
    if not os.path.isfile(args.ref_image):
        raise FileNotFoundError(f"Reference image not found: {args.ref_image}")
    ref_image = Image.open(args.ref_image)

    # --- Instruction: first image = person, second = reference hairstyle ---
    # text_input = "The first image shows a person and the second image is a hairstyle reference. Modify only the person's hair in the first image to closely match the hairstyle, color, and texture of the second image, while keeping the face, expression, body, clothing, lighting, and background unchanged, and output the edited first image."
    # text_input = """Take the first image of the person. Use the second image as a hairstyle reference. Change only the person's hair in the first image to match the style, length, and color from the second image. Ensure the face, expression, body, clothing, lighting, and background remain completely unchanged. The new hair should look naturally integrated, matching the lighting, shadows, and perspective of the first image."""
    text_input = "change only the person's hair in the first image to a realistic version of the hairstyle in the second image and keep everything else exactly the same and output it"

    # --- Collect target images ---
    pattern = os.path.join(args.image_dir, f"*.{args.ext}")
    image_paths = sorted(glob(pattern))

    if not image_paths:
        raise RuntimeError(f"No *.{args.ext} images found in {args.image_dir}")

    # Avoid accidentally editing the reference image if it lives in the same dir
    ref_abs = os.path.abspath(args.ref_image)
    image_paths = [p for p in image_paths if os.path.abspath(p) != ref_abs]

    if not image_paths:
        raise RuntimeError(
            "After removing the reference image from the list, no images remain to edit."
        )

    to_process = image_paths[: args.max_requests]

    print(
        f"Found {len(image_paths)} images (excluding reference), "
        f"processing {len(to_process)} (max_requests={args.max_requests})"
    )

    # --- Loop over images and overwrite in-place ---
    for img_path in tqdm(to_process, desc="Editing images", unit="img"):
        try:
            person_image = Image.open(img_path)

            # Call Gemini: [person_image, ref_image, text_prompt] -> edited image
            response = client.models.generate_content(
                model=args.model,
                contents=[person_image, ref_image, text_input],
            )

            edited_image = None

            # Extract edited image from response (manual inline_data handling
            # like in script 2)
            ind = 0
            # for part in response.candidates[0].content.parts:
            #     if part.text is not None:
            #         print(part.text)
            #     elif part.inline_data is not None:
            #         print(f"Received {part.inline_data.mime_type} data")
            #         edited_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            #         edited_image.save(img_path.replace(".png",f"_{ind}.png"))
            #         ind +=1     
                    
                    
            for part in response.parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None) is not None:
                    edited_image = Image.open(io.BytesIO(inline.data)).convert("RGB")
                    # edited_image.save(img_path.replace(".png",f"_{ind}.png"))
                    ind +=1

                    break

            if edited_image is None:
                print(f"\n[WARN] No image returned for {img_path}, skipping.")
                continue

            # Overwrite original image
            edited_image.save(img_path)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_path}: {e}")


if __name__ == "__main__":
    main()
