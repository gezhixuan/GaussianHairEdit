from google import genai
from google.genai import types
from PIL import Image
import io
import cairosvg
import os
import glob
from tqdm.auto import tqdm

# Render SVG to PNG bytes
client = genai.Client()

# man_image = Image.open('/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/Zhixuan/input/000004.png')

man_image = Image.open('/scratch/hl106/zx_workspace/cto/VcEdit/111/3xpampleimages/Image.png')
# Directory with hair icons (SVGs)
icons_dir = "/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/icons/1"
icon_paths = sorted(glob.glob(os.path.join(icons_dir, "*.svg")))
if not icon_paths:
    raise RuntimeError(f"No SVG icons found in: {icons_dir}")

# text_input = "The first image is a real man and the second image is a cartoon hair icon; change only the man's hairstyle in the first image to a realistic version of the hairstyle represented by the cartoon icon"
# # , while keeping his face and everything else in the image unchanged."

# text_input = "The first image is a photo of a man and the second image is a cartoon hairstyle reference, change only the man's hair in the first image to a realistic version of the hairstyle in the second image and keep everything else the same."
# text_input = "The first image is a photo of a man and the second image is a cartoon hair icon; change only the man's hair in the first image to a realistic version of the icon hairstyle and keep everything else the same."
text_input = "change only the man's hair in the first image to a realistic version of the icon hairstyle in the second image"# and keep everything else the same."



combined_images = []

# this will hold the "registered" original+edited image, created only on first iteration
registered_base = None

# Loop over all icons with tqdm
for svg_path in tqdm(icon_paths, desc="Processing icons"):
    # --- load icon as PNG ---
    png_bytes = cairosvg.svg2png(url=svg_path)
    hair_image = Image.open(io.BytesIO(png_bytes))

    # === Gemini call (unchanged logic, now inside loop) ===
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[man_image, hair_image, text_input],
    )

    edited_image = None

    # Extract edited image from response without using part.as_image()
    for part in response.parts:
        if getattr(part, "text", None) is not None:
            # If the model also returns text, you can see it here
            print(part.text)
        elif getattr(part, "inline_data", None) is not None:
            # inline_data.data is raw image bytes
            edited_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
            break

    if edited_image is None:
        raise RuntimeError(f"No image returned by Gemini for icon: {svg_path}")

    # After the first response from Gemini, generate a registered image of original + edited
    if registered_base is None:
        register_text = (
            "Register/align the first image to the second image spatially"
        )
        register_response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[man_image, edited_image, register_text],
        )

        for part in register_response.parts:
            if getattr(part, "text", None) is not None:
                # Optional: print any text Gemini returns for debugging
                print(part.text)
            elif getattr(part, "inline_data", None) is not None:
                registered_base = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
                break

        if registered_base is None:
            raise RuntimeError("No registered image returned by Gemini.")

    # Ensure original and icon are RGBA
    man_rgba = man_image.convert("RGBA")
    icon_rgba = hair_image.convert("RGBA")

    # 1. Crop the upper width*width region from the original image
    orig_w, orig_h = man_rgba.size
    side = min(orig_w, orig_h)  # keep it in-bounds even if width > height
    top_crop = man_rgba.crop((0, 0, side, side))  # (left, upper, right, lower)

    # 2. Resize that crop to the same size as the edited image
    edit_w, edit_h = edited_image.size
    top_resized = top_crop.resize((edit_w, edit_h), Image.LANCZOS)

    # 3. Scale the icon small and put it at bottom-right of the resized crop
    icon_scale = 0.2  # icon width as a fraction of the base width
    icon_w = int(edit_w * icon_scale)
    icon_h = int(icon_rgba.height * icon_w / icon_rgba.width)
    icon_small = icon_rgba.resize((icon_w, icon_h), Image.LANCZOS)

    margin = 10  # pixels from the edges
    icon_x = edit_w - icon_w - margin
    icon_y = edit_h - icon_h - margin

    # use the registered image (only created on first iteration) as the first row,
    # and add different icons to it for each iteration
    reg_resized = registered_base.resize((edit_w, edit_h), Image.LANCZOS)
    top_with_icon = reg_resized.copy()
    # Use icon_small as its own mask to keep transparency
    top_with_icon.paste(icon_small, (icon_x, icon_y), icon_small)

    # 4. Combine into a 1 column x 2 row big image:
    #    [ top_with_icon ]
    #    [  edited_image ]
    combined_w = edit_w
    combined_h = edit_h * 2

    combined = Image.new("RGBA", (combined_w, combined_h), (255, 255, 255, 255))
    combined.paste(top_with_icon, (0, 0))
    combined.paste(edited_image, (0, edit_h))

    # Optionally save each per-icon combined image
    base_name = os.path.splitext(os.path.basename(svg_path))[0]
    combined.save(f"combined_{base_name}.png")

    combined_images.append(combined)

# === combine all 1col x 2row images into one big image ===
if not combined_images:
    raise RuntimeError("No combined images were created.")

single_w, single_h = combined_images[0].size
n = len(combined_images)

# Here we put them side by side horizontally:
final_w = single_w * n
final_h = single_h

final_combined = Image.new("RGBA", (final_w, final_h), (255, 255, 255, 255))

for i, img in enumerate(combined_images):
    final_combined.paste(img, (i * single_w, 0))

# Save ONLY the big combined image (all icons)
final_combined.save("combined_output_all.png")
