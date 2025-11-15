from google import genai
from google.genai import types
from PIL import Image
import io
import cairosvg
# Render SVG to PNG bytes
client = genai.Client(api_key="AIzaSyCrNtuXfYmlQpP_ys42P4VJ4JVTf8-6Cu4")

man_image = Image.open('/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/Zhixuan/input/000004.png')
svg_path =  "'/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/icons/1/hairstyle-female-svgrepo-com.svg'"
png_bytes = cairosvg.svg2png(url=svg_path)

# Load PNG bytes into Pillow
hair_image = Image.open(io.BytesIO(png_bytes))
# hair_image = Image.open('/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/icons/1/hairstyle-female-svgrepo-com.svg')

# text_input = """Change only the man's hairstyle to match the hairstyle represented by the hair icon, using it as a reference, while keeping his face and everything else in the image unchanged."""
text_input = "The first image is a real man and the second image is a cartoon hair icon; change only the man's hairstyle in the first image to a realistic version of the hairstyle represented by the cartoon icon, while keeping his face and everything else in the image unchanged."


# # Generate an image from a text prompt
# response = client.models.generate_content(
#     model="gemini-2.5-flash-image",
#     contents=[man_image, hair_image, text_input],
# )

# # Grab the first image in the response and save it
# for part in response.parts:
#     if part.text is not None:
#         print(part.text)
#     elif part.inline_data is not None:
#         # SDK helper: converts inline_data -> PIL.Image.Image
#         image = part.as_image()
#         image.save("cat_buzzcut.png")
#         # image.show()
        # break
        
# === code below is new ===

# Generate an image from a text+image prompt
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
    raise RuntimeError("No image returned by Gemini.")

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

top_with_icon = top_resized.copy()
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

# Save ONLY the final combined image (do not save the raw Gemini output)
combined.save("combined_output.png")
# combined.show()  # optional preview
