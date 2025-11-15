from google import genai
from google.genai import types
from PIL import Image
import base64
import io



client = genai.Client(api_key="AIzaSyCrNtuXfYmlQpP_ys42P4VJ4JVTf8-6Cu4")

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)


# Base image prompt: "A photorealistic picture of a fluffy ginger cat sitting on a wooden floor, looking directly at the camera. Soft, natural light from a window."
image_input = Image.open('/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/Zhixuan/input/000004.png')
# text_input = """Using the provided image, please change the hairstyle to buzz cut."""
text_input = "Using the cartoon hair icon only as a style reference, change the man's hairstyle to a realistic version of that haircut while keeping his face and everything else in the image unchanged, rather than copying the icon itself."


# Generate an image from a text + image prompt
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[text_input, image_input],
)


# Grab the first image in the response and save it
for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        # SDK helper: converts inline_data -> PIL.Image.Image
        image = part.as_image()
        image.save("cat_buzzcut.png")
        # image.show()
        break