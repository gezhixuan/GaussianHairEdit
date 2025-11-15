from openai import OpenAI
from PIL import Image
import base64
import io

client = OpenAI()

# Base image prompt: "A photorealistic picture of a fluffy ginger cat sitting on a wooden floor,
# looking directly at the camera. Soft, natural light from a window."
image_path = "/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/Zhixuan/input/000004.png"
text_input = "Using the provided image, please change the hairstyle to buzz cut."

# Read and base64-encode the input image as a data URL
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")
image_data_url = f"data:image/png;base64,{image_b64}"

# Call the Responses API with the image_generation tool
response = client.responses.create(
    model="gpt-5",  # or another supported model, e.g. "gpt-4.1"
    # model="gpt-4.1",  # or another supported model, e.g. "gpt-4.1"
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text_input},
                {
                    "type": "input_image",
                    "image_url": image_data_url,
                },
            ],
        }
    ],
    tools=[{"type": "image_generation"}],
)

# Extract the generated image (base64-encoded)
image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    edited_image_b64 = image_data[0]
    edited_image_bytes = base64.b64decode(edited_image_b64)

    # Load into PIL, save, and show
    edited_image = Image.open(io.BytesIO(edited_image_bytes))
    edited_image.save("cat_buzzcut.png")
    edited_image.show()
