import os
import torch
import clip
from PIL import Image
import pandas as pd

# 1. CONFIG
# IMAGE_DIR = "/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/Liza_gemeni_long/images"  # <- change this
IMAGE_DIR = "/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/Liza_gemeni/images"  # <- change this
OUTPUT_CSV = "clip_scores.csv"
# TEXT_PROMPT = "a woman with long natural curls hair style"
TEXT_PROMPT = "a woman with buzz cut hairstyle"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model(device=DEVICE):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

def get_image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f.lower())[1] in exts
    ]

def score_images_with_clip(image_folder, text_prompt, output_csv=None):
    model, preprocess = load_clip_model()
    image_paths = get_image_files(image_folder)

    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    # 2. Encode text prompt
    with torch.no_grad():
        text_tokens = clip.tokenize([text_prompt]).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Could not open {img_path}: {e}")
            continue

        with torch.no_grad():
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 3. Cosine similarity between image and text (range roughly -1 to 1)
            similarity = (image_features @ text_features.T).squeeze().item()

        results.append({"image_path": img_path, "score": similarity})
        print(f"{img_path}: {similarity:.4f}")

    # 4. Save to CSV if requested
    if output_csv is not None and results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved scores to {output_csv}")

if __name__ == "__main__":
    score_images_with_clip(IMAGE_DIR, TEXT_PROMPT, OUTPUT_CSV)
