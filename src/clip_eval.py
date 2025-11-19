import os
import torch
import clip
from PIL import Image
import pandas as pd

# 1. CONFIG
IMAGE_DIR = "/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_buzz cut_registered/images"  # <- change this
OUTPUT_CSV = "clip_scores_two_prompts.csv"

# # Text prompts
# POSITIVE_PROMPT = "a person with a buzz cut hairstyle"          # positive: default buzz cut
# NEGATIVE_PROMPT = "a person with long hair in a ponytail"      # negative: with long hair in ponytail
# Text prompts
# POSITIVE_PROMPT = "buzz cut"          # positive: default buzz cut
# NEGATIVE_PROMPT = "long hair in a ponytail"      # negative: with long hair in ponytail

POSITIVE_PROMPT = "a person with buzz cut"          # positive: default buzz cut
NEGATIVE_PROMPT = "a person with buzz cut with long hair in a ponytail"      # negative: with long hair in ponytail
# Text prompts
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

def score_images_with_two_prompts(image_folder,
                                  positive_prompt,
                                  negative_prompt,
                                  output_csv=None):
    model, preprocess = load_clip_model()
    image_paths = get_image_files(image_folder)

    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    # 2. Encode both text prompts
    with torch.no_grad():
        text_tokens = clip.tokenize([positive_prompt, negative_prompt]).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features shape: [2, D]

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
            # image_features shape: [1, D]

            # 3. Cosine similarities: image vs both prompts
            # result shape: [1, 2] -> squeeze to [2]
            sims = (image_features @ text_features.T).squeeze(0).tolist()

        pos_score = float(sims[0])
        neg_score = float(sims[1])
        margin = pos_score - neg_score  # optional: how much more "buzz cut" than "long ponytail"
        image_name = img_path.split("/")[-1]
        results.append({
            # "image_path": img_path,
            "image_name": image_name,
            "score_positive": pos_score,
            "score_negative": neg_score,
            "margin_pos_minus_neg": margin,
        })

        print(
            f"{image_name} | "
            f"pos (buzz cut): {pos_score:.4f} | "
            f"neg (long ponytail): {neg_score:.4f} | "
            f"margin: {margin:.4f}"
        )

    # 4. Save to CSV if requested
    if output_csv is not None and results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved scores to {output_csv}")

if __name__ == "__main__":
    score_images_with_two_prompts(
        IMAGE_DIR,
        POSITIVE_PROMPT,
        NEGATIVE_PROMPT,
        OUTPUT_CSV
    )
