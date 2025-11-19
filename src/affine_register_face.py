import os
import cv2
import argparse
import numpy as np

import torch
from PIL import Image
import torchvision.transforms as transforms

from model import BiSeNet          # must be from face-parsing.PyTorch
from insightface.app import FaceAnalysis


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resize + face-align Gemini images to originals using InsightFace, "
            "then preserve non-hair regions via face parsing."
        )
    )
    parser.add_argument(
        "--original-dir",
        type=str,
        required=True,
        help="Directory containing original reference images (e.g. scene/images).",
    )
    parser.add_argument(
        "--gemini-dir",
        type=str,
        required=True,
        help="Directory containing Gemini-regenerated images (same filenames).",
    )
    parser.add_argument(
        "--resized-dir",
        type=str,
        required=True,
        help="Directory to save resized Gemini images (e.g. scene/resized_images).",
    )
    parser.add_argument(
        "--aligned-dir",
        type=str,
        required=True,
        help="Directory to save final aligned+composited images.",
    )
    parser.add_argument(
        "--faceparse-weights",
        type=str,
        default="/scratch/hl106/zx_workspace/cto/VcEdit/ext/face-parsing.PyTorch/res/cp/79999_iter.pth",
        help="Path to BiSeNet face-parsing checkpoint (79999_iter.pth).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------- #
# InsightFace-based face alignment
# ----------------------------------------------------------------------------- #

class InsightFaceAligner:
    """
    Aligns a 'moving' face image to a 'reference' face image using
    InsightFace / ArcFace-style 5-point landmarks and a similarity transform.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size=(640, 640), ctx_id: int = -1):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    @staticmethod
    def _pick_main_face(faces):
        """Choose the largest detected face."""
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    @staticmethod
    def _estimate_similarity(src_points, dst_points):
        """
        Estimate a 2x3 similarity/affine transform from src_points to dst_points.
        src_points, dst_points: (N, 2) float32 arrays.
        """
        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)

        if src.shape != dst.shape or src.shape[0] < 3:
            return None

        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        return M

    def align_pair(self, reference_bgr: np.ndarray, moving_bgr: np.ndarray):
        """
        Align moving_bgr to reference_bgr using main faces' 5 landmarks.
        Returns (aligned_image, transform_matrix_2x3).
        If alignment fails, returns (moving_bgr, identity_matrix).
        """
        h, w = reference_bgr.shape[:2]
        identity = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        ref_faces = self.app.get(reference_bgr)
        mov_faces = self.app.get(moving_bgr)

        ref_face = self._pick_main_face(ref_faces)
        mov_face = self._pick_main_face(mov_faces)

        if ref_face is None or mov_face is None:
            print("Warning: Could not detect face in one of the images. Using identity.")
            return moving_bgr, identity

        ref_kps = ref_face.kps.astype(np.float32)  # (5, 2)
        mov_kps = mov_face.kps.astype(np.float32)  # (5, 2)

        M = self._estimate_similarity(mov_kps, ref_kps)
        if M is None:
            print("Warning: Could not estimate similarity transform. Using identity.")
            return moving_bgr, identity

        aligned = cv2.warpAffine(moving_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
        return aligned, M


# ----------------------------------------------------------------------------- #
# BiSeNet-based face parsing for hair mask + compositing
# ----------------------------------------------------------------------------- #

class HairMaskGenerator:
    """
    Uses BiSeNet (face-parsing.PyTorch) to compute hair masks.
    Default label index for 'hair' is 17 in CelebAMask-HQ.
    """

    def __init__(self, weights_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"BiSeNet weights not found at {weights_path}. "
                f"Pass correct path via --faceparse-weights."
            )

        self.net = BiSeNet(n_classes=19).to(self.device).eval()
        state = torch.load(weights_path, map_location=self.device)
        self.net.load_state_dict(state)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    def _parse_logits(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Run BiSeNet on a BGR image and return parsing map (H, W) with class indices.
        Internally resizes to 512x512, then upscales back to the original size
        using nearest-neighbor.
        """
        h, w = bgr_img.shape[:2]

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((512, 512), Image.BILINEAR)
        tensor = self.to_tensor(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.net(tensor)[0]  # [1, n_classes, H, W]

        parsing_512 = (
            out.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .argmax(0)
        )  # (512, 512)

        # Resize parsing map back to (h, w)
        parsing = cv2.resize(
            parsing_512.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        return parsing

    def hair_mask(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Return a boolean hair mask (H, W) where True means 'hair' pixel.
        By default uses label==17. If you also want hat, OR with label==18.
        """
        parsing = self._parse_logits(bgr_img)
        hair = (parsing == 17)  # hair label in CelebAMask-HQ mapping
        # If you want hat included as 'hair region':
        # hair |= (parsing == 18)
        return hair

    def composite_preserve_non_hair(
        self,
        original_bgr: np.ndarray,
        aligned_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hair masks for original & aligned, take union.
        For pixels in union (hair region in either image): keep the aligned image.
        For other pixels: copy from original into aligned.

        Effect: hair from aligned image, everything else from original.
        """
        if original_bgr.shape[:2] != aligned_bgr.shape[:2]:
            raise ValueError("original and aligned images must have the same spatial size.")

        hair_orig = self.hair_mask(original_bgr)
        hair_aligned = self.hair_mask(aligned_bgr)
        hair_union = np.logical_or(hair_orig, hair_aligned)  # (H, W) bool

        result = aligned_bgr.copy()
        # For non-hair pixels (~union), copy from original
        result[~hair_union] = original_bgr[~hair_union]
        return result


# ----------------------------------------------------------------------------- #
# I/O helpers
# ----------------------------------------------------------------------------- #

def load_image_pairs(original_dir, gemini_dir, resized_dir):
    """
    Load image pairs, resize Gemini images to match originals,
    save the resized versions, and return stacks and filenames.
    """
    os.makedirs(resized_dir, exist_ok=True)

    ori_stack = []
    gemini_stack = []
    filenames = []

    ori_files_sorted = sorted(os.listdir(original_dir))
    gem_files_sorted = sorted(os.listdir(gemini_dir))

    for ori_file, gem_file in zip(ori_files_sorted, gem_files_sorted):
        ori_path = os.path.join(original_dir, ori_file)
        gem_path = os.path.join(gemini_dir, gem_file)

        ori_img = cv2.imread(ori_path)
        gem_img = cv2.imread(gem_path)

        if ori_img is None or gem_img is None:
            print(f"Warning: Could not read {ori_file} or {gem_file}. Skipping.")
            continue

        h, w = ori_img.shape[:2]
        gem_resized = cv2.resize(gem_img, (w, h), interpolation=cv2.INTER_LINEAR)

        resized_save_path = os.path.join(resized_dir, ori_file)
        cv2.imwrite(resized_save_path, gem_resized)

        ori_stack.append(ori_img)
        gemini_stack.append(gem_resized)
        filenames.append(ori_file)

    return ori_stack, gemini_stack, filenames


def align_images_face(reference_stack, moving_stack, aligner: InsightFaceAligner):
    """
    Align each moving image to its corresponding reference image using
    InsightFace-based face alignment.
    """
    aligned_stack = []
    matrices = []

    for idx, (ref_img, mov_img) in enumerate(zip(reference_stack, moving_stack)):
        aligned, M = aligner.align_pair(ref_img, mov_img)
        aligned_stack.append(aligned)
        matrices.append(M)
        print(f"Aligned pair {idx + 1}/{len(reference_stack)}")

    return aligned_stack, matrices


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #

def main():
    args = parse_args()

    original_path = args.original_dir
    gemini_path = args.gemini_dir
    gemini_resized_path = args.resized_dir
    aligned_output_path = args.aligned_dir
    faceparse_weights = args.faceparse_weights

    if not os.path.isdir(original_path) or not os.path.isdir(gemini_path):
        raise RuntimeError(
            f"Could not find original_dir={original_path} or gemini_dir={gemini_path}"
        )

    os.makedirs(gemini_resized_path, exist_ok=True)
    os.makedirs(aligned_output_path, exist_ok=True)

    print("Loading image pairs...")
    ori_stack, gemini_stack, ori_filenames = load_image_pairs(
        original_path, gemini_path, gemini_resized_path
    )

    if not ori_stack or not gemini_stack:
        print("No valid image pairs loaded, aborting.")
        return

    print(f"Loaded {len(ori_stack)} image pairs.")
    print("Initializing InsightFace aligner...")
    aligner = InsightFaceAligner()

    print("Initializing BiSeNet hair mask generator...")
    hair_gen = HairMaskGenerator(faceparse_weights)

    print("Starting face-based alignment...")
    aligned_stack, matrices = align_images_face(ori_stack, gemini_stack, aligner)
    print("Alignment complete.")

    print("Applying hair-union compositing (preserve non-hair from original)...")
    for filename, ori_img, aligned_img in zip(ori_filenames, ori_stack, aligned_stack):
        final_img = hair_gen.composite_preserve_non_hair(ori_img, aligned_img)
        out_path = os.path.join(aligned_output_path, filename)
        cv2.imwrite(out_path, final_img)

    print(f"Done. Final images saved in: {aligned_output_path}")


if __name__ == "__main__":
    main()
