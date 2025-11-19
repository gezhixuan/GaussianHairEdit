import cv2
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize + affine-register Gemini images to originals."
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
        help="Directory to save aligned images (e.g. scene/images in a copy).",
    )
    return parser.parse_args()


def align_images_affine(reference_stack, moving_stack):
    """
    Aligns images from moving_stack to reference_stack using SIFT + affine RANSAC.
    """
    aligned_stack = []
    affine_matrices = []

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)

    for ref_img, mov_img in zip(reference_stack, moving_stack):
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        mov_gray = cv2.cvtColor(mov_img, cv2.COLOR_BGR2GRAY)

        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
        kp_mov, des_mov = sift.detectAndCompute(mov_gray, None)

        if des_ref is None or des_mov is None:
            print("Warning: No descriptors found in one of the images. Using identity.")
            aligned_stack.append(mov_img)
            affine_matrices.append(np.array([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]]))
            continue

        matches = bf.knnMatch(des_mov, des_ref, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        min_match_count = 3
        if len(good_matches) > min_match_count:
            src_pts = np.float32(
                [kp_mov[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_ref[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            M, mask = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
            )

            if M is None:
                print("Warning: Affine estimation failed. Using identity.")
                M = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]])
                aligned_img = mov_img
            else:
                height, width = ref_img.shape[:2]
                aligned_img = cv2.warpAffine(mov_img, M, (width, height))
        else:
            print(
                f"Warning: Not enough matches found ({len(good_matches)}/{min_match_count}). "
                "Using identity transform."
            )
            M = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
            aligned_img = mov_img

        aligned_stack.append(aligned_img)
        affine_matrices.append(M)

    return aligned_stack, affine_matrices


def main():
    args = parse_args()

    original_path = args.original_dir
    gemini_path = args.gemini_dir
    gemini_resized_path = args.resized_dir
    aligned_output_path = args.aligned_dir

    # Basic existence checks
    if not os.path.isdir(original_path) or not os.path.isdir(gemini_path):
        raise RuntimeError(
            f"Could not find original_dir={original_path} or gemini_dir={gemini_path}"
        )

    os.makedirs(gemini_resized_path, exist_ok=True)
    os.makedirs(aligned_output_path, exist_ok=True)

    ori_stack, gemini_stack = [], []
    ori_filenames = []

    # Sorted lists to keep correspondence
    ori_files_sorted = sorted(os.listdir(original_path))
    gem_files_sorted = sorted(os.listdir(gemini_path))

    # Use zip so we only go up to min(len(ori), len(gem))
    for ori_file, gem_file in zip(ori_files_sorted, gem_files_sorted):
        ori_image_path = os.path.join(original_path, ori_file)
        gem_image_path = os.path.join(gemini_path, gem_file)

        ori_image = cv2.imread(ori_image_path)
        gem_image = cv2.imread(gem_image_path)

        if ori_image is None or gem_image is None:
            print(f"Warning: Could not read {ori_file} or {gem_file}. Skipping.")
            continue

        # Resize Gemini image to match original dimensions
        gem_image_resized = cv2.resize(
            gem_image, (ori_image.shape[1], ori_image.shape[0])
        )

        # Save resized Gemini image beside images/ in resized_images/
        resized_save_path = os.path.join(gemini_resized_path, ori_file)
        cv2.imwrite(resized_save_path, gem_image_resized)

        ori_stack.append(ori_image)
        gemini_stack.append(gem_image_resized)
        ori_filenames.append(ori_file)

    print(f"Loaded {len(ori_stack)} image pairs for registration.")

    if not ori_stack or not gemini_stack:
        print("No valid image pairs loaded, aborting registration.")
        return

    print("\nStarting affine registration...")
    aligned_gemini_stack, alignment_matrices = align_images_affine(
        ori_stack, gemini_stack
    )
    print("Registration complete.")
    print(f"Saving aligned images to {aligned_output_path}")

    for filename, aligned_img, M in zip(
        ori_filenames, aligned_gemini_stack, alignment_matrices
    ):
        output_path = os.path.join(aligned_output_path, filename)
        cv2.imwrite(output_path, aligned_img)

    print("Done.")


if __name__ == "__main__":
    main()
