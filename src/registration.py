import cv2
import numpy as np
import os

# --- 1. Image Loading (Modified) ---

# Set paths
original_path = './data/ori/'
gemini_path = './data/gemini_regenerated/'
gemini_resized_path = './data/gemini_resized/' # <-- New folder for resized images
aligned_output_path = './data/gemini_aligned_regenerated/' # <-- Folder for final aligned images

ori_stack, gemini_stack = [], []
ori_filenames = [] # List to store original filenames

# Check if directories exist (basic check)
if not os.path.exists(original_path) or not os.path.exists(gemini_path):
    print(f"Error: Could not find data directories at {original_path} or {gemini_path}")
    # Create dummy data for the script to run without erroring
    print("Creating dummy image stacks for demonstration...")
    ori_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    M_dummy = cv2.getRotationMatrix2D((640//2, 480//2), 5, 1.1)
    gem_image = cv2.warpAffine(ori_image, M_dummy, (640, 480))
    gem_image = cv2.resize(gem_image, (ori_image.shape[1], ori_image.shape[0]))
    
    ori_stack.append(ori_image)
    gemini_stack.append(gem_image)
    ori_filenames.append("dummy_image_00.png")
else:
    # Create output directories if they don't exist
    os.makedirs(gemini_resized_path, exist_ok=True)
    os.makedirs(aligned_output_path, exist_ok=True)
    print(f"Saving resized images to {gemini_resized_path}")

    # Use sorted() to ensure corresponding files are paired
    ori_files_sorted = sorted(os.listdir(original_path))
    gem_files_sorted = sorted(os.listdir(gemini_path))

    for ori_file, gem_file in zip(ori_files_sorted, gem_files_sorted):
        ori_image_path = os.path.join(original_path, ori_file)
        gem_image_path = os.path.join(gemini_path, gem_file)

        ori_image = cv2.imread(ori_image_path)
        gem_image = cv2.imread(gem_image_path)

        # Skip if either image failed to load
        if ori_image is None or gem_image is None:
            print(f"Warning: Could not read {ori_file} or {gem_file}. Skipping this pair.")
            continue

        # Resize gemini image to match original image's dimensions
        gem_image_resized = cv2.resize(gem_image, (ori_image.shape[1], ori_image.shape[0]))

        # --- SAVE THE RESIZED IMAGE HERE ---
        resized_save_path = os.path.join(gemini_resized_path, ori_file)
        cv2.imwrite(resized_save_path, gem_image_resized)
        # -------------------------------------

        # Append to all lists only on success
        ori_stack.append(ori_image)
        gemini_stack.append(gem_image_resized) # Append the resized image
        ori_filenames.append(ori_file) # Store the original filename

print(f"Loaded {len(ori_stack)} images in each stack.")

# --- 2. Part 3: Affine Registration Function ---

def align_images_affine(reference_stack, moving_stack):
    """
    Aligns images from moving_stack to reference_stack using Part 3's method.
    """
    
    aligned_stack = []
    affine_matrices = []

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Initialize BFMatcher (Brute-Force) for feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2)

    for ref_img, mov_img in zip(reference_stack, moving_stack):
        
        # Convert images to grayscale for feature detection
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        mov_gray = cv2.cvtColor(mov_img, cv2.COLOR_BGR2GRAY)

        # Step 3: Detect features and compute descriptors
        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
        kp_mov, des_mov = sift.detectAndCompute(mov_gray, None)

        # Step 4: Match features using k-Nearest Neighbors
        matches = bf.knnMatch(des_mov, des_ref, k=2)

        # Step 5: Filter matches using Lowe's Ratio Test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        min_match_count = 3 
        if len(good_matches) > min_match_count:
            # Extract locations of good matches
            src_pts = np.float32([kp_mov[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Step 6: Estimate Affine Matrix using RANSAC
            M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            # Step 7: Warp the moving image to align with the reference
            height, width = ref_img.shape[:2]
            aligned_img = cv2.warpAffine(mov_img, M, (width, height))

            aligned_stack.append(aligned_img)
            affine_matrices.append(M)
            
        else:
            print(f"Warning: Not enough matches found ({len(good_matches)}/{min_match_count}). Appending unaligned image.")
            aligned_stack.append(mov_img) 
            affine_matrices.append(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    return aligned_stack, affine_matrices

# --- 3. Run the Registration ---

if ori_stack and gemini_stack:
    print("\nStarting affine registration...")
    aligned_gemini_stack, alignment_matrices = align_images_affine(ori_stack, gemini_stack)
    print("Registration complete.")

    # --- 4. Example: Save or Display Results ---
    
    print(f"Saving aligned images to {aligned_output_path}")

    # Zip all three lists together to get the filename, image, and matrix
    for filename, aligned_img, M in zip(ori_filenames, aligned_gemini_stack, alignment_matrices):
        
        # 'filename' now holds the name from the original stack
        output_path = os.path.join(aligned_output_path, filename)
        
        cv2.imwrite(output_path, aligned_img)
    
    print("Done.")

else:
    print("No images were loaded, skipping registration.")