# Hairstyle Modifaction using 3D gaussians

Zhixuan Ge, Liza Jivnani

<p align="center" >
  <img src="figures/3dgs_vs_buzzcut.gif" alt="Ground Truth" width="100%" />
</p>


<br/>

## Installation

```bash
# clone this repository
git clone https://github.com/gezhixuan/GaussianHairEdit.git
cd VcEdit

#############################
# 1. Environment: vcedit
#############################
conda create -n vcedit python=3.8 -y
conda activate vcedit

# Our version is based on CUDA 11.8, see https://pytorch.org/get-started/locally/ to install other PyTorch versions.
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# core dependencies for VcEdit
pip install -r requirements.txt

#############################
# 2. Environment: GaussianHairEdit
#############################
conda create -n GaussianHairEdit python=3.8 -y
conda activate GaussianHairEdit

# dependencies specific to GaussianHairEdit
pip install -r requirements_GaussianHairEdit.txt

#############################
# 3. Environment: Gaussian3d
#############################
conda create -n Gaussian3d python=3.8 -y
conda activate Gaussian3d

# dependencies for 3D Gaussian / vanilla GS part
pip install -r requirements_Gaussian3d.txt

# (Optional) For vanilla Gaussian Splatting implementation, you can still
# follow the official repo setup inside this Gaussian3d environment:
#   https://github.com/graphdeco-inria/gaussian-splatting.git
```

## Example Data

To use your own monocular video as input, follow these steps:

1. Create a new folder under `GaussianHairEdit/gs_data` named with your scene ID, e.g. `GaussianHairEdit/gs_data/$id`.
2. Place your monocular input video in this folder and rename it to `raw.mp4`, so the final path is:
   ```text
   GaussianHairEdit/gs_data/$id/raw.mp4
   ```
3. From the project root, run:
   ```bash
   bash script/preprocess.sh
   ```
   This script will:
   - extract frames from `raw.mp4`,
   - run HyperIQA-based filtering, and  
   - perform a vanilla Gaussian Splatting reconstruction.

After this, the preprocessed data and reconstructed Gaussian scene will be available under `gs_data/$id` for further editing.

## Start Editing

We provide five end-to-end pipelines corresponding to the setups described in our paper. Each pipeline is wrapped in a shell script under `script/`. All scripts assume you have:

- Installed the three environments (`vcedit`, `GaussianHairEdit`, `Gaussian3d`) as described in **Installation**.
- Prepared your scene folder under `gs_data/$id` and run `script/preprocess.sh` on your monocular video (see **Example Data**).

### Pipeline overview and scripts

- **Pipeline 1 – Modified VCEdit (hair-only editing, no face swap)**  
  **Script:** `script/pipeline1_vcedit_without_faceswap.sh`  
  Uses a VCEdit-style view-consistent 3DGS pipeline with explicit hair/foreground masks. Good for testing basic view-consistent hair editing; may still suffer from semantic leakage and over-smoothing in the face region.

- **Pipeline 2 – VCEdit + Face Swap / Compositing (identity preservation)**  
  **Script:** `script/pipeline2_vcedit_with_faceswap.sh`  
  Extends Pipeline 1 with a post-processing face-swap/compositing stage that restores the original face on top of the edited hair, improving identity preservation but potentially introducing boundary artifacts along the hairline.

- **Pipeline 3 – Stable-Hair (2D hair transfer baseline)**  
  **Script:** `script/pipeline3_stable_hair.sh`  
  Uses Stable-Hair as a 2D hair-transfer model to edit each frame given a reference hairstyle. This mainly changes texture/color while largely preserving the original hair geometry, and is included as a 2D baseline / ablation.

- **Pipeline 4 – LMM-Guided Vanilla 3DGS (Gemini + standard Gaussians)**  
  **Script:** `script/pipeline4_llmguided_vanilla_gaussian.sh`  
  Edits frames independently using a multimodal LLM (e.g., Gemini 2.5 Flash), registers the edited frames back to the original cameras, and reconstructs with vanilla 3D Gaussian Splatting. Demonstrates typical “hollow face” / transparency issues without depth-regularized reconstruction.

- **Pipeline 5 – LMM-Guided SparseGS + Deformable Warp (Final / Recommended)**  
  **Script:** `script/pipeline5_llmguided_sparseGS_warp.sh`  
  Our final geometry-aware pipeline. Combines HyperIQA-filtered frames, LMM-guided 2D edits, SparseGS with depth priors, and a learnable warping module to remove letterbox artifacts and produce solid, photorealistic geometry. This is the pipeline we recommend for practical use.

### How to run the pipelines

From the project root, after preprocessing your video:

```bash
# Pipeline 1: Modified VCEdit (no face swap)
conda activate vcedit
bash script/pipeline1_vcedit_without_faceswap.sh

# Pipeline 2: VCEdit + post-process face swap
conda activate vcedit
bash script/pipeline2_vcedit_with_faceswap.sh

# Pipeline 3: Stable-Hair 2D baseline
conda activate GaussianHairEdit
bash script/pipeline3_stable_hair.sh

# Pipeline 4: LMM-guided editing + vanilla 3DGS
conda activate Gaussian3d
bash script/pipeline4_llmguided_vanilla_gaussian.sh

# Pipeline 5: LMM-guided editing + SparseGS with deformable camera refinement (recommended)
conda activate Gaussian3d
bash script/pipeline5_llmguided_sparseGS_warp.sh
````

Each script can be customized (e.g., scene ID, paths, prompt, number of iterations) by editing the corresponding `script/pipeline*.sh` file. For deeper technical details, limitations, and comparisons between these pipelines, please refer to the accompanying paper.



## Citation

If you find our work helps, please cite our paper:
```
@inproceedings{wang2025view,
  title={View-consistent 3d editing with gaussian splatting},
  author={Wang, Yuxuan and Yi, Xuanyu and Wu, Zike and Zhao, Na and Chen, Long and Zhang, Hanwang},
  booktitle={European Conference on Computer Vision},
  pages={404--420},
  year={2025},
  organization={Springer}
}
```

<br/>

## Acknowledgement
Our code is based on these wonderful repos:

* [InfEdit](https://github.com/sled-group/InfEdit/tree/main)
* [Gaussian Editor](https://github.com/buaacyw/GaussianEditor)
* [Threestudio](https://github.com/threestudio-project/threestudio)
* [InstructNerf2Nerf](https://github.com/ayaanzhaque/instruct-nerf2nerf)

