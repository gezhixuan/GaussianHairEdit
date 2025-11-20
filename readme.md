# Hairstyle Modifaction using 3D gaussians

<!-- <a href='https://vcedit.github.io'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2403.11868'><img src='https://img.shields.io/badge/arXiv-2403.11868-b31b1b.svg'></a>   -->

Zhixuan Ge, Liza Jivnani

<p align="center" >
  <img src="figures/3dgs_vs_buzzcut.gif" alt="Ground Truth" width="100%" />
</p>


<br/>

```markdown
## Installation

```bash
# clone this repository
git clone https://github.com/gezhixuan/GaussianHairEdit.git
cd VcEdit

# build environment for VcEdit
conda create -n vcedit python=3.8 -y
conda activate vcedit

# Our version is based on CUDA 11.8, see https://pytorch.org/get-started/locally/ to install other PyTorch versions.
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# install requirements for VcEdit
pip install -r requirements.txt
```

In addition, you need a separate environment for the *vanilla Gaussian Splatting* code:

```bash
# clone the official Gaussian Splatting repository
git clone https://github.com/graphdeco-inria/gaussian-splatting.git

# build environment for vanilla Gaussian Splatting
# (same dependencies as in the official repo, just using this env name)
conda create -n vanilla_gaussian_splatting python=3.8 -y
conda activate vanilla_gaussian_splatting

# now follow the installation steps from the official gaussian-splatting repo
# to install all required packages inside this environment.
```
```

<br/>

```markdown
## Example Data

To use your own monocular video as input, follow these steps:

1. Create a new folder under `VcEdit/gs_data` named with your scene ID, e.g. `VcEdit/gs_data/$id`.
2. Place your monocular input video in this folder and rename it to `raw.mp4`, so the final path is:
   ```text
   VcEdit/gs_data/$id/raw.mp4
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
```

<br/>

## Start Editing
We provide several sample scripts for running the editing process. For example:

`bash script/man_clown.sh`

The editing process typically takes 10 to 25 minutes on a single A100 GPU, depending on the scene size and the number of iterations. The GPU memory usage ranges between 20GB and 40GB. For panorama samples the memory cost for processing all views (which is unnecessary w.r.t. the performance) is a bit higher than 40GB, you can remove some views in the dataset.

<!-- Note that our editing is based on the image editing achieved by [InfEdit](https://github.com/sled-group/InfEdit/tree/main). Due to the diversity of editing scenarios, not all the prompt generates satisfying results. For example, some prompts lead to editing failure or drastic multi-view inconsistency in InfEdit. Please first try image editing using your prompt in [InfEdit](https://github.com/sled-group/InfEdit/tree/main) for prompt availability and satisfying hyper-parameters. -->

Our editing framework builds upon the image editing capabilities provided by [InfEdit](https://github.com/sled-group/InfEdit/tree/main). Due to the diverse nature of editing scenarios, not all prompts yield satisfying results. Some prompts may result in editing failures or too drastic multi-view inconsistencies when using InfEdit.
Before using this framework, we recommend testing your prompts in [InfEdit](https://github.com/sled-group/InfEdit/tree/main) to ensure:

* Prompt compatibility with the editing process.
* Optimal hyper-parameter selection for achieving desired results.
<br/>

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

