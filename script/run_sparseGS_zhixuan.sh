ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Liza_new"


# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_buzz cut_registered_clip-0.3"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_buzz cut_registered_clip0.05"

# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_buzz cut"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Liza_new_buzz cut"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_long natural curls"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Liza_new_long natural curls"

# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_buzz cut_registered_clip0.05"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_long natural curls_faceregistered_clip0.01"


# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Liza_new_buzz cut_registered_clip0.01"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Liza_new_long natural curls_registered_clip0.01"

# IDA="Liza_new"
# IDB="buzz cut"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/$IDA_$IDB"



ORIID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Liza_new_buzz cut"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_buzz cut"
ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_long natural curls"
# ID="/scratch/hl106/zx_workspace/cto/VcEdit/gs_data/AAAFinal_results/Zhixuan_new_long natural curls"



export PROJECT_DIR="/scratch/hl106/zx_workspace/cto/VcEdit"
export DATA_PATH="$PROJECT_DIR/gs_data/$ID"
export CAMERA="PINHOLE"
export GPU=3
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

SCENE_DIR="$DATA_PATH"
PROMPT="buzz cut"   # or whatever prompt you want
# PROMPT="long natural curls"   # or whatever prompt you want
ORI_PROMPT="long hair in a ponytail and long front bangs"
PROMPT_SCENE_DIR="${DATA_PATH}_$PROMPT"
# REGISTER_SCENE_DIR="${PROMPT_SCENE_DIR}_faceregistered"
REGISTER_SCENE_DIR="${PROMPT_SCENE_DIR}_registered"
# We'll save resized Gemini frames beside images/, in resized_images/
RESIZED_DIR="$REGISTER_SCENE_DIR/resized_images"
CLIP_PROMPT="a person with $PROMPT hairstyle"
positive_prompt="a person with $PROMPT"
negative_prompt="a person with $PROMPT with $ORI_PROMPT"

# CLIP_THRESHOLD=0.28
# CLIP_THRESHOLD=0.05
CLIP_THRESHOLD=0.01
CLIP_SUFFIX="clip$CLIP_THRESHOLD"
CLIP_SCENE_DIR="${ID}"

eval "$(conda shell.bash hook)"

# cd "$PROJECT_DIR/SparseGS"
# conda deactivate && conda activate instantsplat
# CUDA_VISIBLE_DEVICES="$GPU" python prepare_gt_depth.py \
#   "$CLIP_SCENE_DIR/images" \
#   "$CLIP_SCENE_DIR/depths"


# cd "$PROJECT_DIR/ext/RobustVideoMatting"
# conda deactivate && conda activate instantsplat
# CUDA_VISIBLE_DEVICES="$GPU" python make_alpha.py \
#   "$CLIP_SCENE_DIR" 

# cd "$PROJECT_DIR/SparseGS"
# conda deactivate && conda activate instantsplat
# CUDA_VISIBLE_DEVICES="$GPU" python train.py --source_path "$CLIP_SCENE_DIR" \
#   --start_checkpoint "$CLIP_SCENE_DIR/3d_gaussian_splatting/point_cloud/iteration_30000/point_cloud.ply" \
#   --model_path "$CLIP_SCENE_DIR/SparseGS" \
#   --beta 5.0 --lambda_pearson 0.05 \
#   --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 \
#   --lambda_diffusion 0.001 --SDS_freq 0.1 --step_ratio 0.99 \
#   --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 \
#   --iterations 40000 --checkpoint_iterations 30000 -r 2

# cd "$PROJECT_DIR/SparseGS"
# conda deactivate && conda activate instantsplat
# CUDA_VISIBLE_DEVICES="$GPU" python train.py --source_path "$CLIP_SCENE_DIR" \
#   --model_path "$CLIP_SCENE_DIR/SparseGS" \
#   --beta 5.0 --lambda_pearson 0.05 \
#   --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 \
#   --lambda_diffusion 0.001 --SDS_freq 0.1 --step_ratio 0.99 \
#   --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 \
#   --iterations 30000 --checkpoint_iterations 30000 -r 2

# cd "$PROJECT_DIR/ext/RobustVideoMatting"
# conda deactivate && conda activate instantsplat
# CUDA_VISIBLE_DEVICES="$GPU" python make_alpha.py \
#   "$ORIID" 
# cp -r "$ORIID/images"  "$ID/ref_images"
# cp -r "$ORIID/alpha"  "$ID/ref_alpha"

cd "$PROJECT_DIR/SparseGS_warp"
conda deactivate && conda activate instantsplat
# rm -rf "$CLIP_SCENE_DIR/SparseGS_warp"
rm -rf "$CLIP_SCENE_DIR/SparseGS_warp_homo"

CUDA_VISIBLE_DEVICES="$GPU" python train.py --source_path "$CLIP_SCENE_DIR" \
  --model_path "$CLIP_SCENE_DIR/SparseGS_warp_homo/" \
  --beta 5.0 --lambda_pearson 0.05 \
  --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 \
  --lambda_diffusion 0.001 --SDS_freq 0.1 --step_ratio 0.99 \
  --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 \
  --iterations 30000 --checkpoint_iterations 30000 -r 2