ID="AAAFinal_results/Liza_new"

export PROJECT_DIR="$(pwd)"
export DATA_PATH="$PROJECT_DIR/gs_data/$ID"
export CAMERA="PINHOLE"
export GPU=1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

SCENE_DIR="$DATA_PATH"
PROMPT="buzz cut"   # or whatever prompt you want
# PROMPT="long natural curls"   # or whatever prompt you want
PROMPT="long straigt hair"
ORI_PROMPT="braids"
PROMPT_SCENE_DIR="${DATA_PATH}_$PROMPT"
REGISTER_SCENE_DIR="${PROMPT_SCENE_DIR}_faceregistered"
# We'll save resized Gemini frames beside images/, in resized_images/
RESIZED_DIR="$REGISTER_SCENE_DIR/resized_images"
CLIP_PROMPT="a person with $PROMPT hairstyle"
positive_prompt="a person with $PROMPT"
negative_prompt="a person with $PROMPT with $ORI_PROMPT"

# CLIP_THRESHOLD=0.28
# CLIP_THRESHOLD=0.05
CLIP_THRESHOLD=0.01
CLIP_SUFFIX="clip$CLIP_THRESHOLD"
CLIP_SCENE_DIR="$PROMPT_SCENE_DIR"


cd "$DATA_PATH"
eval "$(conda shell.bash hook)"

cd "$PROJECT_DIR"
cp -r "$SCENE_DIR" "$PROMPT_SCENE_DIR"
conda deactivate && conda activate zx_3d
CUDA_VISIBLE_DEVICES="$GPU" python src/gemini_batch_edit_list.py \
    --image-dir "$PROMPT_SCENE_DIR/images" \
    --prompt "$PROMPT" \
    --api-key ""

cd "$PROJECT_DIR/SparseGS"
conda deactivate && conda activate instantsplat
CUDA_VISIBLE_DEVICES="$GPU" python prepare_gt_depth.py \
  "$CLIP_SCENE_DIR/images" \
  "$CLIP_SCENE_DIR/depths"

cd "$PROJECT_DIR/SparseGS_warp"
conda deactivate && conda activate instantsplat
CUDA_VISIBLE_DEVICES="$GPU" python train.py --source_path "$CLIP_SCENE_DIR" \
  --model_path "$CLIP_SCENE_DIR/SparseGS" \
  --beta 5.0 --lambda_pearson 0.05 \
  --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 \
  --lambda_diffusion 0.001 --SDS_freq 0.1 --step_ratio 0.99 \
  --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 \
  --iterations 30000 --checkpoint_iterations 30000 -r 2