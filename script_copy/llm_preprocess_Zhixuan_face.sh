# ID="Zhixuan"
# ID="Zhixuan_3"
# ID="Liza"
# ID="Liza_3"
# ID="Liza_new"
# ID="Someone"
# ID="gemini-editted"
# ID="Liza_gemeni"
# ID="Liza_gemeni_long"
# ID="Liza_gemeni_buzzcut_clip"
# ID="Liza_gemeni_buzzcut_clip0.28"
# ID="Liza_aligned"
# ID="Liza_aligned_clip0.3"
# ID="Liza_gemeni_resized"
# ID="Liza_aligned_clip0.24"
# ID="Liza_aligned_clip0.28"
# ID="AAAFinal_results/Liza_aligned_gemini"
# ID="AAAFinal_results/Liza_aligned_clip0.28"
ID="AAAFinal_results/Zhixuan_new"



export PROJECT_DIR="/scratch/hl106/zx_workspace/cto/VcEdit"
export DATA_PATH="$PROJECT_DIR/gs_data/$ID"
export CAMERA="PINHOLE"
export GPU=2
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

SCENE_DIR="$DATA_PATH"
# PROMPT="buzz cut"   # or whatever prompt you want
PROMPT="long natural curls"   # or whatever prompt you want
ORI_PROMPT="long hair in a ponytail and long front bangs"
PROMPT_SCENE_DIR="${DATA_PATH}_$PROMPT"
REGISTER_SCENE_DIR="${PROMPT_SCENE_DIR}_registered"
# We'll save resized Gemini frames beside images/, in resized_images/
RESIZED_DIR="$REGISTER_SCENE_DIR/resized_images"
CLIP_PROMPT="a person with $PROMPT hairstyle"
positive_prompt="a person with $PROMPT"
negative_prompt="a person with $PROMPT with $ORI_PROMPT"

# CLIP_THRESHOLD=0.28
# CLIP_THRESHOLD=0.05
# CLIP_THRESHOLD=0.05
CLIP_THRESHOLD=0.3
CLIP_SUFFIX="clip$CLIP_THRESHOLD"
CLIP_SCENE_DIR="${REGISTER_SCENE_DIR}_$CLIP_SUFFIX"


cd "$DATA_PATH"
eval "$(conda shell.bash hook)"

# cd "$PROJECT_DIR"
# conda deactivate && conda activate zx_3d
# CUDA_VISIBLE_DEVICES="$GPU" python src/filter_scene.py  subsample \
#     --scene gs_data/AAAFinal_results/Zhixuan_new \
#     --max_images 80

# cd "$PROJECT_DIR"
# cp -r "$SCENE_DIR" "$PROMPT_SCENE_DIR"
# conda deactivate && conda activate zx_3d
# CUDA_VISIBLE_DEVICES="$GPU" python src/gemini_batch_edit_list.py \
#     --image-dir "$PROMPT_SCENE_DIR/images" \
#     --prompt "$PROMPT" \
#     --api-key "AIzaSyA446tM4fga-AbJ3pNKg84-fkcqiQCsCEM"

# ---------- 4. Copy edited scene for registration ----------
# cp -r "$PROMPT_SCENE_DIR" "$REGISTER_SCENE_DIR"
# rm -f "$REGISTER_SCENE_DIR/images"/*
# # cd "$PROJECT_DIR/ext/face-parsing.PyTorch/"
# cd "$PROJECT_DIR/ext/RobustVideoMatting/"
# conda deactivate && conda activate zx_3d
# # CUDA_VISIBLE_DEVICES="$GPU" python align_face.py \
# #     --original-dir "$DATA_PATH/images" \
# #     --gemini-dir "$PROMPT_SCENE_DIR/images" \
# #     --resized-dir "$RESIZED_DIR" \
# #     --aligned-dir "$REGISTER_SCENE_DIR/images"
# CUDA_VISIBLE_DEVICES="$GPU" python src/affine_register.py \
#     --original-dir "$DATA_PATH/images" \
#     --gemini-dir "$PROMPT_SCENE_DIR/images" \
#     --resized-dir "$RESIZED_DIR" \
#     --aligned-dir "$REGISTER_SCENE_DIR/images"

cd "$PROJECT_DIR"
conda deactivate && conda activate zx_3d
CUDA_VISIBLE_DEVICES="$GPU" python src/filter_scene_with_double_clip.py clip_filter \
  --src_scene "$REGISTER_SCENE_DIR" \
  --dst_scene "$CLIP_SCENE_DIR" \
  --positive_prompt "$positive_prompt" \
  --negative_prompt "$negative_prompt" \
  --threshold $CLIP_THRESHOLD

cd "$PROJECT_DIR/SparseGS"
conda deactivate && conda activate instantsplat
CUDA_VISIBLE_DEVICES="$GPU" python prepare_gt_depth.py \
  "$CLIP_SCENE_DIR/images" \
  "$CLIP_SCENE_DIR/depths"

cd "$PROJECT_DIR/SparseGS"
conda deactivate && conda activate instantsplat
CUDA_VISIBLE_DEVICES="$GPU" python train.py --source_path "$CLIP_SCENE_DIR" \
  --model_path "$CLIP_SCENE_DIR/SparseGS" \
  --beta 5.0 --lambda_pearson 0.05 \
  --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 \
  --lambda_diffusion 0.001 --SDS_freq 0.1 --step_ratio 0.99 \
  --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 \
  --iterations 30000 --checkpoint_iterations 30000 -r 2
