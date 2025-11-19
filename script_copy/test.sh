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
# ID="AAAFinal_results/Zhixuan_new_buzz cut_registered_clip0.05"
ID="AAAFinal_results/Zhixuan_new_long natural curls_faceregistered_clip0.01"
# ID="AAAFinal_results/Liza_new_long natural curls_registered_clip0.01"
ID="AAAFinal_results/Liza_new_buzz cut_registered_clip0.01"
# ID="Liza_aligned_clip0.0"
ID="Liza_aligned_clip0.3"
ID="AAAFinal_results/Liza_new_buzz cut"

# ID="Liza_gemeni_buzzcut_clip0.28"
ID="AAAFinal_results/Liza_new_buzz cut_registered"


export PROJECT_DIR="/scratch/hl106/zx_workspace/cto/VcEdit"
export DATA_PATH="$PROJECT_DIR/gs_data/$ID"
export CAMERA="PINHOLE"
export GPU=6
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

SCENE_DIR="$DATA_PATH"
# PROMPT="buzz cut"   # or whatever prompt you want
PROMPT="long natural curls"   # or whatever prompt you want
ORI_PROMPT="long hair in a ponytail and long front bangs"
PROMPT_SCENE_DIR="${DATA_PATH}_$PROMPT"
REGISTER_SCENE_DIR="${PROMPT_SCENE_DIR}_faceregistered"
# We'll save resized Gemini frames beside images/, in resized_images/
RESIZED_DIR="$REGISTER_SCENE_DIR/resized_images"
CLIP_PROMPT="a person with $PROMPT hairstyle"
positive_prompt="a person with $PROMPT"
negative_prompt="a person with $PROMPT with $ORI_PROMPT"

# CLIP_THRESHOLD=0.28
# CLIP_THRESHOLD=0.05
CLIP_THRESHOLD=0.05
CLIP_SUFFIX="clip$CLIP_THRESHOLD"
CLIP_SCENE_DIR="${REGISTER_SCENE_DIR}_$CLIP_SUFFIX"


cd "$DATA_PATH"
eval "$(conda shell.bash hook)"


# cd "$PROJECT_DIR/gaussian-splatting"
# cd "$PROJECT_DIR"
# cp -r "$DATA_PATH/3d_gaussian_splatting/cfg_args" "$DATA_PATH/SparseGS_no_black"
# conda deactivate && conda activate vanilla_gaussian_splatting
# cp -r "$DATA_PATH/3d_gaussian_splatting/cfg_args" "$DATA_PATH/SparseGS"
# CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py -s "$DATA_PATH"  \
#       --model_path "$DATA_PATH/SparseGS"


# CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py  \
#       --model_path "$DATA_PATH/SparseGS_no_black"
# rm "$DATA_PATH/SparseGS/cfg_args"

cd "$PROJECT_DIR"
conda deactivate && conda activate vanilla_gaussian_splatting
CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py -s  "$DATA_PATH" \
      --model_path "$DATA_PATH/3d_gaussian_splatting"

conda deactivate && conda activate vanilla_gaussian_splatting
CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py -s  "$DATA_PATH" \
      --iteration  7000 \
      --model_path "$DATA_PATH/3d_gaussian_splatting"

# cd "$PROJECT_DIR"
# cp -r "$DATA_PATH/3d_gaussian_splatting/cfg_args" "$DATA_PATH/SparseGS_warp"
# conda deactivate && conda activate vanilla_gaussian_splatting
# CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py -s "$DATA_PATH"  \
#       --model_path "$DATA_PATH/SparseGS_warp"
# CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py -s "$DATA_PATH"  \
#       --model_path "$DATA_PATH/SparseGS_warp" --iteration 7000

# ID="AAAFinal_results/Liza_new_buzz cut_registered"
# export DATA_PATH="$PROJECT_DIR/gs_data/$ID"
# cd "$PROJECT_DIR"
# cp -r "$DATA_PATH/3d_gaussian_splatting/cfg_args" "$DATA_PATH/SparseGS"
# conda deactivate && conda activate vanilla_gaussian_splatting
# CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py -s "$DATA_PATH"  \
#       --model_path "$DATA_PATH/SparseGS"
# CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/render.py -s "$DATA_PATH"  \
#       --model_path "$DATA_PATH/SparseGS" --iteration 7000
