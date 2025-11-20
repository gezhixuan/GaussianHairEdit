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
CLIP_SCENE_DIR="${REGISTER_SCENE_DIR}_$CLIP_SUFFIX"


cd "$DATA_PATH"
eval "$(conda shell.bash hook)"

cd "$PROJECT_DIR"
cp -r "$SCENE_DIR" "$PROMPT_SCENE_DIR"
conda deactivate && conda activate zx_3d
CUDA_VISIBLE_DEVICES="$GPU" python src/gemini_batch_edit_list.py \
    --image-dir "$PROMPT_SCENE_DIR/images" \
    --prompt "$PROMPT" \
    --api-key ""

cp -r "$PROMPT_SCENE_DIR" "$REGISTER_SCENE_DIR"
rm -f "$REGISTER_SCENE_DIR/images"/*
cd "$PROJECT_DIR"
conda deactivate && conda activate zx_3d
CUDA_VISIBLE_DEVICES="$GPU" python src/affine_register.py \
    --original-dir "$DATA_PATH/images" \
    --gemini-dir "$PROMPT_SCENE_DIR/images" \
    --resized-dir "$RESIZED_DIR" \
    --aligned-dir "$REGISTER_SCENE_DIR/images"

CLIP_SCENE_DIR="${REGISTER_SCENE_DIR}_$CLIP_SUFFIX"
cd "$PROJECT_DIR"
conda deactivate && conda activate zx_3d
CUDA_VISIBLE_DEVICES="$GPU" python src/filter_scene_with_double_clip.py clip_filter \
  --src_scene "$REGISTER_SCENE_DIR" \
  --dst_scene "$CLIP_SCENE_DIR" \
  --positive_prompt "$positive_prompt" \
  --negative_prompt "$negative_prompt" \
  --threshold $CLIP_THRESHOLD

cd "$PROJECT_DIR"
conda deactivate && conda activate vanilla_gaussian_splatting
CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/train.py -s $DATA_PATH \
      --model_path $DATA_PATH/3d_gaussian_splatting --port 1007$GPU
