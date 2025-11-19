ID="Liza_new"
# ID="AAAFinal_results/Liza_aligned_clip0.28"


export PROJECT_DIR="$(pwd)"
export DATA_PATH="$PROJECT_DIR/gs_data/$ID"
export CAMERA="PINHOLE"
export GPU=6
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"


cd "$DATA_PATH"
eval "$(conda shell.bash hook)"


conda deactivate && conda activate vcedit
cd $PROJECT_DIR/ext/Stable-Hair
cp ../../src/infer_all.py ./
CUDA_VISIBLE_DEVICES="$GPU" python infer_all.py \
    --ref_image $PROJECT_DIR/ref_images/buzz_cut.png \
    --base_dir $DATA_PATH