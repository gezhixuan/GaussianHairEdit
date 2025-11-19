ID="Liza_new"
# ID="AAAFinal_results/Liza_aligned_clip0.28"


export PROJECT_DIR="$(pwd)"
export DATA_PATH="$PROJECT_DIR/gs_data/$ID"
export CAMERA="PINHOLE"
export GPU=6
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"


cd "$DATA_PATH"
eval "$(conda shell.bash hook)"


# ffmpeg -noautorotate -i ./merged.mp4 \
#   -vf "transpose=1,setsar=1" \
#   -metadata:s:v:0 rotate=0 -map_metadata -1 \
#   -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
#   -c:a copy ./raw_no90.mp4

# ffmpeg -i ./raw_no90.mp4 -c copy -metadata:s:v:0 rotate=0 ./raw.mp4

conda deactivate && conda activate vcedit
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python preprocess_raw_images.py \
    --data_path $DATA_PATH

conda deactivate && conda activate vcedit
cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python convert.py -s $DATA_PATH \
    --camera $CAMERA --max_size 1024 --no_gpu


cd "$PROJECT_DIR"
conda deactivate && conda activate vanilla_gaussian_splatting
CUDA_VISIBLE_DEVICES="$GPU" python gaussian-splatting/train.py -s $DATA_PATH \
      --model_path $DATA_PATH/3d_gaussian_splatting --port 1007$GPU

