export CUDA_VISIBLE_DEVICES=$1

export dataset_dir=$2
echo 4: dataset_dir, $dataset_dir

export starts=$3
echo 2: starts, $starts

export embeddingFilePath=$4
echo 3: embeddingFilePath , $embeddingFilePath

python3 scripts/extract_scriptor.py --starts $starts --embeddingFilePath=$embeddingFilePath --meizi-path=$dataset_dir  --expdir experiments/joint_3B_0.5/eval_p4_500 --input-size 200  --pooling-exponent 4 --resume-from joint_3B_0.5.pth
