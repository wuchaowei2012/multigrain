export CUDA_VISIBLE_DEVICES=$1

export starts=$2
echo 2: starts, $starts

export embeddingFilePath=$3
echo 1: embeddingFilePath , $embeddingFilePath

python3 scripts/extract_scriptor.py --starts $starts --embeddingFilePath=$embeddingFilePath --expdir experiments/joint_3B_0.5/eval_p4_500 --input-size 200  --pooling-exponent 4 --resume-from joint_3B_0.5.pth