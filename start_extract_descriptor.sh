export CUDA_VISIBLE_DEVICES=0
export DATA_SET='meizi'


python3 scripts/extract_scriptor.py --expdir experiments/joint_3B_0.5/eval_p4_500 --input-size 200 --dataset meizi --pooling-exponent 4 --resume-from joint_3B_0.5.pth
