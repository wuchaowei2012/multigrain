IMAGENET_PATH=/home/Code/multigrain/data

python scripts/finetune_p.py --expdir experiments/joint_3B_0.5/finetune500 \
	--resume-from experiments/joint_3B_0.5 --input-size 500 --imagenet-path $IMAGENET_PATH
