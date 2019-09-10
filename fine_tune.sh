export IMAGENET_PATH=/home/Code/multigrain/data

# train network
python scripts/train.py --expdir experiments/joint_3B_0.5 --repeated-augmentations 3 \
--pooling-exponent 3 --classif-weight 0.5 --imagenet-path $IMAGENET_PATH

# fine-tune p*
python scripts/finetune_p.py --expdir experiments/joint_3B_0.5/finetune500 \
--resume-from experiments/joint_3B_0.5 --input-size 500 --imagenet-path $IMAGENET_PATH

# whitening 
python scripts/whiten.py --expdir experiments/joint_3B_0.5/finetune500_whitened \
--resume-from experiments/joint_3B_0.5/finetune500 --input-size 500 --whiten-path $WHITEN_PATH


# whitening改成 计算 本数据集
export PYTHONPATH=/usr/local/anaconda3/bin/

$PYTHONPATH/python3 scripts/whiten.py --expdir experiments/joint_3B_0.5/finetune500_whitened \
--resume-from experiments/joint_3B_0.5/finetune500_whitened --input-size 500 --whiten-path $IMAGENET_PATH/whiten



# Classification results
# Evaluate a network on ImageNet-val is straightforward using options from evaluate.py. For instance the following command:

python scripts/evaluate.py --expdir experiments/joint_3B_0.5/eval_p4_500 \
--imagenet-path $IMAGENET_PATH --input-size 500 --dataset imagenet-val \
--pooling-exponent 4 --resume-from joint_3B_0.5.pth




import os
index = 1
for f in os.listdir():
    temp = f.split('_')[-1].split('.')[0]
    print(temp.rjust(8, '0'))
    #os.rename(f, "ILSVRC2012_train_"+temp.rjust(8, '0')+'.JPEG')
    os.rename(f, "ILSVRC2012_train_"+str(index).rjust(8, '0')+'.JPEG')

    index = index + 1



