export IMAGENET_PATH=/home/Code/multigrain/data
export WHITEN_PATH=$IMAGENET_PATH/whiten

# train network
python scripts/train.py --expdir experiments/joint_3B_0.5 --repeated-augmentations 3 \
--pooling-exponent 3 --classif-weight 0.5 --imagenet-path $IMAGENET_PATH

# fine-tune p*
python scripts/finetune_p.py --expdir experiments/joint_3B_0.5/finetune500 \
--resume-from experiments/joint_3B_0.5 --input-size 500 --imagenet-path $IMAGENET_PATH

# whitening 
python scripts/whiten.py --expdir experiments/joint_3B_0.5/finetune500_whitened \
--resume-from experiments/joint_3B_0.5/finetune500 --input-size 500 --whiten-path $WHITEN_PATH --workers 2





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

# 下载 whitening 数据库

import os

index = 1
lines = open("whiten.txt",'r')
for f in list(lines):
    
    temp = u"https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/" + f

    print(index)
    index = index + 1
    
    str_input = f.split('/')[-1]
    str_input = str_input.strip()
    print(str_input)

    if os.path.exists(str_input):
        print(str_input, "has been downloaded")
        continue
    os.system("wget {}".format(temp))
    
# 将所有的文件分到不同的文件夹里


import os.path as osp

index = 1
lines = open("whiten.txt",'r')
for f in list(lines):
    print(index)
    index = index + 1

    str_input_0 = f.split('/')[0]
    str_input_1 = f.split('/')[1]
    file_input = f.split('/')[2]
    file_input = file_input.strip()

    if not osp.exists(osp.join(osp.join("/home/Code/multigrain/data/whiten", str_input_0), str_input_1)):
        os.makedirs(osp.join(osp.join("/home/Code/multigrain/data/whiten", str_input_0), str_input_1))

    os.system("cp {} {}".format(file_input, osp.join(osp.join("/home/Code/multigrain/data/whiten", str_input_0), str_input_1)))

    