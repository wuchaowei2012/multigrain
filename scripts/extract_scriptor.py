import torch
from torch.utils.data import DataLoader
import faiss

import json

import set_path
from multigrain.utils import logging
from multigrain.augmentations import get_transforms
from multigrain.lib import get_multigrain, list_collate
from multigrain.datasets import IN1K, IdDataset

from multigrain.datasets.meizi_dataset import meizi_dataset

from multigrain import utils
from multigrain.backbones import backbone_list

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict as OD
from collections import defaultdict
import yaml
import os.path as osp

import ipdb
import numpy
import numpy as np

import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

import ipdb

tic, toc = utils.Tictoc()

class MmapVectorUtils:
    @staticmethod
    def Read(path, dim):
        x = MmapVectorUtils.Open(path, False)
        print(len(x))
        print(len(x)/(dim+1))
        rlt = x.reshape(-1, dim + 1)
        return rlt[:,0:1], rlt[:,1:]

    @staticmethod
    def Open(path, write=False, shape=None):
        xb = np.memmap(path, dtype='float32', mode='w+' if write else 'r', shape=shape)
        return xb 

    @staticmethod
    def Write(xb, offset, ids, vectors):
        last = offset + len(ids)
        xb[offset:last,0:1] = ids
        xb[offset:last,1:] = vectors


def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    try:
        return list_collate(batch)
    except:
        return None

def run(args):
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(argstr)

    argfile = osp.join(osp.join(args.expdir), 'evaluate_args.yaml')

    args.cuda = not args.no_cuda

    if not args.dry:
        utils.ifmakedirs(args.expdir)
        logging.print_file(argstr, argfile)
    
    collate_fn = dict(collate_fn=my_collate) if args.input_crop == 'rect' else {}

    transforms = get_transforms(input_size=args.input_size, \
        crop=(args.input_crop == 'square'), need=('val',),
        backbone=args.backbone)

    if args.dataset.startswith('imagenet'):
        dataset = IdDataset(IN1K(args.imagenet_path, args.dataset[len('imagenet-'):],
                                 transform=transforms['val']))
        mode = "classification"
    else:
        dataset = IdDataset(meizi_dataset(root=args.meizi_path, transform=transforms['val'], starts=args.starts))
        mode = "retrieval"

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle,
                        pin_memory=True, **collate_fn)
    model = get_multigrain(args.backbone, include_sampling=False, pretrained_backbone=args.pretrained_backbone)

    p = model.pool.p
    checkpoints = utils.CheckpointHandler(args.expdir)

    if checkpoints.exists(args.resume_epoch, args.resume_from):
        epoch = checkpoints.resume(model, resume_epoch=args.resume_epoch, resume_from=args.resume_from,
                                   return_extra=False)
    else:
        raise ValueError('Checkpoint ' + args.resume_from + ' not found')

    if args.pooling_exponent is not None:  # overwrite stored pooling exponent
        p.data.fill_(args.pooling_exponent)

    print("Multigrain model with {} backbone and p={} pooling:".format(args.backbone, p.item()))
    # print(model)

    if True:
        model = utils.cuda(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # freeze batch normalization

    metrics_history = OD()
    metrics = defaultdict(utils.HistoryMeter)

    index = None
    tic()

    start_id = 0
    
    imageCount = len(dataset)
    Dim = 2048

    xb = MmapVectorUtils.Open(args.embeddingFilePath, True, shape=(imageCount, Dim + 2))

    for i, batch in enumerate(loader):
        # ipdb.set_trace()
        #try:
        batch_vid = batch['vid']
        batch_fid = batch['frame_id']

        with torch.no_grad():
            #if args.cuda:
            if True:
                batch = utils.cuda(batch)

            metrics["data_time"].update(1000 * toc());
            tic()
            output_dict = model(batch['input'])

        if mode == "classification":
            # target = batch['classifier_target']
            # top1, top5 = utils.accuracy(output_dict['classifier_output'], target, topk=(1, 5))
            # metrics["val_top1"].update(top1, n=len(batch['input']))
            # metrics["val_top5"].update(top5, n=len(batch['input']))
            raise ValueError('only focus on retrival')

        elif mode == "retrieval":
            descriptors = output_dict['normalized_embedding']

            n, dim = descriptors.shape
            end_id = n + start_id
            end_id = min(imageCount, end_id)
            n = end_id - start_id
            
            xb[start_id:end_id, 0:1] = np.array(batch_vid).reshape(n,1)
            xb[start_id:end_id, 1:2] = np.array(batch_fid).reshape(n,1)

            xb[start_id:end_id, 2:] = descriptors[0:n].cpu().numpy()

            start_id += n
        metrics["batch_time"].update(1000 * toc());
        tic()
        print(logging.str_metrics(metrics, iter=i, num_iters=len(loader), epoch=epoch, num_epochs=epoch))

    print(logging.str_metrics(metrics, epoch=epoch, num_epochs=1))
    for k in metrics: metrics[k] = metrics[k].avg

    toc()

    metrics_history[epoch] = metrics
    # checkpoints.save_metrics(metrics_history)

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['imagenet-val', 'imagenet-trainaug', 'holidays', 'copydays', 'ukbench', 'meizi'],  
    default='meizi', help='which evaluation to make')
    parser.add_argument('--shuffle', action='store_true', help='shuffle dataset before evaluation')
    parser.add_argument('--expdir', default='experiments/resnet50/finetune500_whitened/holidays500',
                        help='evaluation destination directory')
    parser.add_argument('--resume-epoch', default=-1, type=int, help='resume epoch (-1: last, 0: from scratch)')
    parser.add_argument('--resume-from', default=None, help='resume checkpoint file/folder')
    parser.add_argument('--input-size', default=500, type=int, help='images input size')
    parser.add_argument('--input-crop', default='rect', choices=['square', 'rect'], help='crop the input or not')
    parser.add_argument('--batch-size', default=1000, type=int, help='batch size')
    parser.add_argument('--backbone', default='resnet50', choices=backbone_list, help='backbone architecture')
    parser.add_argument('--pretrained-backbone', action='store_const', const='imagenet', help='use pretrained backbone')
    parser.add_argument('--pooling-exponent', default=None, type=float,
                        help='pooling exponent in GeM pooling (default: use value from checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', help='do not use CUDA', default=False)
    parser.add_argument('--imagenet-path', default='/home/data/ilsvrc2012', help='ImageNet data root')
    ##################################################################################################
    parser.add_argument('--meizi-path', default='/home/meizi/long_video_pic', help='meizi data root')
    parser.add_argument('--holidays-path', default='/home/data/Holidays', help='INRIA Holidays data root')
    parser.add_argument('--UKBench-path', default='/home/data/UKBench', help='UKBench data root')
    parser.add_argument('--copydays-path', default='/home/data/Copydays', help='INRIA Copydays data root')
    parser.add_argument('--distractors-list', default='data/distractors.txt', help='list of distractor images')
    parser.add_argument('--distractors-path', default='data/distractors', help='path to distractor images')
    parser.add_argument('--num_distractors', default=0, type=int, help='number of distractor images.')
    parser.add_argument('--preload-dir-imagenet', default=None,
                        help='preload imagenet in this directory (useful for slow networks')
    parser.add_argument('--workers', default=5, type=int, help='number of data-fetching workers')
    parser.add_argument('--dry', action='store_true', help='do not store anything')
    parser.add_argument('--embeddingFilePath', default='/home/meizi/short1_125.txt', help='embedding file')
    parser.add_argument('--starts', default='1', help='dataset filtering')
    
    args = parser.parse_args()
    run(args)