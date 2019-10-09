import glob
import os
import pandas as pd
import sys

import itertools


def ListFiles(path):
    rlt = glob.glob(path + "/*")
    rlt.sort()
    # return list(map(lambda x:x.split('/')[-1], rlt))
    return rlt

def Listvid(path):
    rlt = glob.glob(path + "/*")
    rlt.sort()
    return list(map(lambda x:x.split('/')[-1], rlt))

def Total_jpg(list_folder):
    jpg_list = list()

    # get jpg files in a specific folder
    def extract_jpg(folder):
        jpg =glob.glob(folder + "/*")
        jpg.sort()
        return list(filter(lambda x:x.endswith('jpg'), jpg))
    
    list2d = list(map(lambda folder: extract_jpg(folder), list_folder))
    total_jpgs = list(itertools.chain(*list2d))

    return total_jpgs

path_root = "/devdata/videos/match_rst/"

folder_list = ListFiles(path_root)

jpg_lists = list()
for folder in folder_list:
    if not os.path.isdir(folder):
        continue
    jpg_list = [item for item in ListFiles(folder) if item.endswith('jpg')]
    jpg_lists.extend(jpg_list)

jpg_lists.sort()



with open('total_matched_jpgs.csv','w+') as f:
    f.writelines('jpg')
    f.writelines('\n')

    for item in total_jpgs:
        f.writelines(item)
        f.writelines('\n')



# Total long Video counts          120534
# Total distinct long Video counts         120534
# the long Video which has not yet been processed          1781
# the jpgs from video video        70208251