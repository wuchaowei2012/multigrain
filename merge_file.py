import hashlib
# import cv2
import os
from pathlib import Path
import pickle
import shutil
import glob
import numpy as np
from tqdm import tqdm


class MmapVectorUtils:
    @staticmethod
    def Print(path, dim):
        xb = MmapVectorUtils.Open(path, False)
        rlt = xb.reshape(-1, dim)
        print(rlt.shape)
        print(rlt)

    @staticmethod
    def Read(path, dim, copyToMemory = False):
        x = MmapVectorUtils.Open(path, False)
        # print(len(x))
        # print(len(x)/(dim))
        x = x.reshape(-1, dim)
        rlt = x.copy() if copyToMemory else x
        return rlt[:, 0:1], rlt[:, 1:2],  rlt[:, 2:]

    @staticmethod
    def Open(path, write=False, shape=None):
        xb = np.memmap(path, dtype='float32',
                       mode='w+' if write else 'r', shape=shape)
        return xb

    @staticmethod
    def Write(xb, offset, vids, fids, vectors):
        last = offset + len(vids)
        xb[offset:last, 0:1] = vids
        xb[offset:last, 1:2] = fids
        xb[offset:last, 2:] = vectors


def Merger(dst_file, src_file, deleteAfterMeger=False):
    try:
        print(src_file)
        with open(src_file, "rb") as fr, open(dst_file, 'ab') as fw:  # fr读文件
            while True:
                data = fr.read(4096)
                if not data:
                    break
                fw.write(data)
        if deleteAfterMeger:
            os.remove(src_file)
    except OSError:
        print("open file error!")
        return False
    except:
        print("Merger failed")
    return True


def FilterOutEmptyVector(filePath, dim, backup=False):
    xi, fi,  xd = MmapVectorUtils.Read(filePath,dim)
    count = sum([1 for i in xi if i==0])
    print("total count:{} empty item count:{}".format(len(xi), count))
    size = int(len(xi)-count)

    tempFileName = filePath+".tmp"
    xd1 = MmapVectorUtils.Open(tempFileName, True, (size, dim))
    
    j = 0
    for i in tqdm(range(len(xi))):
        if 0 != xi[i]:
            xd1[j,0] = xi[i]
            xd1[j,1] = fi[i]
            xd1[j,2:] = xd[i]
            j+=1

    if backup:
        os.rename(filePath, filePath+".bak")
    else:
        os.remove(filePath)
    os.rename(tempFileName, filePath)




# 检查这三个文件对应的embedding,可能出现问题
# 'short_s2_h135.txt','short_s3_h125.txt','short_s3_h128.txt'

# list_short = ['short_s1_h125.txt','short_s1_h128.txt','short_s1_h129.txt','short_s1_h130.txt','short_s1_h131.txt','short_s1_h132.txt',\
# 'short_s1_h133.txt','short_s1_h134.txt','short_s1_h135.txt','short_s2_h125.txt','short_s2_h128.txt','short_s2_h129.txt','short_s2_h130.txt',\
# 'short_s2_h131.txt','short_s2_h132.txt','short_s2_h133.txt','short_s2_h134.txt','short_s2_h135.txt','short_s3_h125.txt','short_s3_h128.txt',\
# 'short_s3_h129.txt','short_s3_h130.txt','short_s3_h131.txt','short_s3_h132.txt','short_s3_h133.txt','short_s3_h134.txt','short_s3_h135.txt',\
# 'short_s4_h125.txt','short_s4_h128.txt','short_s4_h129.txt','short_s4_h130.txt','short_s4_h131.txt','short_s4_h132.txt','short_s4_h133.txt',\
# 'short_s4_h134.txt','short_s4_h135.txt','short_s5_h125.txt','short_s5_h128.txt','short_s5_h129.txt','short_s5_h130.txt','short_s5_h131.txt',\
# 'short_s5_h132.txt','short_s5_h133.txt','short_s5_h134.txt','short_s5_h135.txt','short_s6_h125.txt','short_s6_h128.txt','short_s6_h129.txt',\
# 'short_s6_h130.txt','short_s6_h131.txt','short_s6_h132.txt','short_s6_h133.txt','short_s6_h134.txt','short_s6_h135.txt','short_s7_h125.txt',\
# 'short_s7_h128.txt','short_s7_h129.txt','short_s7_h130.txt','short_s7_h134.txt','short_s8_h125.txt','short_s9_h133.txt']


list_long = [
    'long_s1_h125.txt','long_s1_h128.txt','long_s1_h129.txt','long_s1_h130.txt','long_s1_h131.txt','long_s1_h132.txt','long_s1_h133.txt','long_s1_h134.txt',\
    'long_s1_h135.txt','long_s2_h125.txt','long_s2_h128.txt','long_s2_h129.txt','long_s2_h130.txt','long_s2_h131.txt','long_s2_h132.txt','long_s2_h133.txt',\
    'long_s2_h134.txt','long_s2_h135.txt','long_s3_h125.txt','long_s3_h128.txt','long_s3_h129.txt','long_s3_h130.txt','long_s3_h131.txt','long_s3_h132.txt',\
    'long_s3_h133.txt','long_s3_h134.txt','long_s3_h135.txt','long_s4_h125.txt','long_s4_h128.txt','long_s4_h129.txt','long_s4_h130.txt','long_s4_h131.txt',\
    'long_s4_h132.txt','long_s4_h133.txt','long_s4_h134.txt','long_s4_h135.txt','long_s5_h125.txt','long_s5_h128.txt','long_s5_h129.txt','long_s5_h130.txt',\
    'long_s5_h131.txt','long_s5_h132.txt','long_s5_h133.txt','long_s5_h134.txt','long_s5_h135.txt','long_s6_h125.txt','long_s6_h128.txt','long_s6_h129.txt',\
    'long_s6_h130.txt','long_s6_h131.txt','long_s6_h132.txt','long_s6_h133.txt','long_s6_h134.txt','long_s6_h135.txt','long_s7_h125.txt','long_s7_h128.txt',\
    'long_s7_h129.txt','long_s7_h130.txt','long_s7_h131.txt','long_s7_h132.txt','long_s7_h133.txt','long_s7_h134.txt','long_s7_h135.txt'
]

# long_s4_h130.txt 可能有bug 

for filePath in list_long:

    if os.path.getsize(filePath) == 0:
        continue
    try:
        FilterOutEmptyVector(filePath, 2050, backup=True)

        Merger("LongVideo.vec", filePath, deleteAfterMeger=True)
    except:
        print("error \t",filePath)