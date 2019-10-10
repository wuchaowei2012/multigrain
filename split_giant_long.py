import os
import numpy as np
import faiss
from PIL import Image
import os.path as osp

from tqdm import tqdm

import itertools

from multiprocessing.pool import Pool


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


# def split_giant_new(filePath, dim, ends_digit):
#     xi, fi,  xd = MmapVectorUtils.Read(filePath,dim)
#     count = sum([1 for i in xi if i % 10 == ends_digit])
#     print("total count which ends with {}: \t {} ".format(ends_digit, count))

#     tempFileName = filePath.split('.')[0] + "_part" + str(ends_digit)

#     # def Read(path, dim, copyToMemory = False)
#     xd1 = MmapVectorUtils.Open(tempFileName, True, (count, dim))
    
#     j = 0
#     for i in tqdm(range(count)):
#         if ends_digit == xi[i] % 10:
#             xd1[j,0] = xi[i]
#             xd1[j,1] = fi[i]
#             xd1[j,2:] = xd[i]
#             j+=1

def split_giant_new(filePath, dim):
    xi, fi,  xd = MmapVectorUtils.Read(filePath,dim)

    file_length = 2000000
    
    # ends_digit = 0
    # tempFileName = filePath.split('.')[0] + "_part" + str(ends_digit)
    # xd1 = MmapVectorUtils.Open(tempFileName, True, (file_length, dim))    
    # j  = 0
    ends_digit = -1

    for i in tqdm(range(xi.shape[0])):
        if i % file_length !=0:
            xd1[j,0] = xi[i]
            xd1[j,1] = fi[i]
            xd1[j,2:] = xd[i]
            j+=1
        else:
            j = 0
            ends_digit = ends_digit + 1
            tempFileName = filePath.split('.')[0] + "_part" + str(ends_digit)
            xd1 = MmapVectorUtils.Open(tempFileName, True, (file_length, dim))

            xd1[j,0] = xi[i]
            xd1[j,1] = fi[i]
            xd1[j,2:] = xd[i]
            j+=1
    

# 生成新的子数据集

embding_data_root = "/devdata/videos/"
str_file_nm="LongVideo.vec"
path_giant_file = os.path.join(embding_data_root, str_file_nm)

ShortVideo

split_giant_new(path_giant_file, 2050)


# -rw-r--r--     1 root root   38G Oct  4 20:04 long_s3_h125.txt
# -rw-r--r--     1 root root   37G Oct  7 00:59 long_s4_h132.txt
# -rw-r--r--     1 root root   32G Oct  5 23:28 long_s3_h135.txt
# -rw-r--r--     1 root root   29G Oct  5 23:24 long_s5_h129.txt
# -rw-r--r--     1 root root   22G Oct  7 05:17 long_s6_h128.txt
# -rw-r--r--     1 root root   22G Oct  5 13:53 long_s5_h130.txt

# -rw-r--r--     1 root root   17G Oct  3 20:32 long_s5_h125.txt
# -rw-r--r--     1 root root   14G Oct  6 18:23 long_s3_h128.txt
# -rw-r--r--     1 root root   13G Oct  5 07:52 long_s4_h130.txt
# -rw-r--r--     1 root root   12G Oct  5 07:51 long_s3_h130.txt
# -rw-r--r--     1 root root   12G Oct  6 04:42 long_s4_h135.txt
