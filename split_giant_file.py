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


# def extract_embd(str_path):    
    
#     # delete null lines
#     FilterOutEmptyVector(str_path, 2050, backup=False)

#     vid_embedding = np.memmap(str_path, dtype='float32', mode='r', shape=None)
#     vid_embedding = vid_embedding.reshape(-1, 2050)
#     print("\t---- total rows: \t", vid_embedding.shape[0])

#     return vid_embedding[vid_embedding[:,0] != 0]#.astype(np.float16)


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


def split_giant_new(filePath, dim, ends_digit):
    xi, fi,  xd = MmapVectorUtils.Read(filePath,dim)
    count = sum([1 for i in xi if i % 10 == ends_digit])
    print("total count which ends with {}: \t {} ".format(ends_digit, count))

    tempFileName = filePath.split('.')[0] + "_part" + str(ends_digit)

    # def Read(path, dim, copyToMemory = False)
    xd1 = MmapVectorUtils.Open(tempFileName, True, (count, dim))
    
    j = 0
    for i in tqdm(range(len(xi))):
        if ends_digit == xi[i] % 10:
            xd1[j,0] = xi[i]
            xd1[j,1] = fi[i]
            xd1[j,2:] = xd[i]
            j+=1


# vid_giant_file = extract_embd(path_giant_file)

# def split_giant(digit_list, str_file_nm_new):

#     array_vid = vid_giant_file[:,0].astype('int') 
#     array_last_digit = np.apply_along_axis(lambda x:x%10, 0, array_vid)

#     print("extract last digt ...")
#     array_filter = array_last_digit == digit_list[0]

#     for digit in digit_list[1:]:
#         array_filter = np.logical_or(array_filter, array_last_digit == digit)

#     rst_temp = vid_giant_file[array_filter]
    
#     path_part_file = os.path.join(embding_data_root, str_file_nm_new)
#     print("trying to save into disk ...")
#     xb = MmapVectorUtils.Open(path_part_file, True, shape=rst_temp.shape)

#     xb[:,:] = np.array(rst_temp).reshape(rst_temp.shape[0],rst_temp.shape[1])

#     print("smaller file shape \t", rst_temp.shape)


# 可用于合并两个numpy memmap
#line_counts = vid_long_all0.shape[0] + vid_long_all1.shape[0]
#vid_embedding_dir = os.path.join(embedding_dir, '3_vid_embedding_multiGrain.txt')
#vid_long_all = np.memmap(vid_embedding_dir, dtype='float64', mode='r+', shape=(line_counts,2050))
#vid_long_all[vid_long_all0.shape[0]:,:] = vid_long_all1
#del vid_long_all1,  vid_long_all0

# print("0 ...")
# split_giant(digit_list=[0], str_file_nm_new='long_s4_h132_part0.txt')
# split_giant(digit_list=[1], str_file_nm_new='long_s4_h132_part1.txt')
# print("2 ...")
# split_giant(digit_list=[2], str_file_nm_new='long_s4_h132_part2.txt')
# split_giant(digit_list=[3], str_file_nm_new='long_s4_h132_part3.txt')
# print("4 ...")
# split_giant(digit_list=[4], str_file_nm_new='long_s4_h132_part4.txt')
# split_giant(digit_list=[5], str_file_nm_new='long_s4_h132_part5.txt')
# print("6 ...")
# split_giant(digit_list=[6], str_file_nm_new='long_s4_h132_part6.txt')
# split_giant(digit_list=[7], str_file_nm_new='long_s4_h132_part7.txt')
# print("8 ...")
# split_giant(digit_list=[8], str_file_nm_new='long_s4_h132_part8.txt')
# split_giant(digit_list=[9], str_file_nm_new='long_s4_h132_part9.txt')

# 生成新的子数据集

embding_data_root = "/devdata/videos/"
str_file_nm="long_s4_h133.txt"
path_giant_file = os.path.join(embding_data_root, str_file_nm)

for ends_digit in range(10):
    # print("ends_digit:\t",ends_digit)
    split_giant_new(path_giant_file, 2050, ends_digit)



# 可用于合并两个numpy memmap
#line_counts = vid_long_all0.shape[0] + vid_long_all1.shape[0]
#vid_embedding_dir = os.path.join(embedding_dir, '3_vid_embedding_multiGrain.txt')
#vid_long_all = np.memmap(vid_embedding_dir, dtype='float64', mode='r+', shape=(line_counts,2050))
#vid_long_all[vid_long_all0.shape[0]:,:] = vid_long_all1
#del vid_long_all1,  vid_long_all0


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
