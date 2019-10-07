import os
import numpy as np
import faiss
from PIL import Image
import os.path as osp

import itertools

from multiprocessing.pool import Pool


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


def extract_embd(str_path):
    # vid_embedding = np.memmap(str_path, dtype='float16', mode='r', shape=None)
    vid_embedding = np.memmap(str_path, dtype='float32', mode='r', shape=None)
    vid_embedding = vid_embedding.reshape(-1, 2050)
    print("\t---- total rows: \t", vid_embedding.shape[0])
    
    vid_embedding = vid_embedding[vid_embedding[:,0] != 0]
    print("\t---- valid rows: \t", vid_embedding.shape[0])
    return vid_embedding#.astype(np.float16)

# test_path = "/devdata/videos/test1234"
# rst = extract_embd(test_path)

embding_data_root = "/devdata/videos/"

def split_giant(digit_list, str_file_nm_new, str_file_nm="short_s7_h128.txt"):

    path__giant_file = os.path.join(embding_data_root, str_file_nm)
    vid_giant_file = extract_embd(path__giant_file)

    array_vid = vid_giant_file[:,0].astype('int') 
    array_last_digit = np.apply_along_axis(lambda x:x%10, 0, array_vid)

    array_filter = array_last_digit == digit_list[0]

    for digit in digit_list[1:]:
        array_filter = np.logical_or(array_filter, array_last_digit == digit)

    rst_temp = vid_giant_file[array_filter]
    
    path_part_file = os.path.join(embding_data_root, str_file_nm_new)

    xb = MmapVectorUtils.Open(path_part_file, True, shape=rst_temp.shape)

    xb[:,:] = np.array(rst_temp).reshape(rst_temp.shape[0],rst_temp.shape[1])

    print("smaller file shape \t", rst_temp.shape)


#xb = MmapVectorUtils.Open("/home/meizi/3_vid_embedding_multiGrain_1million.txt", True, shape=(1000000,2050))
#xb[0:1000000,:] = vid_long_all[:1000000,:]

# 可用于合并两个numpy memmap
#line_counts = vid_long_all0.shape[0] + vid_long_all1.shape[0]
#vid_embedding_dir = os.path.join(embedding_dir, '3_vid_embedding_multiGrain.txt')
#vid_long_all = np.memmap(vid_embedding_dir, dtype='float64', mode='r+', shape=(line_counts,2050))
#vid_long_all[vid_long_all0.shape[0]:,:] = vid_long_all1
#del vid_long_all1,  vid_long_all0


split_giant(digit_list=[1,2,3, 4], str_file_nm_new='test1234')
# 生成新的子数据集

# -rw-r--r--     1 root root   54G Oct  6 10:39 long_s4_h134.txt
# -rw-r--r--     1 root root   51G Oct  3 15:34 short_s3_h125.txt
# -rw-r--r--     1 root root   41G Oct  5 09:12 long_s4_h125.txt
# -rw-r--r--     1 root root   41G Oct  5 19:07 long_s4_h131.txt
# -rw-r--r--     1 root root   38G Oct  6 06:14 long_s4_h133.txt
# -rw-r--r--     1 root root   38G Oct  4 20:04 long_s3_h125.txt
# -rw-r--r--     1 root root   37G Oct  7 00:59 long_s4_h132.txt
# -rw-r--r--     1 root root   32G Oct  5 23:28 long_s3_h135.txt
# -rw-r--r--     1 root root   32G Oct  3 05:26 short_s2_h125.txt
# -rw-r--r--     1 root root   29G Oct  5 23:24 long_s5_h129.txt
# -rw-r--r--     1 root root   23G Oct  6 23:01 short_s4_h128.txt
# -rw-r--r--     1 root root   22G Oct  7 05:17 long_s6_h128.txt
# -rw-r--r--     1 root root   22G Oct  5 13:53 long_s5_h130.txt
# -rw-r--r--     1 root root   22G Sep 30 20:55 short_s4_h125.txt
# -rw-r--r--     1 root root   20G Oct  7 05:20 short_s4_h135.txt
# -rw-r--r--     1 root root   19G Oct  6 09:35 short_s4_h132.txt
# -rw-r--r--     1 root root   19G Oct  6 05:52 short_s4_h130.txt
# -rw-r--r--     1 root root   19G Oct  6 22:07 short_s4_h134.txt
# -rw-r--r--     1 root root   19G Oct  6 09:49 short_s4_h133.txt
# -rw-r--r--     1 root root   19G Oct  6 07:03 short_s4_h131.txt
# -rw-r--r--     1 root root   18G Oct  7 04:53 short_s5_h128.txt
# -rw-r--r--     1 root root   17G Oct  3 20:32 long_s5_h125.txt
# -rw-r--r--     1 root root   15G Oct  7 17:20 short_s5_h134.txt
# -rw-r--r--     1 root root   15G Oct  7 01:51 short_s5_h135.txt
# -rw-r--r--     1 root root   15G Oct  6 10:47 short_s5_h130.txt
# -rw-r--r--     1 root root   15G Oct  6 11:53 short_s5_h131.txt
# -rw-r--r--     1 root root   14G Oct  6 17:35 short_s5_h133.txt
# -rw-r--r--     1 root root   14G Oct  6 18:23 long_s3_h128.txt
# -rw-r--r--     1 root root   14G Oct  6 17:05 short_s5_h132.txt
# -rw-r--r--     1 root root   13G Oct  5 07:52 long_s4_h130.txt
# -rw-r--r--     1 root root   13G Oct  6 19:09 short_s4_h129.txt
# -rw-r--r--     1 root root   12G Oct  5 07:51 long_s3_h130.txt
# -rw-r--r--     1 root root   12G Oct  6 04:42 long_s4_h135.txt
# -rw-r--r--     1 root root   11G Oct  6 22:37 short_s5_h129.txt








# 可用于合并两个numpy memmap
#line_counts = vid_long_all0.shape[0] + vid_long_all1.shape[0]
#vid_embedding_dir = os.path.join(embedding_dir, '3_vid_embedding_multiGrain.txt')
#vid_long_all = np.memmap(vid_embedding_dir, dtype='float64', mode='r+', shape=(line_counts,2050))
#vid_long_all[vid_long_all0.shape[0]:,:] = vid_long_all1
#del vid_long_all1,  vid_long_all0