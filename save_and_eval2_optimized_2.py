import os
import numpy as np
import faiss
from PIL import Image
import os.path as osp

import itertools
from multiprocessing.pool import Pool


import os
import numpy as np
import faiss
from PIL import Image
import os.path as osp

from tqdm import tqdm

import itertools

from multiprocessing.pool import Pool
import parmap

import warnings
warnings.filterwarnings('ignore')


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


dict_host = {
        'h125': "/devdata/videos/", 'h128' : '/nfs/nfs128/videos', 'h129':'/nfs/nfs129/videos', 'h130':'/nfs/nfs130/videos', \
        'h131':'/nfs/nfs131/videos', 'h132':'/nfs/nfs132/videos', 'h133':'/nfs/nfs133/videos', 'h134':'/nfs/nfs134/videos',\
        'h135':'/nfs/nfs135/videos'}


# str_path='/devdata/videos/match_rst/asssemble_rst1.txt'
# temp = extract_embd(str_path,5) 

def extract_embd(str_path, ndim=2050):
    # vid_embedding = np.memmap(str_path, dtype='float16', mode='r', shape=None)
    vid_embedding = np.memmap(str_path, dtype='float32', mode='r', shape=None)
    vid_embedding = vid_embedding.reshape(-1, ndim)
    print("\t---- total rows: \t", vid_embedding.shape[0])
    
    return vid_embedding#.astype(np.float16)


def create_indexing(gpu_index, embding_data_root="/devdata/videos", long_vid_nm='long_s6_h125.txt'):
    # global gpu_index
    ####################################################
    print("long video preprocessing: \t")
    vid_long_all = extract_embd(osp.join(embding_data_root, long_vid_nm))
    vid_long = np.ascontiguousarray(vid_long_all[:, 2:])

    ####################################################
    print("creating indexing ...")
    gpu_index.add(vid_long)              # add vectors to the index
    print("total indexing \t",gpu_index.ntotal)
    return vid_long_all[:,0:2]


# black frame 
black_frame = np.loadtxt('black_frames.txt', dtype=np.float32, delimiter=",")

def is_valid_frame_new(short):
    try:
        return not np.sum(np.sum(np.square(short - black_frame), axis = 1) < 0.03 * 0.5) > 0
    except:
        return True

########################## 更改代码处 ##########################

def matching_frame(gpu_index, vid_long_all, embding_data_root, vid_short_nm, data_root_long, \
    data_root_short, save_root,  match_rst = 'temp'):
    # 从这两个文件夹处读取数据

    step = 5000

    print("short video preprocessing: \t")
    vid_short_all_total = extract_embd(osp.join(embding_data_root, vid_short_nm))

    counts_show = 0

    # list_long = list_long[0:5]
    dst_file = "/devdata/videos/match_rst/LongVideo_part0_4.txt"

    for m in tqdm(range(vid_short_all_total.shape[0] // step + 1)):
        
        if m != vid_short_all_total.shape[0] // step:
            vid_short_all = vid_short_all_total[m * step:(m + 1) * step, :]
        else:
            vid_short_all = vid_short_all_total[m * step:, :]
        vid_short = np.ascontiguousarray(vid_short_all[:, 2:])

        total_processing = vid_short.shape[0]

        D, I = gpu_index.search(vid_short, k = 1) 

        temp_D = D[D[:, 0] < 0.03]
        if temp_D.shape[0] == 0:
            continue

        filter_bool = parmap.map(is_valid_frame_new, vid_short, pm_processes=4)

        del vid_short
        np0 = np.array(list(zip(D[:,0], vid_short_all[:, 0], vid_short_all[:, 1], vid_long_all[I[:,0], 0], vid_long_all[I[:,0], 1])))
        np1 = np0[filter_bool]
        # np1 = np0
        del np0

        rst = np1[np1[:, 0] < 0.03]

        with open(dst_file, 'ab') as fw:  # fr读文件
            fw.write(rst)



def match_long_short(tuple_long_shortlist):
    # print('-' * 100)
    #print (tuple_long_shortlist)

    embding_data_root = "/devdata/videos/"

    save_root = "/devdata/videos/match_rst"

    long_vid_nm = tuple_long_shortlist[0]
    list_short = tuple_long_shortlist[1]

    str_host_long = long_vid_nm.split('_')[-1].split('.')[0]
    data_root_long = dict_host.get(str_host_long)

    # 检查文件是否存在
    if not os.path.exists(os.path.join(embding_data_root, long_vid_nm)):
        print("no long_vid_nm file \t", long_vid_nm)
        return

    # 检验long_vid embding rst quality
    if os.path.getsize(os.path.join(embding_data_root, long_vid_nm)) == 0:
        print("long embedding file size 0kb\t", os.path.join(embding_data_root, long_vid_nm))
        return
    else:
        print("long embedding file size \t", os.path.join(embding_data_root, long_vid_nm), '\t', \
        os.path.getsize(os.path.join(embding_data_root, long_vid_nm)))

    #--------------------------create gpu indexing --------------------------
    print("beginning to try create indexing")
    ngpus = faiss.get_num_gpus()
    print("number of GPUs:\t", ngpus)

    cpu_index = faiss.IndexFlatL2(2048)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    vid_long_all = create_indexing(gpu_index, embding_data_root=embding_data_root, long_vid_nm=long_vid_nm)

    for vid_short_nm in list_short:
        if not os.path.exists(os.path.join(embding_data_root, vid_short_nm)):
            print("no vid_short_nm file \t", vid_short_nm)
            continue

        str_host_short = vid_short_nm.split('_')[-1].split('.')[0] 
        data_root_short = dict_host.get(str_host_short)

        if os.path.getsize(os.path.join(embding_data_root, vid_short_nm)) == 0:
            print("short embedding file size 0kb\t", os.path.join(embding_data_root, vid_short_nm))
            return
        else:
            print("short embedding file size \t", os.path.join(embding_data_root, vid_short_nm), '\t', \
            os.path.getsize(os.path.join(embding_data_root, vid_short_nm)))

        #--------------------------create needed folders --------------------------
        match_rst = long_vid_nm.split('.')[0] + '_' + vid_short_nm.split('.')[0]
        str_match_rst_path = osp.join(save_root, match_rst)
        if not os.path.exists(str_match_rst_path):
            os.system("mkdir -p {}".format(str_match_rst_path))
        else:
            print("need to process !!!")
            #return

        matching_frame(gpu_index, vid_long_all=vid_long_all, embding_data_root=embding_data_root,\
        vid_short_nm=vid_short_nm, data_root_long=data_root_long, data_root_short=data_root_short,\
        save_root=save_root,  match_rst = match_rst)



# yy = np.loadtxt('black_frames.txt', dtype=np.float32, delimiter=",")
# temp_list=list(yy)
# print("length of black frame\t", len(temp_list))

# def add_black_frame(vid, fid, vid_long_all):
#     global temp_list
#     tensor_black = vid_long_all[np.logical_and(vid_long_all[:, 0] == int(vid), vid_long_all[:, 1] == int(fid))][:,2:]
#     temp_list.append(tensor_black[0])

#     black_frames = np.asanyarray(temp_list).reshape(len(temp_list), 2048)
#     np.savetxt(u'black_frames.txt', black_frames, fmt='%.8e', delimiter=",")

if __name__ == "__main__":

    list_long = ['LongVideo_part0','LongVideo_part1','LongVideo_part10','LongVideo_part11',\
    'LongVideo_part12','LongVideo_part13','LongVideo_part14','LongVideo_part15','LongVideo_part16',\
    'LongVideo_part17','LongVideo_part18','LongVideo_part19','LongVideo_part2','LongVideo_part20',\
    'LongVideo_part21','LongVideo_part22','LongVideo_part23','LongVideo_part24','LongVideo_part25',\
    'LongVideo_part26','LongVideo_part27','LongVideo_part28','LongVideo_part29','LongVideo_part3',\
    'LongVideo_part30','LongVideo_part31','LongVideo_part32','LongVideo_part33','LongVideo_part34',\
    'LongVideo_part35','LongVideo_part36','LongVideo_part37','LongVideo_part38','LongVideo_part4',\
    'LongVideo_part5','LongVideo_part6','LongVideo_part7','LongVideo_part8','LongVideo_part9']

    # list_long = [item for item in list_long if item.startswith('LongVideo_part1')]
    list_long = list_long[0:5]


    list_long.sort()
    print(list_long)

    list_short=['ShortVideo.vec']

    list_tuple_long_shortlist = [(long, list_short) for long in set(list_long)]
    
    print("total batch\t", len(list_tuple_long_shortlist))

    for tuple_long_shortlist in list_tuple_long_shortlist:
        print('-' * 100)
        print (tuple_long_shortlist)
        match_long_short(tuple_long_shortlist)

    
    #list_tuple_long_shortlist.sort()
    #p = Pool(processes=1)
    #i = p.map(match_long_short, list_tuple_long_shortlist)
