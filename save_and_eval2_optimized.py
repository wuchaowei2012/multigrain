import os
import numpy as np
import faiss
from PIL import Image
import os.path as osp

import itertools
from multiprocessing.pool import Pool


dict_host = {
        'h125': "/devdata/videos/", 'h128' : '/nfs/nfs128/videos', 'h129':'/nfs/nfs129/videos', 'h130':'/nfs/nfs130/videos', \
        'h131':'/nfs/nfs131/videos', 'h132':'/nfs/nfs132/videos', 'h133':'/nfs/nfs133/videos', 'h134':'/nfs/nfs134/videos',\
        'h135':'/nfs/nfs135/videos'}


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
    
    # vid_embedding = vid_embedding[vid_embedding[:,0] != 0]
    # print("\t---- valid rows: \t", vid_embedding.shape[0])
    return vid_embedding#.astype(np.float16)


def create_indexing(gpu_index, embding_data_root="/devdata/videos", long_vid_nm='long_s6_h125.txt'):
    # global gpu_index
    # embding_data_root, default long video type
    ####################################################
    print("long video preprocessing: \t")
    vid_long_all = extract_embd(osp.join(embding_data_root, long_vid_nm))
    vid_long = np.ascontiguousarray(vid_long_all[:, 2:])

    ####################################################
    print("creating indexing ...")
    gpu_index.add(vid_long)              # add vectors to the index
    print("total indexing \t",gpu_index.ntotal)
    return vid_long_all[:,0:2]


########################## 更改代码处 ##########################
# data_root_long, data_root_short, embding_data_root, vid_short_nm , 读取文件
# save_root 保存路径

def matching_frame(gpu_index, vid_long_all, embding_data_root, vid_short_nm, data_root_long, \
    data_root_short, save_root,  match_rst = 'temp'):
    # global gpu_index
    # 从这两个文件夹处读取数据
    # embding_data_root="/devdata/videos"

    path_long = osp.join(data_root_long, "long_video_pic")
    path_short = osp.join(data_root_short, "short_video_pic")

    # black frame 
    black_frame = np.loadtxt('black_frames.txt', dtype=np.float32, delimiter=",")

    step = 1000

    print("short video preprocessing: \t")
    vid_short_all_total = extract_embd(osp.join(embding_data_root, vid_short_nm))

    counts_show = 0

    for m in range(vid_short_all_total.shape[0] // step):
        vid_short_all = vid_short_all_total[m * step:(m + 1) * step, :]
        vid_short = np.ascontiguousarray(vid_short_all[:, 2:])

        print("will search based on indexing", m * step, (m + 1) * step)

        D, I = gpu_index.search(vid_short, k = 1) # sanity check

        # 统计 D
        temp = D[:,0]

        # extract information in a specific format
        cal_shortemb_shortvid_shortfid_D0_longvid_longfid = list(zip(vid_short_all[:, 2:],vid_short_all[:, 0:2], I[:,0], D[:,0]))

        cal_shortemb_shortvid_shortfid_D0_longvid_longfid = \
            [[shortemb,int(short[0]), int(short[1]), d, int(vid_long_all[i, 0]), int(vid_long_all[i, 1])] for shortemb, short, i, d in cal_shortemb_shortvid_shortfid_D0_longvid_longfid]
        
        for shortemb, short_vid, short_fid, d0, long_vid, long_fid in cal_shortemb_shortvid_shortfid_D0_longvid_longfid:
            threshold_val = 0.03
            # safety_factor = 0.001
            if np.isnan(d0):
                continue
            if(d0 > threshold_val):
                continue
            if np.sum(np.sum(np.square(shortemb - black_frame), axis = 1) < 0.03) > 0:
                continue
            long_path_a = osp.join(osp.join(path_long, str(int(long_vid))), str(int(long_fid)).rjust(5, '0') + '.jpg')
            short_path_a = osp.join(osp.join(path_short, str(int(short_vid))), str(int(short_fid)).rjust(5, '0') + '.jpg')
            ########################################################################3
            path_save = osp.join(save_root, match_rst)

            save_nm_long = "{}_l_{}_{}_{}.jpg".format(counts_show, long_vid,long_fid,d0)
            save_nm_short = "{}_s_{}_{}_{}.jpg".format(counts_show, short_vid,short_fid, d0)

            os.system("cp {} {}/{}".format(long_path_a, path_save,save_nm_long))
            os.system("cp {} {}/{}".format(short_path_a, path_save, save_nm_short))

            counts_show+=1

        print(match_rst)
        print("counts,\t", len(temp),"\tmean\t",np.mean(temp))

        print("mean +- 3 sigma\t" ,np.mean(temp) + 3 * np.std(temp) , np.mean(temp) - 3 * np.std(temp))
        print("max / min", np.max(temp), '\t', np.min(temp))

        print('*' * 100)


def match_long_short(tuple_long_shortlist):
    print('-' * 100)
    print (tuple_long_shortlist)

    embding_data_root = "/devdata/videos/"
    save_root = "/devdata/videos/match_rst"

    long_vid_nm = tuple_long_shortlist[0]
    list_short = tuple_long_shortlist[1]

    str_host_long = long_vid_nm.split('_')[-1].split('.')[0]
    data_root_long = dict_host.get(str_host_long)

    # 检查文件是否存在
    if not os.path.exists(os.path.join(embding_data_root, long_vid_nm)):
        return

    # 检验long_vid embding rst quality
    if os.path.getsize(os.path.join(embding_data_root, long_vid_nm)) == 0:
        print("long embedding file size 0kb\t", os.path.join(embding_data_root, long_vid_nm))
        return
    else:
        print("long embedding file size \t", os.path.join(embding_data_root, long_vid_nm), '\t', \
        os.path.getsize(os.path.join(embding_data_root, long_vid_nm)))

    #--------------------------create gpu indexing --------------------------
    # --------------------------once for a list of short video----------------------
    ngpus = faiss.get_num_gpus()
    print("number of GPUs:\t", ngpus)

    cpu_index = faiss.IndexFlatL2(2048)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    vid_long_all = create_indexing(gpu_index, embding_data_root=embding_data_root, long_vid_nm=long_vid_nm)

    # long_vid_nm vid_short_nm 
    # long_s6_h131.txt
    for vid_short_nm in list_short:
        if not os.path.exists(os.path.join(embding_data_root, vid_short_nm)):
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
        # os.chdir(save_root)
        if not os.path.exists(str_match_rst_path):
            os.system("mkdir -p {}".format(str_match_rst_path))
        else:
            #os.system("rm {}/*jpg".format(str_match_rst_path))
            return

        matching_frame(gpu_index, vid_long_all=vid_long_all, embding_data_root=embding_data_root,\
        vid_short_nm=vid_short_nm, data_root_long=data_root_long, data_root_short=data_root_short,\
        save_root=save_root,  match_rst = match_rst)


# with open('/home/Code/Code/multigrain/vid_embedding_multiGrain.txt/string_vid_embedding128.txt', 'w') as f:
#     for idx in range(vid_embedding.shape[0]):
#         str = vid_embedding[idx][0].astype('int').astype('str')+'  '+','.join(vid_embedding[idx][1:].astype('str'))
#         f.write(str+'\n')


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

    list_long = ['long_s1_h125.txt','long_s1_h128.txt','long_s1_h129.txt','long_s1_h130.txt',\
    'long_s1_h131.txt','long_s1_h132.txt','long_s1_h133.txt','long_s1_h134.txt','long_s1_h135.txt',\
    'long_s2_h125.txt','long_s2_h128.txt','long_s2_h129.txt','long_s2_h130.txt','long_s2_h131.txt',\
    'long_s2_h132.txt','long_s2_h133.txt','long_s2_h134.txt','long_s2_h135.txt','long_s3_h125.txt',\
    'long_s3_h128.txt','long_s3_h129.txt','long_s3_h130.txt','long_s3_h131.txt','long_s3_h132.txt',\
    'long_s3_h133.txt','long_s3_h134.txt','long_s3_h135.txt','long_s4_h125.txt','long_s4_h128.txt',\
    'long_s4_h129.txt','long_s4_h130.txt','long_s4_h131.txt','long_s4_h132.txt','long_s4_h133.txt',\
    'long_s4_h134.txt','long_s4_h135.txt','long_s5_h125.txt','long_s5_h128.txt','long_s5_h129.txt',\
    'long_s5_h130.txt','long_s5_h131.txt','long_s5_h132.txt','long_s5_h133.txt','long_s5_h134.txt',\
    'long_s5_h135.txt','long_s6_h125.txt','long_s6_h128.txt','long_s6_h129.txt','long_s6_h130.txt',\
    'long_s6_h131.txt','long_s6_h132.txt','long_s6_h133.txt','long_s6_h134.txt','long_s6_h135.txt',\
    'long_s7_h125.txt','long_s7_h128.txt','long_s7_h129.txt','long_s7_h130.txt','long_s7_h131.txt',\
    'long_s7_h132.txt','long_s7_h133.txt','long_s7_h134.txt','long_s7_h135.txt','long_s8_h125.txt',\
    'long_s8_h128.txt','long_s8_h129.txt','long_s8_h130.txt','long_s8_h131.txt','long_s8_h132.txt',\
    'long_s8_h133.txt','long_s8_h134.txt','long_s8_h135.txt','long_s9_h125.txt','long_s9_h128.txt',\
    'long_s9_h129.txt','long_s9_h130.txt','long_s9_h131.txt','long_s9_h132.txt','long_s9_h133.txt',\
    'long_s9_h134.txt','long_s9_h135.txt']

    list_short=['short_s1_h125.txt','short_s1_h128.txt','short_s1_h129.txt','short_s1_h130.txt',\
    'short_s1_h131.txt','short_s1_h132.txt','short_s1_h133.txt','short_s1_h134.txt','short_s1_h135.txt',\
    'short_s2_h125.txt','short_s2_h128.txt','short_s2_h129.txt','short_s2_h130.txt','short_s2_h131.txt',\
    'short_s2_h132.txt','short_s2_h133.txt','short_s2_h134.txt','short_s2_h135.txt','short_s3_h125.txt',\
    'short_s3_h128.txt','short_s3_h129.txt','short_s3_h130.txt','short_s3_h131.txt','short_s3_h132.txt',\
    'short_s3_h133.txt','short_s3_h134.txt','short_s3_h135.txt','short_s4_h125.txt','short_s4_h128.txt',\
    'short_s4_h129.txt','short_s4_h130.txt','short_s4_h131.txt','short_s4_h132.txt','short_s4_h133.txt',\
    'short_s4_h134.txt','short_s4_h135.txt','short_s5_h125.txt','short_s5_h128.txt','short_s5_h129.txt',\
    'short_s5_h130.txt','short_s5_h131.txt','short_s5_h132.txt','short_s5_h133.txt','short_s5_h134.txt',\
    'short_s5_h135.txt','short_s6_h125.txt','short_s6_h128.txt','short_s6_h129.txt','short_s6_h130.txt',\
    'short_s6_h131.txt','short_s6_h132.txt','short_s6_h133.txt','short_s6_h134.txt','short_s6_h135.txt',\
    'short_s7_h125.txt','short_s7_h128.txt','short_s7_h129.txt','short_s7_h130.txt','short_s7_h131.txt',\
    'short_s7_h132.txt','short_s7_h133.txt','short_s7_h134.txt','short_s7_h135.txt','short_s8_h125.txt',\
    'short_s8_h128.txt','short_s8_h129.txt','short_s8_h130.txt','short_s8_h131.txt','short_s8_h132.txt',\
    'short_s8_h133.txt','short_s8_h134.txt','short_s8_h135.txt','short_s9_h125.txt','short_s9_h128.txt',\
    'short_s9_h129.txt','short_s9_h130.txt','short_s9_h131.txt','short_s9_h132.txt','short_s9_h133.txt',\
    'short_s9_h134.txt','short_s9_h135.txt']

    # long_short_pair =list(itertools.product(list_long,list_short))

    list_tuple_long_shortlist = [(long, list_short) for long in list_long]

    # for tuple_long_shortlist in list_tuple_long_shortlist:
    #     print('-' * 100)
    #     print (tuple_long_shortlist)
    #     match_long_short(tuple_long_shortlist)

    
    list_tuple_long_shortlist.sort()
    p = Pool(processes=3)
    i = p.map(match_long_short, list_tuple_long_shortlist)
