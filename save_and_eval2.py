import os
import numpy as np
import faiss
from PIL import Image
import os.path as osp


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


########################## 更改代码处 ##########################

# 可用于合并两个numpy memmap
#line_counts = vid_long_all0.shape[0] + vid_long_all1.shape[0]
#vid_embedding_dir = os.path.join(embedding_dir, '3_vid_embedding_multiGrain.txt')
#vid_long_all = np.memmap(vid_embedding_dir, dtype='float64', mode='r+', shape=(line_counts,2050))
#vid_long_all[vid_long_all0.shape[0]:,:] = vid_long_all1
#del vid_long_all1,  vid_long_all0


def create_indexing(data_root="/devdata/videos", long_vid_nm='long_s6_h125.txt'):
    global gpu_index
    ####################################################
    print("long video preprocessing: \t")
    vid_long_all = extract_embd(osp.join(data_root, long_vid_nm))
    vid_long = np.ascontiguousarray(vid_long_all[:, 2:])

    ####################################################
    print("creating indexing ...")
    gpu_index.add(vid_long)              # add vectors to the index
    print("total indexing \t",gpu_index.ntotal)
    return vid_long_all[:,0:2]


# 生成新的子数据集
#xb = MmapVectorUtils.Open("/home/meizi/3_vid_embedding_multiGrain_1million.txt", True, shape=(1000000,2050))
#xb[0:1000000,:] = vid_long_all[:1000000,:]
########################## 更改代码处 ##########################
def matching_frame(vid_long_all, save_root="/root/Fred_wu/Code/multigrain", data_root="/devdata/videos/", \
    vid_short_nm = 'short4_125.txt', match_rst = 'temp'):
    global gpu_index

    path_long = osp.join(data_root, "long_video_pic")
    path_short = osp.join(data_root, "short_video_pic")

    print("short video preprocessing: \t")
    vid_short_all_total = extract_embd(osp.join(data_root, vid_short_nm))

    step = 1000

    # black frame 
    black_frame = np.loadtxt('black_frames.txt', dtype=np.float32, delimiter=",")

    counts_show = 0

    for m in range(vid_short_all_total.shape[0] // step):
        vid_short_all = vid_short_all_total[m * step:(m + 1) * step, :]
        vid_short = np.ascontiguousarray(vid_short_all[:, 2:])

        print("will search based on indexing", m * step, (m + 1) * step)

        D, I = gpu_index.search(vid_short, k = 5) # sanity check

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
            # rel_im = Image.open(long_path_a)
            # cal_im = Image.open(short_path_a)

            path_save = osp.join(save_root, match_rst)

            save_nm_long = "{}_long_vid_{}_{}_{}.jpg".format(counts_show, long_vid,long_fid,d0)
            save_nm_short = "{}_short_vid_{}.jpg".format(counts_show, d0)

            os.system("cp {} {}/{}".format(long_path_a, path_save,save_nm_long))
            os.system("cp {} {}/{}".format(short_path_a, path_save, save_nm_short))
            
            # rel_im.save('{}/{}/{}_real_picture_{}_{}_{}.jpg'.\
            #     format(save_root, match_rst,counts_show, long_vid,long_fid,d0))
            # cal_im.save('{}/{}/{}_short_vid_{}.jpg'.\
            #     format(save_root,match_rst, counts_show, d0))

            counts_show+=1

        print(match_rst)
        print("counts,\t", len(temp),"\tmean\t",np.mean(temp))

        print("mean +- 3 sigma\t" ,np.mean(temp) + 3 * np.std(temp) , np.mean(temp) - 3 * np.std(temp))
        print("max / min", np.max(temp), '\t', np.min(temp))

        print('*' * 100)


if __name__ == "__main__":
    data_root = "/devdata/videos/"
    save_root = "/root/Fred_wu/multigrain_data/multigrain"
    long_vid_nm='long_s6_h125.txt'
    vid_short_nm='short_s6_h125.txt'

    #--------------------------create needed folders --------------------------
    match_rst = long_vid_nm.split('.')[0] + '_' + vid_short_nm.split('.')[0]
    str_match_rst_path = osp.join(save_root, match_rst)
    # os.chdir(save_root)
    if not os.path.exists(str_match_rst_path):
        os.system("mkdir -p {}".format(str_match_rst_path))
    else:
        os.system("rm {}/*jpg".format(str_match_rst_path))

    #--------------------------create gpu indexing --------------------------
    ngpus = faiss.get_num_gpus()
    print("number of GPUs:\t", ngpus)

    cpu_index = faiss.IndexFlatL2(2048)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)


    vid_long_all = create_indexing(data_root=data_root, long_vid_nm=long_vid_nm)

    matching_frame(vid_long_all=vid_long_all, save_root=save_root, data_root=data_root,vid_short_nm=vid_short_nm, match_rst = match_rst)




# with open('/home/Code/Code/multigrain/vid_embedding_multiGrain.txt/string_vid_embedding128.txt', 'w') as f:
#     for idx in range(vid_embedding.shape[0]):
#         str = vid_embedding[idx][0].astype('int').astype('str')+'  '+','.join(vid_embedding[idx][1:].astype('str'))
#         f.write(str+'\n')


# yy = np.loadtxt('black_frames.txt', dtype=np.float32, delimiter=",")
# temp_list=list(yy)
# print("length of black frame\t", len(temp_list))

def add_black_frame(vid, fid, vid_long_all=vid_long_all):
    global temp_list
    tensor_black = vid_long_all[np.logical_and(vid_long_all[:, 0] == int(vid), vid_long_all[:, 1] == int(fid))][:,2:]
    temp_list.append(tensor_black[0])

    black_frames = np.asanyarray(temp_list).reshape(len(temp_list), 2048)
    np.savetxt(u'black_frames.txt', black_frames, fmt='%.8e', delimiter=",")

