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

#embedding_file_nm = '5_vid_embedding_multiGrain.txt'

embedding_dir = "/devdata/videos/"
embedding_dir = "/home/meizi/"

def extract_embd(str_path):
    vid_embedding_dir = os.path.join(embedding_dir, str_path)
    vid_embedding_long = np.memmap(vid_embedding_dir, dtype='float32', mode='r', shape=None)
    # vid_embedding 代表长短片所有的embedding
    return vid_embedding_long.reshape(-1, 2050)

########################## 更改代码处 ##########################

# 可用于合并两个numpy memmap
#line_counts = vid_long_all0.shape[0] + vid_long_all1.shape[0]
#vid_embedding_dir = os.path.join(embedding_dir, '3_vid_embedding_multiGrain.txt')
#vid_long_all = np.memmap(vid_embedding_dir, dtype='float64', mode='r+', shape=(line_counts,2050))
#vid_long_all[vid_long_all0.shape[0]:,:] = vid_long_all1
#del vid_long_all1,  vid_long_all0

#vid_long_all = extract_embd('s3_sv125_vid_embedding_multiGrain_new.txt')
# 3_vid_embedding_multiGrain_1million is a subste of s3_sv125_vid_embedding_multiGrain_new
vid_long_all = extract_embd('3_vid_embedding_multiGrain_1million.txt')

# 生成新的子数据集
#xb = MmapVectorUtils.Open("/home/meizi/3_vid_embedding_multiGrain_1million.txt", True, shape=(1000000,2050))
#xb[0:1000000,:] = vid_long_all[:1000000,:]
########################## 更改代码处 ##########################
embedding_file_nm = 'short33_125.txt'

match_rst = 'short33_125'
os.system("rm /home/Code/Code/multigrain/{}/*jpg".format(match_rst))
####################################################
vid_short_all_total = extract_embd(embedding_file_nm)

print("total short embedding file\t", vid_short_all_total.shape)
vid_short_all_total = vid_short_all_total[vid_short_all_total[:,0] !=0]
print("valid short embedding file\t", vid_short_all_total.shape)

vid_long = np.ascontiguousarray(vid_long_all[:, 2:])

index = faiss.IndexFlatL2(2048)   # build the index
print(index.is_trained)

print("will add indexing")
index.add(vid_long)                # add vectors to the index
print("index.ntotal \t",index.ntotal)

step = 1000

# black frame 
black_frame = np.loadtxt('black_frames.txt', dtype=np.float32, delimiter=",")

counts_show = 0

for m in range(vid_short_all_total.shape[0] // step):
    vid_short_all = vid_short_all_total[m * step:(m + 1) * step, :]

    vid_short = np.ascontiguousarray(vid_short_all[:, 2:])

    print("will search based on indexing", m * step, (m + 1) * step)
    print("query input: \t", vid_short_all.shape)

    D, I = index.search(vid_short, k = 5) # sanity check

    # 统计 D
    temp = D[:,0]

    # extract information in a specific format
    cal_shortemb_shortvid_shortfid_D0_longvid_longfid = list(zip(vid_short_all[:, 2:],vid_short_all[:, 0:2], I[:,0], D[:,0]))

    cal_shortemb_shortvid_shortfid_D0_longvid_longfid = \
        [[shortemb,int(short[0]), int(short[1]), d, int(vid_long_all[i, 0]), int(vid_long_all[i, 1])] for shortemb, short, i, d in cal_shortemb_shortvid_shortfid_D0_longvid_longfid]

    path_long = '/home/meizi/long_video_pic/'
    path_short = '/home/meizi/short_video_pic/'
    for shortemb, short_vid, short_fid, d0, long_vid, long_fid in cal_shortemb_shortvid_shortfid_D0_longvid_longfid:
        threshold_val = 0.003
        # safety_factor = 0.001
        if(d0 > threshold_val):
            continue

        if np.sum(np.sum(np.square(shortemb - black_frame), axis = 1) < 0.002) > 0:
            continue

        long_path_a = osp.join(osp.join(path_long, str(int(long_vid))), str(int(long_fid)).rjust(5, '0') + '.jpg')

        if embedding_file_nm == '5_vid_embedding_multiGrain.txt':
            short_path_a = osp.join(osp.join(path_short, str(int(short_vid))), str(int(short_fid)).rjust(6, '0') + '.jpg')
        else:
            short_path_a = osp.join(osp.join(path_short, str(int(short_vid))), str(int(short_fid)).rjust(5, '0') + '.jpg')
        
        rel_im = Image.open(long_path_a)

        cal_im = Image.open(short_path_a)
        
        rel_im.save('/home/Code/Code/multigrain/{}/{}_real_picture_{}_{}_{}.jpg'.\
            format(match_rst,counts_show, long_vid,long_fid,d0))

        cal_im.save('/home/Code/Code/multigrain/{}/{}rm {}.jpg'.\
            format(match_rst, counts_show, d0))

        print("short_frame_info\t", short_vid, short_fid, "position in the folder", counts_show)
        print("long_frame_info\t", long_vid, long_fid, "position in the folder", counts_show)

        counts_show+=1

        print("success", long_path_a)
        print("success", short_path_a)
        #except:
        #    print("error long_path_a \t", long_path_a)
        #    print("error short_path_a \t", short_path_a)
        #    continue

    print('-' * 50)
    print(match_rst)
    print("counts,\t", len(temp),"\tmean\t",np.mean(temp))

    print("mean +- 3 sigma\t" ,np.mean(temp) + 3 * np.std(temp) , np.mean(temp) - 3 * np.std(temp))
    print("max / min", np.max(temp), '\t', np.min(temp))

    print('*' * 100)

# with open('/home/Code/Code/multigrain/vid_embedding_multiGrain.txt/string_vid_embedding128.txt', 'w') as f:
#     for idx in range(vid_embedding.shape[0]):
#         str = vid_embedding[idx][0].astype('int').astype('str')+'  '+','.join(vid_embedding[idx][1:].astype('str'))
#         f.write(str+'\n')

