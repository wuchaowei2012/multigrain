cd /root/Fred_wu/Code/multigrain/data/long_video_pic/28710


ls | xargs -i ffmpeg -i {} -vf scale=320:240 new{}
