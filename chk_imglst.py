#!/usr/bin/env python
# coding=utf-8
import os
import os.path
import sys
import numpy as np
from PIL import Image
from lz import *
import cvbase as cvb
from IPython import embed

filename = "/home/wangxinglu/prj/few-shot/data/imglst/img10k.train.txt"
img_root_path = "/mnt/nfs1703/test/prj/few-shot/data/imagenet-raw/"
fp_in = open(filename, "r")
line = fp_in.readline()
image_label = line.strip().split()
fail = []
while line:
    image_label = line.strip().split()
    label = int(image_label[1])
    img_path = img_root_path + image_label[0]
    cls = img_path.split('/')[-2]
    if cls in fail: continue
    try:
        # print('-->', img_path)
        img = Image.open(img_path)
    except Exception as inst:
        print(inst)
        print(img_path)
        fail.append(cls)
    line = fp_in.readline()
fail = np.unique(fail)
np.savetxt('fail.txt', fail, fmt='%s')

# fail=[]
# filename = "/home/wangxinglu/prj/few-shot/data/imglst/img10k.test.txt"
# img_root_path = "/mnt/nfs1703/test/prj/few-shot/data/imagenet-raw/"
# fp_in = open(filename, "r")
# line = fp_in.readline()
# image_label = line.strip().split()
# while line:
#     image_label = line.strip().split()
#     label = int(image_label[1])
#     img_path = img_root_path + image_label[0]
#     cls = img_path.split('/')[-2]
#     if cls in fail: continue
#     try:
#         img = Image.open(img_path)
#     except Exception as inst:
#         print(inst)
#         print(img_path)
#         fail.append(cls)
#     line = fp_in.readline()
# fail=np.unique(fail)
# np.savetxt('fail.test.txt',fail,fmt='%s')
