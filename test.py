import numpy as np
import matplotlib.pyplot as plt
from load import load_ground_truths, write_npy, read_npy, display_gts
from nfov_util import get_nfov_img, get_equi_img
import cv2
import time
import glob
import sys

if __name__ == '__main__':
    for j in range(272, 302):
        data = load_ground_truths(j)
        #print(data.shape)
        vid_writer = cv2.VideoWriter('ground_truths/{}_gt.avi'.format(j),cv2.VideoWriter_fourcc(*'DIVX'), 30, (data.shape[2], data.shape[1]), isColor=False)
        #write_npy(data, j)
        for i in range(data.shape[0]):
            ratio = 255 / np.max(data[i])
            vid_writer.write(np.uint8(data[i] * ratio))
        vid_writer.release()
        print("done gt", j)
        img_path = '360_Saliency_dataset_2018ECCV/{}/*.jpg'.format(j)
        img_paths = glob.glob(img_path)
        vid_writer = cv2.VideoWriter('raw/{}_raw.avi'.format(j),cv2.VideoWriter_fourcc(*'DIVX'), 30, (480, 240), isColor=True)
        for path in img_paths:
            img = cv2.imread(path)
            vid_writer.write(cv2.resize(img, (480, 240)))
        vid_writer.release()
        print("done img", j)

    

