import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from load import load_ground_truths, read_npy, write_npy
from frame import FrameGT
from shot import Shot
from nfov_util import get_nfov_img
from precompute import get_params, precompute
from config import *
from merge import *

ASPECT_RATIO = 1.778

def compile_shots(data, total_len, shot_len, heat_table):
    shot_list = []
    for i in range(min(heat_table.shape[0], total_len)):
        new_shot = Shot(i, i * shot_len, shot_len, heat_table)
        shot_list.append(new_shot)
        #print(new_shot.anchor)
    return shot_list

def cubicb_spline(shot1, shot2):
    M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]) 
    # print(M.shape)
    count = int((shot1.frame_count() + 1) / 2) + int(shot2.frame_count() / 2)

    steps = np.linspace(0, 1, count)

    anchor1 = shot1.anchor
    anchor2 = shot2.anchor
    if anchor1[1] - anchor2[1] > np.pi:
        anchor2 = shot2.anchor + np.array([0, np.pi * 2, 0])
    elif anchor2[1] - anchor1[1] > np.pi:
        anchor1 = shot1.anchor + np.array([0, np.pi * 2, 0])
    V = np.array([anchor1, anchor1 * 2/3 + anchor2 / 3, anchor2 * 2/3 + anchor1 / 3, anchor2])
    #print(V.shape)
    U = np.array([steps**3, steps**2, steps, np.ones(count)])
    #print(U.shape)
    ans = np.matmul(np.matmul(U.T, M), V)
    #print(ans.shape)
    return ans

#def stitch(shot1, shot2):
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    parser.add_argument("optimization", type=str)
    parser.add_argument("--length", type=int, default=150)
    opt = parser.parse_args()

    file_name = opt.file_name
    optimization = opt.optimization
    length = opt.length

    if os.path.exists('{}.npy'.format(file_name)):
        data = read_npy(file_name)
    else:
        data = load_ground_truths(file_name)
        write_npy(data, file_name)

    params_file_name = "{}_params_{}_{}_{}.npy".format(file_name, NUM_FOV, NUM_H, NUM_V)

    t1 = time.time()
    if os.path.exists(params_file_name):
        params = np.load(params_file_name)
    else:
        params = get_params(MIN_FOV, MAX_FOV, NUM_FOV, NUM_H, NUM_V)
        np.save(params_file_name, params)
    t2 = time.time()

    print("Got params")
    table_file_name = "{}_table_{}_{}_{}_{}.npy".format(file_name, SAMPLE_RATE, NUM_FOV, NUM_H, NUM_V)
    if os.path.exists(table_file_name):
        heat_table = np.load(table_file_name)
    else:
        heat_table = precompute(data, file_name, SAMPLE_RATE, params, write_masks=True)
        np.save(table_file_name, heat_table)
    t3 = time.time()

    print("Parameter time: {}".format(t2 - t1))
    print("Heat Table Time: {}".format(t3 - t2))

    img_path = '360_Saliency_dataset_2018ECCV/{}/*.jpg'.format(file_name)
    img_paths = glob.glob(img_path)
    vid_len = min(length, len(img_paths))

    width = 960
    height = int(width/ASPECT_RATIO)
    
    params_str = "_".join(str(x) for x in [file_name, optimization, length, SAMPLE_RATE, NUM_FOV, NUM_H, NUM_V, INIT_SHOT_LENGTH])
    vid_writer = cv2.VideoWriter('out/{}.avi'.format(params_str),cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    t4 = time.time()

    #shot_list = compile_shots(data, int(length / shot_len), shot_len, heat_table)
    if optimization == "merge":
        shot_list, cost_heat_by_iter, cost_trans_by_iter = solveByMerge(heat_table[:int(length/SAMPLE_RATE)], params)
    else:
        print("incorrect optimization")
    
    print("Shot list length: {}".format(len(shot_list)))
    t5 = time.time()
    print("Making shots time: {}".format(t5-t4))
    img_index = 0
    # plot values
    long_values = []
    lat_values = []
    fov_values = []
    cost_heat = []
    cost_fov = []
    cost_v = []
    cost_l = []
    for i in range(len(shot_list)):
        equi_imgs = []
        shot = shot_list[i]
        for j in range(shot.frame_count):
            equi_imgs.append(cv2.imread(img_paths[img_index + j]))
        nfov_imgs, path = shot.apply_to_imgs(equi_imgs, width, height)
        long_values = np.append(long_values, path[:, 0])
        lat_values = np.append(lat_values, path[:, 1])
        fov_values = np.append(fov_values, path[:, 2])
        for j in range(shot.length):
            cost_heat.append(shot.cost_h[j])
            cost_fov.append(shot.cost_fov[j])
            cost_v.append(shot.cost_v[j])
            cost_l.append(shot.cost_l[j])
        img_index += shot.frame_count
        for img in nfov_imgs:
            vid_writer.write(img)
        if i % 20 == 0:
            print(i) 
        '''equi_imgs = []
        for j in range(shot_len * 2):
            equi_imgs.append(cv2.imread(img_paths[i * shot_len + j]))
        if i == 0:
            print(int(shot_len/2))
            nfov_imgs = shot_list[0].apply_to_imgs(equi_imgs[:int(shot_len/2)], width, height)
            for img in nfov_imgs:
                vid_writer.write(img)

        coords = cubicb_spline(shot_list[i], shot_list[i+1])
        #print(coords.shape[0])
        for j in range(coords.shape[0]):
            nfov_img, _ = get_nfov_img(width, height, coords[j, 0], coords[j, 1], coords[j, 2], equi_imgs[int(shot_len/ 2) + j])
            vid_writer.write(nfov_img)
        if i % 20 == 0:
            print(i) '''
        #if i >= len(shot_list) - 4:
        #    nfov_imgs = shots[i + 3].apply_to_imgs(equi_imgs[:int(shot_len/2)], width, height)
        #equi_imgs = []
        #for j in range(shot_len):
        #    equi_imgs.append(cv2.imread(img_paths[i+j]))
        #nfov_imgs = new_shot.apply_to_imgs(equi_imgs, width, height)
        #print(nfov_imgs[0].shape)
        #for j in range(shot_len):
        #    vid_writer.write(nfov_imgs[j])
    domain = range(len(long_values))
    plt.figure()
    plt.plot(domain, long_values)
    plt.title('Longitude vs Frame')
    plt.xlabel('Frame')
    plt.ylim((-1.5 * np.pi, 1.5 * np.pi))
    plt.ylabel('Longitude')
    plt.savefig('plots/{}_long.png'.format(params_str))

    plt.figure()
    plt.plot(domain, lat_values)
    plt.title('Latitude vs Frame')
    plt.xlabel('Frame')
    plt.ylim((-np.pi/2, np.pi/2))
    plt.ylabel('Latitude')
    plt.savefig('plots/{}_lat.png'.format(params_str))

    plt.figure()
    plt.plot(domain, fov_values)
    plt.title('FOV vs Frame')
    plt.xlabel('Frame')
    plt.ylim((np.pi/4, 3*np.pi/4))
    plt.ylabel('FOV')
    plt.savefig('plots/{}_fov.png'.format(params_str))

    plt.figure()
    cost_heat = np.array(cost_heat)
    cost_fov = np.array(cost_fov)
    cost_v = np.array(cost_v)
    cost_l = np.array(cost_l)
    domain = range(len(cost_heat))
    plt.plot(domain, cost_heat + cost_fov + cost_v + cost_l, label='total')
    plt.plot(domain, cost_heat, label='heat')
    plt.plot(domain, cost_fov, label='fov')
    plt.plot(domain, cost_v, label='velocity')
    plt.plot(domain, cost_l, label='length')
    plt.legend()
    plt.title('Total Cost vs Frame')
    plt.xlabel('Frame')
    plt.ylabel('Cost')
    plt.savefig('plots/{}_cost.png'.format(params_str))

    plt.figure()
    domain = range(len(cost_heat_by_iter))
    plt.plot(domain, cost_heat_by_iter, label='segments')
    plt.plot(domain, cost_trans_by_iter, label='transition')
    plt.plot(domain, cost_heat_by_iter + cost_trans_by_iter, label='total')
    plt.legend()
    plt.title('Cost vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.savefig('plots/{}_cost_over_time.png'.format(params_str))

    vid_writer.release()
    t6 = time.time()
    print("Time to make video: {}".format(t6-t5))
    #plt.show()
    #rame = FrameGT(0, data[0])
    #equi_img = cv2.imread(img_paths[0])
    #nfov_img = frame.apply_to_img(equi_img, width)
    #plt.imshow(cv2.cvtColor(nfov_img, cv2.COLOR_BGR2GRAY))
    #plt.show()

    #frame1 = FrameGT(0, data[0])
    #frame1.print_mask()
    #print(frame1)
    #print(np.sum(frame1.gt))
    