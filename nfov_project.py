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
from precompute import get_params, precompute, normalize_table
from config import *
from merge import *
from pyramid import *

ASPECT_RATIO = 1.778
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def compile_shots(data, total_len, shot_len, heat_table):
    shot_list = []
    for i in range(min(heat_table.shape[0], total_len)):
        new_shot = Shot(i, i * shot_len, shot_len, heat_table)
        shot_list.append(new_shot)
        #print(new_shot.anchor)
    return shot_list

#perform cubic b spline patching
def refine_path(shot1, shot2, path):
    M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]) 
    # print(M.shape)
    count = int(shot1.frame_count / 3) + int(shot2.frame_count / 3)

    steps = np.linspace(0, 1, count)
    anchor1s = path[shot1.start * SAMPLE_RATE]#np.append(shot1.start_pos, shot1.start_fov)
    anchor1e = path[shot2.start * SAMPLE_RATE - 1]#np.append(shot1.end_pos, shot1.end_fov)
    anchor2s = path[shot2.start * SAMPLE_RATE]#np.append(shot2.start_pos, shot2.start_fov)
    anchor2e = path[shot2.start * SAMPLE_RATE + shot2.frame_count]#np.append(shot2.end_pos, shot2.end_fov)

    V = np.array([anchor1e * 2/3 + anchor1s * 1/3, 2 * anchor1e / 3 + anchor2s / 3, 2 * anchor2s / 3 + anchor1e / 3, anchor2s * 2/3 + anchor2e / 3])
    #print(V.shape)
    U = np.array([steps**3, steps**2, steps, np.ones(count)])
    #print(U.shape)
    ans = np.matmul(np.matmul(U.T, M), V)
    #print(ans.shape)
    start = shot1.start * SAMPLE_RATE + int(2/3 * shot1.frame_count)
    path[start:start+count] = ans
    return path
#def stitch(shot1, shot2):
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    parser.add_argument("optimization", type=str)
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument("--table", type=int, default=REGULAR)
    parser.add_argument("--tag", type=str, default="")
    opt = parser.parse_args()

    file_name = opt.file_name
    optimization = opt.optimization
    length = opt.length
    table_type = opt.table
    tag = opt.tag
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
    print(params.shape)
    print("Got params")
    table_file_name = "{}_table_{}_{}_{}_{}_{}.npy".format(file_name, table_type, SAMPLE_RATE, NUM_FOV, NUM_H, NUM_V)
    if os.path.exists(table_file_name):
        heat_table = np.load(table_file_name)
    elif table_type == 0:
        heat_table = precompute(data, file_name, SAMPLE_RATE, params, write_masks=True)
        np.save(file_name, heat_table)
    elif table_type == 1:
        temp_name = "{}_table_{}_{}_{}_{}_{}.npy".format(file_name, REGULAR, SAMPLE_RATE, NUM_FOV, NUM_H, NUM_V)
        if os.path.exists(temp_name):
            print("triggered")
            heat_table = np.load(temp_name)
        else:
            heat_table = precompute(data, file_name, SAMPLE_RATE, params, write_masks=True)
            np.save(temp_name, heat_table)
        heat_table = normalize_table(heat_table)
        np.save(table_file_name, heat_table)
    
    t3 = time.time()

    print("Parameter time: {}".format(t2 - t1))
    print("Heat Table Time: {}".format(t3 - t2))

    img_path = '360_Saliency_dataset_2018ECCV/{}/*.jpg'.format(file_name)
    img_paths = glob.glob(img_path)
    vid_len = min(length, len(img_paths))

    width = 960
    height = int(width/ASPECT_RATIO)
    
    params_str = "_".join(str(x) for x in [file_name, optimization, table_type, length, SAMPLE_RATE, NUM_FOV, NUM_H, NUM_V, INIT_SHOT_LENGTH, tag])
    vid_writer = cv2.VideoWriter('out/{}.avi'.format(params_str),cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    t4 = time.time()

    os.makedirs('out_images/{}'.format(params_str), exist_ok=True)
    #shot_list = compile_shots(data, int(length / shot_len), shot_len, heat_table)
    if optimization == "merge":
        shot_list, cost_heat_by_iter, cost_trans_by_iter = solveByMerge(heat_table[:int(length/SAMPLE_RATE)], table_type)
    elif optimization == "pyramid":
        shot_list, cost_heat_by_iter, cost_trans_by_iter = solveByPyramid(heat_table[:int(length/SAMPLE_RATE)], table_type)
    elif optimization == 'center':
        shot_list = [Shot(0, int(length/SAMPLE_RATE), heat_table, table_type, PYRAMID, np.array([0,0,MAX_FOV]), np.array([0,0,MAX_FOV]))]    
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
    counter = []
    if not SPLINE:
        for i in range(len(shot_list)):
            equi_imgs = []
            shot = shot_list[i]
            for j in range(shot.frame_count):
                equi_imgs.append(cv2.imread(img_paths[img_index + j]))
            nfov_imgs, path = shot.apply_to_imgs(equi_imgs, width, height)
            long_values.append(path[:, 0])
            lat_values.append(path[:,1])
            fov_values.append(path[:,2])
            counter.append(img_index)
            #long_values = np.append(long_values, path[:, 0])
            #lat_values = np.append(lat_values, path[:, 1])
            #fov_values = np.append(fov_values, path[:, 2])
            for j in range(shot.length):
                cost_heat.append(shot.cost_h[j])
                cost_fov.append(shot.cost_fov[j])
                cost_v.append(shot.cost_v[j])
                cost_l.append(shot.cost_l[j])
            for img in nfov_imgs:
                vid_writer.write(img)
                if img_index % 30 == 0:
                    cv2.imwrite('out_images/{}/{:03d}.jpg'.format(params_str, img_index), img)
                img_index += 1
            if i % 20 == 0:
                print(i) 

        counter.append(img_index)
        domain = range(len(long_values))
        plt.figure()
        for i in range(len(long_values)):
            plt.plot(range(counter[i], counter[i+1]), long_values[i], color='#1f77b4')
        #plt.plot(domain, long_values)
        plt.title('Longitude vs Frame')
        plt.xlabel('Frame')
        plt.ylim((-2 * np.pi, 2 * np.pi))
        plt.ylabel('Longitude')
        plt.savefig('plots/{}_long.png'.format(params_str))

        plt.figure()
        for i in range(len(long_values)):
            plt.plot(range(counter[i], counter[i+1]), lat_values[i], color='#ff7f0e')
        #plt.plot(domain, lat_values)
        plt.title('Latitude vs Frame')
        plt.xlabel('Frame')
        plt.ylim((-np.pi/2, np.pi/2))
        plt.ylabel('Latitude')
        plt.savefig('plots/{}_lat.png'.format(params_str))

        plt.figure()
        for i in range(len(long_values)):
            plt.plot(range(counter[i], counter[i+1]), fov_values[i], color='#2ca02c')
        #plt.plot(domain, fov_values)
        plt.title('FOV vs Frame')
        plt.xlabel('Frame')
        plt.ylim((np.pi/4, 3*np.pi/4))
        plt.ylabel('FOV')
        plt.savefig('plots/{}_fov.png'.format(params_str))
    else:
        final_path = np.zeros((1, 3))
        for i in range(len(shot_list)):
            #equi_imgs = []
            shot = shot_list[i]
            #for j in range(shot.frame_count):
            #    equi_imgs.append(cv2.imread(img_paths[img_index + j]))
            #nfov_imgs, path = shot.apply_to_imgs(equi_imgs, width, height)
            path = shot.gen_path(shot.frame_count)
            final_path = np.vstack((final_path, path))
            # counter.append(img_index)
            for j in range(shot.length):
                cost_heat.append(shot.cost_h[j])
                cost_fov.append(shot.cost_fov[j])
                cost_v.append(shot.cost_v[j])
                cost_l.append(shot.cost_l[j])
            # for img in nfov_imgs:
            #   vid_writer.write(img)
            #    if img_index % 30 == 0:
            #        cv2.imwrite('out_images/{}/{:03d}.jpg'.format(params_str, img_index), img)
            #    img_index += 1
        final_path = final_path[1:,:]
        for i in range(len(shot_list)-2):
            if np.linalg.norm(shot_list[i].end_pos - shot_list[i+1].start_pos) < np.pi / 6:
                final_path = refine_path(shot_list[i], shot_list[i+1], final_path)
        for i in range(len(final_path)):
            equi_img = cv2.imread(img_paths[i])
            img, _ = get_nfov_img(width, height, final_path[i, 2], final_path[i, 0], final_path[i, 1], equi_img)
            vid_writer.write(img)
            if i % 30 == 0:
                cv2.imwrite('out_images/{}/{:03d}.jpg'.format(params_str, i), img)

        domain = range(len(final_path))
        plt.figure()
        plt.scatter(domain, final_path[:,0], color='#1f77b4')
        #plt.plot(domain, long_values)
        plt.title('Longitude vs Frame')
        plt.xlabel('Frame')
        plt.ylim((-2 * np.pi, 2 * np.pi))
        plt.ylabel('Longitude')
        plt.savefig('plots/{}_long.png'.format(params_str))

        plt.figure()
        plt.scatter(domain, final_path[:,1], color='#ff7f0e')
        #plt.plot(domain, lat_values)
        plt.title('Latitude vs Frame')
        plt.xlabel('Frame')
        plt.ylim((-np.pi/2, np.pi/2))
        plt.ylabel('Latitude')
        plt.savefig('plots/{}_lat.png'.format(params_str))

        plt.figure()
        plt.scatter(domain, final_path[:,2], color='#2ca02c')
        #plt.plot(domain, fov_values)
        plt.title('FOV vs Frame')
        plt.xlabel('Frame')
        plt.ylim((np.pi/4, 3*np.pi/4))
        plt.ylabel('FOV')
        plt.savefig('plots/{}_fov.png'.format(params_str))
    vid_writer.release()
    t6 = time.time()
    print("Time to make video: {}".format(t6-t5))

    if optimization == 'merge' or optimization == 'pyramid':
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
    