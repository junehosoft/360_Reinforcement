import numpy as np
import matplotlib.pyplot as plt
from nfov_util import get_nfov_img
from load import load_ground_truths, write_npy, read_npy, display_gts

def get_params(min_fov, max_fov, num_fov, num_h, num_v):
    fov_space = np.linspace(min_fov, max_fov, num_fov)
    h_space = np.linspace(-np.pi, np.pi, num_h + 1)[:-1]
    v_space = np.linspace(-np.pi/2, np.pi / 2, num_v)
    # arbitrary order because numpy is weird
    hs, fovs, vs = np.meshgrid(h_space, fov_space, v_space)
    return np.stack((fovs, hs, vs), axis=0)

# params is a 4d volume: has fov, cam_h, cam_v coordinates
def get_masks(mask_shape, params):
    print("getting masks")
    temp_img = np.zeros((mask_shape[0], mask_shape[1], 3))
    masks = np.zeros((params.shape[1], params.shape[2], params.shape[3], mask_shape[0], mask_shape[1]))
    h = int(mask_shape[0] / 2)
    w = int(mask_shape[1] / 2)
    max_fov = np.max(params[0,:,0,0])
    for i in range(params.shape[1]):
        print(i)
        for j in range(params.shape[2]):
            for k in range(params.shape[3]):
                _, par_set = get_nfov_img(w, h, params[0, i, j, k], params[1, i, j, k], params[2, i, j, k], temp_img)
                masks[i, j, k] = par_set[-1] * params[0, i, j, k] #/ max_fov

    return masks

def precompute(data, fname, shot_length, params, write_masks = False):
    masks = get_masks(data[0].shape, params)
    mask_name = "{}_mask_{}_{}_{}.npy".format(fname, params.shape[1], params.shape[2], params.shape[3])
    if write_masks:
        np.save(mask_name, masks)
    print("done getting masks")
    total_len = int(data.shape[0] / shot_length)
    heat_table = np.zeros((total_len, params.shape[1], params.shape[2], params.shape[3], 4))
    for s in range(total_len):
        print(s)
        shot_sum = np.zeros(data[s*shot_length].shape)
        for i in range(s * shot_length, s * shot_length + shot_length):
            shot_sum += data[i]
        for i in range(params.shape[1]):
            for j in range(params.shape[2]):
                for k in range(params.shape[3]):
                    heat = np.sum(masks[i, j, k] * shot_sum) / shot_length / np.sum(masks[i, j, k])
                    heat_table[s, i, j, k] = np.array([heat, params[0, i, j, k], params[1, i, j, k], params[2, i, j, k]])
    
    return heat_table
        

if __name__ == '__main__':
    params = get_params(np.pi / 4, np.pi * 3 / 4, 5, 5, 5)
    fname = 220
    data = read_npy(fname)
    shot_length = 5
    heat_table = precompute(data, fname, shot_length, params, write_masks=True)
    np.save("{}_table_{}_{}_{}.npy".format(fname, params.shape[1], params.shape[2], params.shape[3]), heat_table)
    
