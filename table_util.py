import numpy as np
from config import *

def find_max_heat(heat_table, index):
    max_params = np.zeros(4)
    table = heat_table[index]

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            for k in range(table.shape[2]):
                if max_params[0] < table[i, j, k, 0]:
                    max_params = table[i, j, k]

    return max_params

def interpolate(heat_table, index, fov, lon, lat):
    table = heat_table[index]

    lon_index = (lon + np.pi) / (2 * np.pi) * (NUM_H - 1)
    if lon_index < 0:
        lon_index = 0
    if lon_index > NUM_H - 1:
        lon_index = NUM_H - 1

    lat_index = (lat + 0.5 * np.pi) / (np.pi) * (NUM_V - 1)
    if lat_index < 0:
        lat_index = 0
    if lat_index > NUM_V - 1:
        lat_index = NUM_V - 1

    lon_floor = int(np.floor(lon_index))
    lon_ceil = int(np.ceil(lon_index))

    lat_floor = int(np.floor(lat_index))
    lat_ceil = int(np.ceil(lat_index))

    fov_range = MAX_FOV - MIN_FOV 
    fov_index = (fov - MIN_FOV) / (fov_range) * (NUM_FOV - 1)
    if fov_index < 0:
        fov_index = 0
    if fov_index > NUM_FOV - 1:
        fov_index = NUM_FOV - 1

    fov_floor = int(np.floor(fov_index))
    fov_ceil = int(np.ceil(fov_index))

    lon_weight = lon_ceil - lon_index
    lat_weight = lat_ceil - lat_index
    fov_weight = fov_ceil - fov_index

    heat = fov_weight * lon_weight * lat_weight * table[fov_floor, lon_floor, lat_floor, 0] 
    heat += fov_weight * (1 - lon_weight) * lat_weight * table[fov_floor, lon_ceil, lat_floor, 0] 
    heat += fov_weight * lon_weight * (1 - lat_weight) * table[fov_floor, lon_floor, lat_ceil, 0] 
    heat += fov_weight * (1- lon_weight) * (1 - lat_weight) * table[fov_floor, lon_ceil, lat_ceil, 0] 
    heat += (1 - fov_weight) * lon_weight * lat_weight * table[fov_ceil, lon_floor, lat_floor, 0] 
    heat += (1 - fov_weight) * (1 - lon_weight) * lat_weight * table[fov_ceil, lon_ceil, lat_floor, 0] 
    heat += (1 - fov_weight) * lon_weight * (1 - lat_weight) * table[fov_ceil, lon_floor, lat_ceil, 0] 
    heat += (1 - fov_weight) * (1- lon_weight) * (1 - lat_weight) * table[fov_ceil, lon_ceil, lat_ceil, 0] 

    return heat


        