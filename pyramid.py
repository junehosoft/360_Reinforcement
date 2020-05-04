import numpy as np
from config import *
from shot import *
from scipy.optimize import minimize

def get_bounds(pivot, length):
    delta_fov = 0.025
    min_fov = max(MIN_FOV, pivot[2] - (0.1 + delta_fov * length))
    max_fov = min(MAX_FOV, pivot[2] + (0.1 + delta_fov * length))
    delta_lat = np.pi / NUM_V / 4
    min_lat = max(-np.pi, pivot[1] - (np.pi / NUM_V + delta_lat * length))
    max_lat = min(np.pi, pivot[1] + (np.pi / NUM_V + delta_lat * length))
    delta_long = 2 * np.pi / (NUM_H) / 4
    min_long = max(- np.pi, pivot[0] - ( 2 * np.pi / (NUM_H) + delta_long * length))
    max_long = min(np.pi, pivot[0] + (2 * np.pi / (NUM_H) + delta_long * length))

    return np.array([min_long, max_long, min_lat, max_lat, min_fov, max_fov])

def solveByPyramid(heat_table, table_type):
    segments = []
    init_segment_length = INIT_SHOT_LENGTH
    unit_long = np.pi / (NUM_H)
    unit_lat = np.pi / NUM_V / 2
    unit_fov = 0.05
    cost_heat_by_iter = []
    cost_trans_by_iter = []
    opt_cost = 0
    opt_cost_heat = 0
    opt_cost_trans = 0
    current_shot = Shot(0, 1, heat_table, table_type)
    cache = []
    for f in range(1, heat_table.shape[0]):
        print("on frame ", f)
        s = np.append(current_shot.start_pos, current_shot.start_fov)
        e = np.append(current_shot.end_pos, current_shot.end_fov)
        # consider this new frame as extension of velocity of current shot
        next_pos, next_fov = current_shot.get_next()
        next_move = np.append(next_pos, next_fov)
        new_shot_a = Shot(current_shot.start, f-current_shot.start + 1, heat_table, table_type, PYRAMID, s, next_move)
        new_cost_a = new_shot_a.cost
        # locally optimize this segment 
        new_shot_b = Shot(current_shot.start, f-current_shot.start, heat_table, table_type)
        new_cost_b = new_shot_b.cost
        new_trans_a = 0
        new_trans_b = 0
        if len(segments) > 0:
            new_trans_a, _ = costCut(segments[-1], new_shot_a)
            new_trans_b, _ = costCut(segments[-1], new_shot_b)

        if new_trans_a + new_cost_a < new_trans_b + new_cost_b:
            min_shot = new_shot_a
            min_cost = new_cost_a
            min_trans = new_trans_a
        else:
            min_shot = new_shot_b
            min_cost = new_cost_b
            min_trans = new_trans_b
        

        if current_shot.length >= 5:
            anchor = Shot(f, 1, heat_table, table_type)
            min_shot_c = anchor
            min_index = -1
            min_cost_c = 10000
            pivot = np.append(anchor.start_pos, anchor.start_fov)
            for i in range(current_shot.length - 3):
                bounds = get_bounds(pivot, i+1)
                lb = bounds[::2]
                ub = bounds[1::2]
                def obj_func(x):
                    this_shot = Shot(f - (i+1), i+2, heat_table, table_type, PYRAMID, x, pivot)
                    prev_shot = cache[int(-1 - i)]
                    prev_trans_cost = 0
                    if len(segments) > 0:
                        prev_trans_cost, _ = costCut(segments[-1], prev_shot)
                    this_trans_cost, _ = costCut(prev_shot, this_shot)
                    total_cost = prev_trans_cost + prev_shot.cost + this_shot.cost
                    return total_cost
                for j in range(3):
                    x0 = pivot
                    #print(x0, lb, ub)
                    result = minimize(obj_func, x0=x0, bounds = np.vstack((lb, ub)).T)
                    start = result.x
                    '''for z in np.arange(bounds[4], bounds[5], unit_fov):
                        for lo in np.arange(bounds[0], bounds[1], unit_long):
                            for lat in np.arange(bounds[2], bounds[3], unit_lat):
                                start = [lo, lat, z]
                                this_shot = Shot(f - (i+1), i+2, heat_table, table_type, PYRAMID, start, pivot)
                                prev_shot = cache[int(-1 - i)]
                                prev_trans_cost = 0
                                if len(segments) > 0:
                                    prev_trans_cost, _ = costCut(segments[-1], prev_shot)
                                this_trans_cost, _ = costCut(prev_shot, this_shot)
                                total_cost = prev_trans_cost + prev_shot.cost + this_shot.cost '''
                    this_shot = Shot(f - (i+1), i+2, heat_table, table_type, PYRAMID, start, pivot)
                    prev_shot = cache[int(-1 - i)]
                    prev_trans_cost = 0
                    if len(segments) > 0:
                        prev_trans_cost, _ = costCut(segments[-1], prev_shot)
                    this_trans_cost, _ = costCut(prev_shot, this_shot)
                    total_cost = prev_trans_cost + prev_shot.cost + this_shot.cost
                    if total_cost < min_cost_c:
                        min_cost_c = total_cost
                        min_shot_c = this_shot
                        min_index = i
        
            if min_cost_c < min_cost:
                prev_shot = cache[int(-1-min_index)]
                opt_cost_heat += prev_shot.cost
                if len(segments) > 0:
                    opt_cost_trans += costCut(segments[-1], prev_shot)[0]
                cost_heat_by_iter.append(opt_cost_heat)
                cost_trans_by_iter.append(opt_cost_trans)
                segments.append(prev_shot)
                print("appending shot starting at {}".format(prev_shot.start))
                current_shot = min_shot_c
                # refill the cache
                cache = []
                for i in range(3, current_shot.length):
                    temp_shot = Shot(current_shot.start, i, heat_table, table_type)
                    cache.append(temp_shot)
                continue

        cache.append(current_shot)
        current_shot = min_shot

    opt_cost_heat += current_shot.cost
    if len(segments) > 0:
        opt_cost_trans += costCut(segments[-1], current_shot)[0]
    cost_heat_by_iter.append(opt_cost_heat)
    cost_trans_by_iter.append(opt_cost_trans)
    segments.append(current_shot)
    print(opt_cost_heat, opt_cost_trans
    return segments, np.array(cost_heat_by_iter), np.array(cost_trans_by_iter)
        

        
        
