import numpy as np
from config import *
from shot import *

def solveByMerge(heat_table, table_type):
    segments = []
    init_segment_length = INIT_SHOT_LENGTH
    num_segment = int(np.floor(heat_table.shape[0] / init_segment_length))
    print(heat_table.shape[0], num_segment)
    cost_heat_by_iter = []
    cost_trans_by_iter = []
    opt_cost = 0
    opt_cost_heat = 0
    opt_cost_trans = 0
    for i in range(num_segment-1):
        S = Shot(i * init_segment_length, init_segment_length, heat_table, table_type)
        segments.append(S)
        opt_cost_heat += S.cost
        if i > 0:
            cut_cost, cut_type = costCut(segments[i-1], S)
            # print(cut_cost)
            opt_cost_trans += cut_cost
    opt_cost = opt_cost_heat + opt_cost_trans
    cost_heat_by_iter.append(opt_cost_heat)
    cost_trans_by_iter.append(opt_cost_trans)
    print("starting total cost: {}".format(opt_cost))
    i_opt = 0
    opt_cost_h = 0
    opt_cost_l = 0
    opt_cost_v = 0
    opt_cost_zoom = 0
    opt_shot_type = -1
    counter = 0
    while i_opt != -1:
        print(counter)
        counter += 1
        i_opt = -1
        min_cost = 0
        min_cost_heat = 0
        min_cost_trans = 0
        opt_S = segments[0]
        for i in range(len(segments)-1):
            new_S = Shot(segments[i].start, segments[i].length + segments[i+1].length, heat_table, table_type)
            delta_cost_heat = new_S.cost - (segments[i].cost + segments[i+1].cost)
            #print(i)
            #print(delta_cost_heat)
            cut_cost, cut_type = costCut(segments[i], segments[i+1])
            delta_cost_trans = -cut_cost
            #print(cut_cost)
            # delta_cost = delta_cost - cut_cost
            if i > 0:
                cut_cost1, cut_type1 = costCut(segments[i-1], new_S)
                cut_cost2, cut_type2 = costCut(segments[i-1], segments[i])
                delta_cost_trans += cut_cost1 - cut_cost2
                #print(cut_cost1, cut_cost2)
            if i < len(segments) - 2:
                cut_cost1, cut_type1 = costCut(new_S, segments[i+2])
                cut_cost2, cut_type2 = costCut(segments[i+1], segments[i+2])
                delta_cost_trans += cut_cost1 - cut_cost2
                #print(cut_cost1, cut_cost2)
            # print(delta_cost_trans)
            delta_cost = delta_cost_heat + delta_cost_trans
            if delta_cost < min_cost:
                i_opt = i
                # opt_cost = new_S.cost
                min_cost = delta_cost
                min_cost_heat = delta_cost_heat
                min_cost_trans = delta_cost_trans
                opt_S = new_S
        print("{}th iteration chose: {}".format(counter, i_opt))
        if i_opt == -1:
            break
        del segments[i_opt + 1]

        segments[i_opt] = opt_S
        opt_cost += min_cost
        opt_cost_heat += min_cost_heat
        opt_cost_trans += min_cost_trans
        cost_heat_by_iter.append(opt_cost_heat)
        cost_trans_by_iter.append(opt_cost_trans)

    print("final total cost: {}".format(opt_cost))
    return segments, np.array(cost_heat_by_iter), np.array(cost_trans_by_iter)


