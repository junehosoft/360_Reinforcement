import numpy as np
from frame import FrameGT
from nfov_util import get_nfov_img
from table_util import find_max_heat, interpolate
from config import *
from scipy.optimize import minimize

class Shot():
    def __init__(self, start, length, heat_table, table_type, opt=MERGE, s=[-1,-1,-1], e=[-1,-1,-1]):
        self.start = start  # starting frame index, inclusive
        self.end = start + length  # ending frame index, exclusive
        self.length = length
        self.frame_count = SAMPLE_RATE * length
        self.tt = table_type

        #max_params = find_max_heat(heat_table, start)
        if opt == MERGE:
            cost, cost_h, cost_fov, cost_v, cost_l, x = self.optimizeSegment(heat_table)
            self.start_pos = x[0:2]
            self.start_fov = x[4]
            self.end_pos = x[2:4]
            self.end_fov = x[5]
        else:
            self.start_pos = s[0:2]
            self.start_fov = s[2]
            self.end_pos = e[0:2]
            self.end_fov = e[2]
            cost, cost_h, cost_fov, cost_v, cost_l = costShot(heat_table, self.tt, self.start, self.start_pos, self.start_fov, self.end, self.end_pos, self.end_fov)
        
        self.shotType = STATIC
        self.cost = cost
        self.cost_h = cost_h
        self.cost_fov = cost_fov
        self.cost_v = cost_v
        self.cost_l = cost_l

    def optimizeSegment(self, heat_table):
        lb = np.array([-np.pi, -np.pi/2,  -np.pi, -np.pi/2, MIN_FOV, MIN_FOV])
        ub = np.array([ np.pi, np.pi/2, np.pi, np.pi/2, MAX_FOV, MAX_FOV])
        def obj_func(x):
            return costShot(heat_table, self.tt, self.start, x[0:2], x[4], self.end, x[2:4], x[5])[0]

        cost = 10000000
        final_x = np.zeros(6)
        for i in range(4):
            x0 = np.array([0, 0, 0, 0, np.pi/2, np.pi/2])

            result = minimize(obj_func, x0=x0, bounds = np.vstack((lb, ub)).T)

            x = result.x
            new_cost, new_cost_h, new_cost_fov, new_cost_v, new_cost_l = costShot(heat_table, self.tt, self.start, x[0:2], x[4], self.end, x[2:4], x[5])
            if new_cost < cost:
                cost = new_cost
                cost_h = new_cost_h
                cost_fov = new_cost_fov
                cost_v = new_cost_v
                cost_l = new_cost_l
                final_x = x

        return cost, cost_h, cost_fov, cost_v, cost_l, final_x

    #def apply_to_img(self, img, width, height):
        #nfov_img, par_set = get_nfov_img(width, height, self.anchor[0], self.anchor[1], self.anchor[2], img)

        #return nfov_img, par_set
    def gen_path(self, num):
        delta = self.end_pos - self.start_pos
        if delta[0] > 0:
            delta2 = delta[0] - 2 * np.pi
        else:
            delta2 = delta[0] + 2 * np.pi
        if abs(delta2) < abs(delta[0]):
            delta[0] = delta2
        delta_pos = delta / (num-1)
        delta_fov = (self.end_fov - self.start_fov) / (num - 1)
        path = []
        pos = self.start_pos
        fov = self.start_fov
        for i in range(num):
            path.append([pos[0], pos[1], fov])
            pos += delta_pos
            fov += delta_fov
        path = np.array(path)
        #print(path.shape)
        return path

    def apply_to_imgs(self, imgs, width, height):
        assert self.frame_count == len(imgs)
        #nfov_img, par_set = get_nfov_img(width, height, self.anchor[0], self.anchor[1], self.anchor[2], imgs[0])

        ans = []
        delta = self.end_pos - self.start_pos
        if delta[0] > 0:
            delta2 = delta[0] - 2 * np.pi
        else:
            delta2 = delta[0] + 2 * np.pi
        if abs(delta2) < abs(delta[0]):
            delta[0] = delta2
        delta_pos = delta / (len(imgs) - 1)
        delta_fov = (self.end_fov - self.start_fov) / (len(imgs) - 1)
        path = []
        pos = self.start_pos
        fov = self.start_fov
        for i in range(len(imgs)):
            new_img, _ = get_nfov_img(width, height, fov, pos[0], pos[1], imgs[i])
            path.append([pos[0], pos[1], fov])
            pos += delta_pos
            fov += delta_fov
            ans.append(new_img)
        path = np.array(path)
        #print(path.shape)
        return ans, path

    def get_next(self):
        if self.length > 1:
            delta_pos = (self.end_pos - self.start_pos) / (self.length - 1)
            delta_fov = (self.end_fov - self.start_fov) / (self.length - 1)
        else:
            delta_pos = 0
            delta_fov = 0
        return self.end_pos + delta_pos, self.end_fov + delta_fov

def costShot(heat_table, tt, start, start_pos, start_fov, end, end_pos, end_fov):
    num = end - start
    pos = np.zeros((num, 2))
    pos[0] = start_pos
    pos[-1] = end_pos
    fov = np.linspace(start_fov, end_fov, num)

    #fill latitude
    for i in range(1, num-1):
        pos[i, 1] = (pos[0, 1] * (num - i - 1) + pos[-1, 1] * i) / (num - 1)
    #fill longitude
    delta1 = pos[-1, 0] - pos[0, 0]
    if delta1 < 0:
        delta2 = delta1 + 2 * np.pi
    else:
        delta2 = delta1 - 2 * np.pi
    pos1 = pos
    pos2 = pos
    for i in range(1, num-1):
        pos1[i, 0] = FixLon(pos[0, 0] + delta1 * i / (num-1))
        pos2[i, 0] = FixLon(pos[0, 0] + delta2 * i / (num-1))
    
    cost1, cost1_h, cost1_fov, cost1_v, cost1_l = costLine(heat_table, tt, start, pos1, fov)
    cost2, cost2_h, cost2_fov, cost2_v, cost2_l = costLine(heat_table, tt, start, pos2, fov)
    
    if cost1 < cost2:
        pos = pos1
        cost = cost1
        cost_h = cost1_h
        cost_fov = cost1_fov
        cost_v = cost1_v
        cost_l = cost1_l
    else:
        pos = pos2
        cost = cost2
        cost_h = cost2_h
        cost_fov = cost2_fov
        cost_v = cost2_v
        cost_l = cost2_l
    
    return cost, cost_h, cost_fov, cost_v, cost_l

def FixLon(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi

def costLine(heat_table, tt, start, pos_path, fov):
    #print(pos_path)
    cost_h = []
    cost = 0
    num = pos_path.shape[0]
    for i in range(num):
        heat = - 10 * interpolate(heat_table, start + i, fov[i], pos_path[i, 0], pos_path[i, 1])
        cost = cost + heat
        cost_h.append(heat)

    cost_h = np.array(cost_h)

    cost_v = []
    if pos_path.shape[0] > 1:
        dist = np.linalg.norm(pos_path[1] - pos_path[0])
    else:
        dist = 0
    if dist > np.pi/12:
        for i in range(num):
            cost_v.append(10 * dist**2)
        cost += np.mean(cost_v)
    else:
        for i in range(num):
            cost_v.append(0)

    cost_l = 0
    if num < 3:
        cost_l = 1000/num
    elif num > 10:
        cost_l = num/10
    cost += cost_l
    cost_l = np.repeat(cost_l/num, num)

    if tt == REGULAR:
        cost_fov = 0.1 * (fov - 1.7) * (fov - 1.7)
        cost += np.sum(cost_fov)  
    else:
        cost_fov = np.zeros(len(fov))

    return cost, cost_h, cost_fov, cost_v, cost_l

def costCut(last_shot, this_shot):
    # return 0, 0
    cut_const = 10
    eps = np.pi / 6
    f_cut = 0
    cut_type = -1
    vec_trans = this_shot.start_pos - last_shot.end_pos
    vec_last = last_shot.end_pos - last_shot.start_pos
    vec_this = this_shot.end_pos - this_shot.start_pos
    flag = 0
    dist_last = np.linalg.norm(vec_last)
    dist_this = np.linalg.norm(vec_this)
    '''if dist_last > 0.01:
        vec_last = vec_last/dist_last
        flag += 1
    if np.linalg.norm(vec_this) > 0.01:
        vec_this = vec_this/np.linalg.norm(vec_this)
        flag += 1
    if flag == 2:
        f_cut = 100000
        cut_type = -1'''
    #else:
    londist = this_shot.start_pos[0] - last_shot.end_pos[0]
    latdist = this_shot.start_pos[1] - last_shot.end_pos[1]
    a = np.sqrt(londist**2 + latdist**2)
    if a > eps:
        f_cut = 1
        cut_type = 0
    elif abs(last_shot.end_fov - this_shot.start_fov) > 0.3:
        f_cut = 1000 * (last_shot.end_fov - this_shot.start_fov)**2
        cut_type = -1
    else:
        cut_angle = (- np.dot(vec_trans, vec_last) - np.dot(vec_trans, vec_this)) / 2
        if cut_angle > 0:
            f_cut = a/eps * cut_angle**2
        else:
            f_cut = 0
    return f_cut, cut_type