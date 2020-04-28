import numpy as np
import matplotlib.pyplot as plt
import glob

# given the folder number, reads the ground truths from the folder 
# and compiles into a single numpy array of size (# of frames, height, width)
def load_ground_truths(fname):
    folder_path = '360_Saliency_dataset_2018ECCV/{}/*.npy'.format(fname)
    paths = glob.glob(folder_path)
    frames = len(paths)
    shape = np.load(paths[1]).shape
    data = np.zeros((frames, shape[1], shape[2]))
    for i in range(frames):
        data[i] = np.load(paths[i])[0, :, :]
    return data

# write the given numpy array into an npy file with name
def write_npy(data, name):
    np.save('{}.npy'.format(name), data)

# reads the npy file
def read_npy(name):
    return np.load('{}.npy'.format(name))

def display_gts(data, num):
    plt.ion()
    count = min(num, data.shape[0])
    for i in range(count):
        plt.imshow(data[i])
        plt.pause(0.2)

    plt.ioff()

def load_image(fname):
    return
