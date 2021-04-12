import utils.start_tf
from utils.start_tf import *
import glow_api
from utils.test_dir import save_make_dir
import glob
from tqdm import tqdm
import os

source_dir = '../data/celeba/img/'
source_dir = '../data/test/'
target_dir = '../data/celeba/np/'
save_make_dir(target_dir)


def convert_img_2_np():
    cwd = os.getcwd()
    os.chdir(source_dir)
    aux = list(glob.glob('*'))
    os.chdir(cwd)

    for img_file in tqdm(aux):
        eps = glow_api.load_encode(source_dir, img_file)
        np.save(target_dir+img_file[:-4]+'.npy', eps)


if __name__ == '__main__':
    print('converting')
    convert_img_2_np()
    print('done')
