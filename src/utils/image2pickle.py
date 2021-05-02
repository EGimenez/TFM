import utils.start_tf
from utils.start_tf import *
import glow_api
from utils.test_dir import save_make_dir
import glob
from tqdm import tqdm
import os
import pickle
import config


source_dir = config.data_path / 'data_256_fast'
target_dir = config.data_path / 'pickle'
save_make_dir(target_dir)


def convert_img_2_pickle():
    cwd = os.getcwd()
    os.chdir(source_dir)
    aux = list(glob.glob('*'))
    os.chdir(cwd)

    for img_file in tqdm(aux):
        if not os.path.exists(target_dir / (img_file[:-4]+'.pkl')):
            eps = glow_api.load_encode(source_dir, img_file)
            with open(target_dir / (img_file[:-4]+'.pkl'), 'wb') as f:
                pickle.dump(eps, f)


if __name__ == '__main__':
    print('converting')
    convert_img_2_pickle()
    print('done')
