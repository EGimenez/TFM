from utils.test_dir import save_make_dir
import glob
from tqdm import tqdm
import os
import pickle
import config
import numpy as np

source_dir = config.data_path / 'np'
target_dir = config.data_path / 'packed_np'
save_make_dir(target_dir)


def pack_np():
    cwd = os.getcwd()
    os.chdir(source_dir)
    file_list = os.listdir()
    file_list.sort()
    os.chdir(cwd)

    block_size = config.block_size
    block_data = {}
    block_num = block_size
    file_block_file = ('%06d' % block_num) + '.pkl'

    for img_file in tqdm(file_list):
        if int(img_file[:-4]) > block_num:
            with open(target_dir / file_block_file, 'wb') as f:
                pickle.dump(block_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            block_data = {}
            block_num += block_size
            file_block_file = ('%06d' % block_num) + '.pkl'

        with open(source_dir / (img_file[:-4]+'.npy'), 'rb') as f:
            aux = np.load(f)
        block_data[img_file[:-4]] = aux

    if len(block_data) > 0:
        with open(target_dir / file_block_file, 'wb') as f:
            pickle.dump(block_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('converting')
    pack_np()
    print('done')
