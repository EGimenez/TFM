from utils.test_dir import save_make_dir
import glob
from tqdm import tqdm
import os
import utils.glow_api as glow_api

source_dir = '../data/celeba_wild/data_256_fast/'
target_dir = '../data/celeba_wild/data_encode_decode_test/'
save_make_dir(target_dir)


def encode_decode():
    cwd = os.getcwd()
    os.chdir(source_dir)
    aux = list(glob.glob('*'))
    os.chdir(cwd)

    i = 0
    for img_file in tqdm(aux):
        eps = glow_api.load_encode(source_dir, img_file)
        glow_api.save_decode(eps, target_dir, img_file)
        i += 1
        if i > 10:
            break


if __name__ == '__main__':
    print('converting')
    encode_decode()
    print('done')