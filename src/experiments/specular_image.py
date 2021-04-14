import utils.glow_api as glow_api
from utils.test_dir import save_make_dir
from utils.get_celeba_info import get_celeba_index
from numpy import linalg as LA
import pickle
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-n', action='store', type=int, default=10, help='num of trials')
args = my_parser.parse_args()
args = vars(args)

my_num = args['n']

source_dir = '../data/celeba/img/'
result_dir = '../data/celeba/results/specular/'
save_make_dir(result_dir)

celeb_index = get_celeba_index()

distances = list()
for n in range(my_num):
    print(n)
    file_name = celeb_index.sample(n=1)['img_id'].values
    file_name = file_name[0]


    eps_1 = glow_api.load_encode(source_dir, file_name)

    glow_api.save_decode(eps_1, result_dir, file_name[0:-4] + '.jpg')
    glow_api.save_decode(-eps_1, result_dir, file_name[0:-4] + '_specular' + '.jpg')

print('done')
