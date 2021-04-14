import utils.glow_api as glow_api
from utils.test_dir import save_make_dir
import pickle
import sys

component_1 = sys.argv[1]
component_2 = sys.argv[2]

source_dir = '../data/celeba/img/'
result_dir = '../data/celeba/results/random_and_center/'
save_make_dir(result_dir)

with open(result_dir+component_1+'.pickle', 'rb') as handle:
    component_1_ = pickle.load(handle)

with open(result_dir+component_2+'.pickle', 'rb') as handle:
    component_2_ = pickle.load(handle)


glow_api.save_decode(component_1_ + component_2_, result_dir, 'add_' + component_1 + '_' + component_2 + '.jpg')

print('done')
