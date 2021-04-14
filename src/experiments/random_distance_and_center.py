import utils.glow_api as glow_api
from utils.test_dir import save_make_dir
from utils.get_celeba_info import get_celeba_index
from numpy import linalg as LA
import pickle
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('-a', action='store', type=str, default='global', help='attribute name')
my_parser.add_argument('-v', action='store', type=int, default=None, help='attribute value')
my_parser.add_argument('-n', action='store', type=int, default=1000, help='num of trials')
args = my_parser.parse_args()
args = vars(args)

my_attribute = args['a']
my_value = args['v']
my_num = args['n']

source_dir = '../data/celeba/img/'
result_dir = '../data/celeba/results/random_and_center/'
save_make_dir(result_dir)

celeb_index = get_celeba_index()

distances = list()
for n in range(my_num):
    print(n)
    if my_attribute != 'global':
        file_name_1, file_name_2 = celeb_index[celeb_index[my_attribute] == my_value].sample(n=2)['img_id'].values
    else:
        file_name_1, file_name_2 = celeb_index.sample(n=2)['img_id'].values

    eps_1 = glow_api.load_encode(source_dir, file_name_1)
    eps_2 = glow_api.load_encode(source_dir, file_name_2)

    distances.append(LA.norm(eps_1 - eps_2))

    try:
        component += eps_1
        component += eps_2
    except:
        component = eps_1
        component += eps_2

component = component/(my_num*2)

with open(result_dir+'distances_a_'+my_attribute+'_v_'+str(my_value)+'_n_'+str(my_num)+'.pickle', 'wb') as handle:
    pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(result_dir+'component_a_'+my_attribute+'_v_'+str(my_value)+'_n_'+str(my_num)+'.pickle', 'wb') as handle:
    pickle.dump(component, handle, protocol=pickle.HIGHEST_PROTOCOL)

glow_api.save_decode(component, result_dir, 'avg_img_a_' + my_attribute + '_v_' + str(my_value) + '_n_' + str(my_num) + '.jpg')

print('done')
