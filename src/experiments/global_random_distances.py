import utils.glow_api as glow_api
from utils.test_dir import save_make_dir
from utils.get_celeba_info import get_celeba_index
from numpy import linalg as LA
import pickle

source_dir = '../data/celeba/img/'
result_dir = '../data/celeba/results/random_distances/'
save_make_dir(result_dir)

celeb_index = get_celeba_index()

# We generate global distance distribution
distances = list()
for n in range(1000):
    print(n)
    file_name_1, file_name_2 = celeb_index.sample(n=2)['img_id'].values

    eps_1 = glow_api.load_encode(source_dir, file_name_1)
    eps_2 = glow_api.load_encode(source_dir, file_name_2)

    distances.append(LA.norm(eps_1 - eps_2))

with open(result_dir+'global_distances.pickle', 'wb') as handle:
    pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('done')
