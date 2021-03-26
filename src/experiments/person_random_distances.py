import utils.glow_api as glow_api
from utils.test_dir import save_make_dir
from utils.get_celeba_info import get_celeba_index
from numpy import linalg as LA
import pickle

source_dir = '../data/celeba/img/'
result_dir = '../data/celeba/results/random_distances/'
save_make_dir(result_dir)

celeb_index = get_celeba_index()


# We generate distance distribution condition to a person
celeb_index_id = celeb_index.groupby('person_id').count()
celeb_index_id = celeb_index_id[['img_id']]
celeb_index_id = celeb_index_id[celeb_index_id['img_id'] >= 15]

person_distances = list()
for n in range(1000):
    print(n)
    person_id = celeb_index_id.sample().index.values[0]
    file_name_1, file_name_2 = celeb_index[celeb_index['person_id'] == person_id].sample(n=2)['img_id'].values

    eps_1 = glow_api.load_encode(source_dir, file_name_1)
    eps_2 = glow_api.load_encode(source_dir, file_name_2)

    person_distances.append(LA.norm(eps_1 - eps_2))

with open(result_dir+'person_distances.pickle', 'wb') as handle:
    pickle.dump(person_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('done')
