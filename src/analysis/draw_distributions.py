from utils.test_dir import save_make_dir
import pickle
import sys
import matplotlib.pyplot as plt

#component_1 = sys.argv[1]

global_distances = 'global_distances'
person_distances = 'person_distances'

result_dir = '../data/celeba/results/random_distances/'
save_make_dir(result_dir)

with open(result_dir+global_distances+'.pickle', 'rb') as handle:
    global_distances = pickle.load(handle)

with open(result_dir+person_distances+'.pickle', 'rb') as handle:
    person_distances = pickle.load(handle)


plt.hist(global_distances, bins='auto')
plt.title('global_distances')
plt.show()

plt.hist(person_distances, bins='auto')
plt.title('person_distances')
plt.show()



print('done')
