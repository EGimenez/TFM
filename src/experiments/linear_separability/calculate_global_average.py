# This script is produced to see if the average of all points in the latent space is zero
from glob import glob
import numpy as np
from tqdm import tqdm
import platform

if platform.system() == 'Windows':
    root_path = f'../data/celeba/'
else:
    root_path = f'../data/celeba_wild/'

np_path = root_path + 'np/'
re_path = root_path + 'results/linear_analysis/'

file_num = 0
the_average = None

for file in tqdm(glob(np_path + '*.npy')):
    an_np = np.load(file)
    an_np = an_np.copy()
    file_num += 1

    try:
        the_average += an_np
    except:
        the_average = an_np

the_average = the_average/file_num

print('The average')
the_norm = np.linalg.norm(the_average)
print(the_norm)

with open(re_path + 'the_average.txt', 'w') as f:
    f.write(str(the_norm))


# The images
from utils.glow_api import save_decode
save_decode(the_average, re_path, 'the_average.jpg')

print('hola')