# This script is produced to see if the average of all points in the latent space is zero
from glob import glob
import numpy as np
from tqdm import tqdm
import platform
import config
from utils.test_dir import save_make_dir
import os
from utils.FileNPAbstraction import fnpa

os.nice(0)


np_path = config.data_path / 'np_test'
re_path = config.results_path / 'linear_analysis_test_no_block'
save_make_dir(re_path)

file_num = 0
the_average = None

cwd = os.getcwd()
os.chdir(np_path)
file_list = os.listdir()
file_list.sort()
os.chdir(cwd)

for file in tqdm(file_list):
    # an_np = np.load(np_path / file)
    an_np = fnpa.get_np(file)
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

with open(re_path / 'the_average.txt', 'w') as f:
    f.write(str(the_norm))


# The images
#from utils.glow_api import save_decode
#save_decode(the_average, re_path, 'the_average.jpg')

print('hola')