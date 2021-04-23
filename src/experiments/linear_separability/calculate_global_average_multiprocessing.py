# https://pymotw.com/3/multiprocessing/communication.html
# This script is produced to see if the average of all points in the latent space is zero

from utils.multiprocessing.MultiProcessingFramework import Processor, MultiProcessManager, chunks

import numpy as np
from numpy import dot
from numpy.linalg import norm
from glob import glob
import platform

if platform.system() == 'Windows':
    root_path = f'../data/celeba/'
else:
    root_path = f'../data/celeba_wild/'

np_path = root_path + 'np/'
re_path = root_path + 'results/linear_analysis/'

class AvProcessor(Processor):

    def __init__(self):
        Processor.__init__(self)
        self.my_value = None
        self.my_count = 0

    def process_task(self, task):
        print('here')
        for np_name in task['img']:
            an_np = np.load(np_name)
            an_np = an_np.copy()
            try:
                self.my_value += an_np
                self.my_count += 1
            except:
                self.my_value = an_np
                self.my_count += 1
        return None

    def process_global_task(self):
        result = {}
        result['value'] = self.my_value
        result['count'] = self.my_count

        print('The average local')
        the_norm = np.linalg.norm(self.my_value/self.my_count)
        print(the_norm)

        return result


if __name__ == '__main__':
    mpm = MultiProcessManager()

    # Tasks
    the_files = list(glob(np_path + '*.npy'))
    the_files_num = len(the_files)

    gp_file_names_iterator = chunks(the_files)

    # Start consumers
    num_consumers = 1 # mpm.get_max_processors_num()
    print('Creating %d consumers' % num_consumers)
    for w in range(num_consumers):
        mpm.add_processor(AvProcessor)

    # Enqueue jobs
    num_jobs = 0
    for the_nps in gp_file_names_iterator:
        mpm.send_task({'img': the_nps})

    # We sum everything up
    the_result = {}

    while True:
        result = mpm.next_result()

        if not result:
            break

        try:
            the_result['value'] += result['value']
            the_result['count'] += result['count']
        except:
            the_result['value'] = result['value']
            the_result['count'] = result['count']

    # We average them up
    the_average = the_result['value']/the_result['count']

    print('The average')
    the_norm = np.linalg.norm(the_average)
    print(the_norm)

    with open(re_path + 'the_average_multi.txt', 'w') as f:
        f.write(str(the_norm))

    # The images
    from utils.glow_api import save_decode
    save_decode(the_average, re_path, 'the_average_multi.jpg')

    print('hola')