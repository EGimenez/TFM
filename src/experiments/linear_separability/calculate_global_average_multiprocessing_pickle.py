# https://pymotw.com/3/multiprocessing/communication.html
# This script is produced to see if the average of all points in the latent space is zero

from utils.multiprocessing_eg.MultiProcessingFramework import Processor, MultiProcessManager, chunks
from utils.test_dir import save_make_dir

import numpy as np
from numpy import dot
from numpy.linalg import norm
from glob import glob
import platform
import config
from time import time
import psutil
import pickle


np_path = config.data_path / 'pickle'
#np_path = config.data_path / 'np'
re_path = config.results_path / 'linear_analysis_multiprocessing_test'
save_make_dir(re_path)

class AvProcessor(Processor):

    def __init__(self):
        Processor.__init__(self)
        self.my_value = 0
        self.my_count = 0

    def process_task(self, task):
        print('Process {} on {}'.format(self.name, psutil.Process().cpu_num()))
        t_0 = time()
        loading = 0
        copying = 0
        summing = 0
        # print('receiving list: {}'.format(task['img']))
        #self.lock.acquire()
        for np_name in task['img']:
            t0 = time()
            #an_np = np.load(np_name)
            with open(np_name, 'rb') as f:
                an_np = pickle.load(f)
            t1 = time()
            loading += t1 - t0
            t0 = time()
            an_np = an_np.copy()
            t1 = time()
            copying += t1 - t0
            t0 = time()
            try:
                self.my_value += an_np
                self.my_count += 1
            except:
                self.my_value = an_np
                self.my_count += 1
            t1 = time()
            summing += t1 - t0
        #self.lock.release()
        t_1 = time()
        print('Processor: {}, Total_Files: {}, Time: {}'.format(self.name, len(task['img']), t_1-t_0))
        print('Processor: {}, Loading: {}, Coppying: {}, Summing: {}'.format(self.name, loading, copying, summing))
        return None

    def process_global_task(self):
        result = {}
        result['value'] = self.my_value
        result['count'] = self.my_count

        print('The average local')
        try:
            the_norm = np.linalg.norm(self.my_value/self.my_count)
        except:
            the_norm = -1
        print(the_norm)

        return result


if __name__ == '__main__':
    mpm = MultiProcessManager()

    # Tasks
    the_files = list(glob((np_path / '*.pkl').as_posix()))
    #the_files = list(glob((np_path / '*.npy').as_posix()))
    the_files_num = len(the_files)

    gp_file_names_iterator = chunks(the_files, 1000)

    # Start consumers
    num_consumers = 4 # mpm.get_max_processors_num()
    print('Creating %d consumers' % num_consumers)
    for w in range(num_consumers):
        mpm.add_processor(AvProcessor)

    # Enqueue jobs
    num_jobs = 0
    for the_nps in gp_file_names_iterator:
        # print('sending list: {}'.format(the_nps))
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

    with open(re_path / 'the_average_multi.txt', 'w') as f:
        f.write(str(the_norm))

    # The images
#    from utils.glow_api import save_decode
#    save_decode(the_average, re_path, 'the_average_multi.jpg')

    print('hola')