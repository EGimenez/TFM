# https://pymotw.com/3/multiprocessing/communication.html


import multiprocessing
import argparse
import platform
import json
from utils.GlowProvider import GlowProvider
from pathlib import Path
from numpy import dot
from numpy.linalg import norm

class AvConsumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, index_path, np_path):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.gp = GlowProvider(index=index_path, np_path=np_path)

    def run(self):
        proc_name = self.name
        the_averages = {}
        features = self.gp.get_features()

        while True:
            # {'img': imgs, 'num_job': num_jobs, 'total_num_job': gp_file_total}
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.result_queue.put(the_averages)
                self.task_queue.task_done()
                break

            print('{}: {} / {}'.format(proc_name, next_task['num_job'], next_task['total_num_job']))
            for np_name in next_task['img']:
                np, feature_values = self.gp.get_np_with_features(np_name)

                for f in features:
                    try:
                        the_averages[(f, feature_values[f])]['np'] += np.copy()
                        the_averages[(f, feature_values[f])]['count'] += 1
                    except:
                        the_averages[(f, feature_values[f])] = {}
                        the_averages[(f, feature_values[f])]['np'] = np.copy()
                        the_averages[(f, feature_values[f])]['count'] = 1

            self.task_queue.task_done()
        return


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    # Get Process Variables
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--index', required=False, default=Path() / '..' / 'data' / 'celeba' / 'index' / 'list_attr_celeba_clean.txt',  help='path index file')
    if platform.system() == 'Windows':
        # ap.add_argument('-n', '--np_files', required=False, default=Path('E:/NOT_BACKUP/TFM') / 'data' / 'celeba' / 'np', help='path directory containing the np files')
        ap.add_argument('-n', '--np_files', required=False, default=Path() / '..' / 'data' / 'celeba' / 'np', help='path directory containing the np files')
    else:
        ap.add_argument('-n', '--np_files', required=False, default=Path() / '..' / 'data' / 'celeba' / 'np', help='path directory containing the np files')
    ap.add_argument('-r', '--result', required=False, default=Path() / '..' / 'data' / 'celeba' / 'results' / 'linear_analysis',
                    help='path where result file will be set')
    ap.add_argument('-s', '--sampling', required=False, type=float, help='If sampling, indicate the percentage')
    ap.add_argument('-m', '--max_sampling', required=False, type=float, default=20, help='If sampling, indicate the percentage')
    ap.add_argument('-c', '--chunk_size', required=False, type=int, default=100, help='Chunk size')
    args = vars(ap.parse_args())

    # We load the GlowProvider
    gp = GlowProvider(index=args['index'], np_path=args['np_files'])

    gp_file_names_iterator = chunks(gp.get_image_names(), args['chunk_size'])
    gp_file_total = gp.get_np_num() // args['chunk_size']

    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start consumers
    # num_consumers = multiprocessing.cpu_count()
    num_consumers = 4
    print('Creating %d consumers' % num_consumers)
    consumers = [AvConsumer(tasks, results, index_path=args['index'], np_path=args['np_files']) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    # Enqueue jobs
    num_jobs = 0
    for imgs in gp_file_names_iterator:
        num_jobs += 1
        tasks.put({'img': imgs, 'num_job': num_jobs, 'total_num_job': gp_file_total})

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # We sum everything up
    the_results = {}
    while num_consumers:
        result = results.get()
        num_consumers -= 1

        for key in result:
            if key == ('Male', 1):
                print('hola')
            try:
                the_results[key]['np'] += result[key]['np'].copy()
                the_results[key]['count'] += result[key]['count']
            except:
                the_results[key] = {}
                the_results[key]['np'] = result[key]['np'].copy()
                the_results[key]['count'] = result[key]['count']
            try:
                print(the_results[('Male', 1)]['np'][0, 1])
            except:
                print('nop')

    # We average them up
    for key in the_results:
        if key == ('Male', 1):
            print('hola')
        the_results[key]['np'] /= the_results[key]['count']
        print(the_results[('Male', 1)]['np'][0, 1])

    # The norms
    the_norms = {}
    for key in the_results:
        the_norms[key[0]+'_'+str(key[1])] = norm(the_results[key]['np'])

    with open(args['result'] / 'norms.json', 'w') as o:
        json.dump(str(the_norms), o)

    # Cosine Similarity
    the_cosines = {}
    features = gp.get_features()
    for f in features:
        f_0 = the_results[(f, 0)]['np']
        f_1 = the_results[(f, 1)]['np']

        the_cosines[f] = dot(f_0, f_1.transpose())/(norm(f_0)*norm(f_1))

    with open(args['result'] / 'cosines.json', 'w') as o:
        json.dump(str(the_cosines), o)

    # The images
    from utils.glow_api import save_decode
    for key in the_results:
        save_decode(the_results[key]['np'], args['result'], key[0] + '_' +str(key[1]) + '.jpg')

    print('hola')
