import config
import numpy as np
import pickle


class FileNPAbstraction(object):

    def __init__(self):
        self.pack_np_dir = config.data_path / 'packed_np'
        self.block_data = None
        self.block_num = 0

    def get_np(self, file_name):
        file_name_num = int(file_name[:-4])

        if file_name_num > self.block_num:
            block_num = '%06d' % int(config.block_size * np.ceil(file_name_num / config.block_size))
            self.block_num = int(block_num)
            with open(self.pack_np_dir / (block_num + '.pkl'), 'rb') as f:
                self.block_data = pickle.load(f)

        try:
            result = self.block_data[file_name[:-4]]
        except:
            result = None
        return result


fnpa = FileNPAbstraction()