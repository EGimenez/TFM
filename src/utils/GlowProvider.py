import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils.get_celeba_info import get_celeba_index
from typing import Tuple


class GlowProvider(object):

    def __init__(self, index, img_path: Path = None, np_path: Path = None):
        # self.index: pd.DataFrame = pd.read_csv(str(index), sep=' ')
        aux = get_celeba_index(index)
        aux.set_index('img_id', inplace=True, drop=False)
        self.index = aux
        self.img_path = img_path
        self.np_path = np_path

    def filter_index(self, features: dict) -> pd.DataFrame:
        df = self.index.copy()
        for key, val in features.items():
            df = df[df[key] == val]
        return df

    def get_np(self, features: dict, sample=0.1, sample_max=np.Inf) -> np.ndarray:
        nps = list()
        nps_generator = self.get_np_yield(features=features, sample=sample, sample_max=sample_max)

        for n in nps_generator:
            nps.append(n)

        return np.concatenate(nps)

    def get_np_yield(self, features: dict, sample=0.1, sample_max=np.Inf) -> np.ndarray:
        df = self.filter_index(features)
        max_elements = min(sample * len(df) // 1, sample_max)
        cur_elements = 0
        indexes = df.index.values
        new_indexes = np.random.permutation(indexes)

        for i in new_indexes:
            file_name = df.loc[i, 'img_id']
            try:
                result = np.load(str((self.np_path / (file_name[:-4] + '.npy'))))
                cur_elements += 1
                if cur_elements % 1000 == 0:
                    print(cur_elements)
                yield result
                if cur_elements >= max_elements:
                    break
            except:
                pass

    def get_features(self) -> list:
        result = self.index.columns.values.tolist()

        try:
            result.remove('img_id')
        except:
            pass
        try:
            result.remove('person_id')
        except:
            pass

        return result

    def get_np_with_features(self, np_name: str) -> Tuple[np.ndarray, dict]:
        indexes = self.index

        result = np.load(str((self.np_path / (np_name[:-4] + '.npy'))))
        return result.copy(), self.index.loc[np_name].to_dict()

    def get_image_names(self):
        image_names = []
        for file_name in self.index.index.tolist():
            if (self.np_path / (file_name[:-4] + '.npy')).exists():
                image_names.append(file_name)

        return image_names

    def get_np_num(self):
        list = os.listdir(self.np_path)
        return len(list)


if __name__ == '__main__':
    fp = GlowProvider(index=Path() / '..' / 'data' / 'celeba' / 'index' / 'list_attr_celeba_clean.txt',
                      np_path=Path('E:/NOT_BACKUP/TFM') / 'data' / 'celeba' / 'np')

    images = fp.get_np({'Blond_Hair': 1})

    print('hola')
