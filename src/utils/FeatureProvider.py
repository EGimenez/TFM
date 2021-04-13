import pandas as pd
import numpy as np
from pathlib import Path


class FeatureProvider(object):

    def __init__(self, index, img_path: Path = None, np_path: Path = None):
        self.index = pd.read_csv(str(index), sep=' ')
        self.img_path = img_path
        self.np_path = np_path

    def filter_index(self, features: dict) -> pd.DataFrame:
        df = self.index.copy()
        for key, val in features.items():
            df = df[df[key] == val]
        return df

    def get_np(self, features: dict, sample=0.1, sample_max=np.Inf):
        nps = list()
        df = self.filter_index(features)
        max_elements = min(sample * len(df) // 1, sample_max)
        cur_elements = 0
        indexes = df.index.values
        new_indexes = np.random.permutation(indexes)

        for i in new_indexes:
            file_name = df.loc[i, 'img_id']
            try:
                nps.append(np.load(str((self.np_path / (file_name[:-4] + '.npy')))))
                cur_elements += 1
                if cur_elements % 1000 == 0:
                    print(cur_elements)

                if cur_elements >= max_elements:
                    break
            except:
                pass

        return np.concatenate(nps)

    def get_features(self) -> list:
        result = self.index.columns.values.tolist()
        result.remove('img_id')
        return result

if __name__ == '__main__':
    fp = FeatureProvider(index=Path() / '..' / 'data' / 'celeba' / 'index' / 'list_attr_celeba_clean.txt',
                         np_path=Path('E:/NOT_BACKUP/TFM') / 'data' / 'celeba' / 'np')

    images = fp.get_np({'Blond_Hair': 1})

    print('hola')
