import argparse
from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--index', required=False, default=Path() / '..' / 'data' / 'celeba_test' / 'index' / 'list_attr_celeba_clean.txt',  help='path index file')
    ap.add_argument('-n', '--np_files', required=False, default=Path() / '..' / 'data' / 'celeba_test' / 'np', help='path directory containing the np files')
    ap.add_argument('-s', '--sampling', required=False, type=float, help='If sampling, indicate the percentage')
    args = ap.parse_args()

    list_att = ['img_id', 'F1', 'F2', 'F3']

    list_att_df = pd.DataFrame(columns=list_att)
    list_att_df_v = pd.DataFrame(columns=[*list_att, 'V1', 'V2', 'V3'])

    points = np.random.normal(0, 1, size=(800, 3))
    center = points.mean(axis=0)
    points = points - center

    def v(val):
        return 1 if val > 0 else 0

    for p in range(points.shape[0]):
        list_att_df.loc[p] = ['%05d.jpg' % p, v(points[p, 0]), v(points[p, 1]), v(points[p, 2])]
        list_att_df_v.loc[p] = ['%05d.jpg' % p, v(points[p, 0]), v(points[p, 1]), v(points[p, 2]), (points[p, 0]), (points[p, 1]), (points[p, 2])]

        np.save(args.np_files / ('%05d' % p), points[p])

    list_att_df.to_csv(args.index, index=False, sep=' ')
    list_att_df_v.to_csv(args.index / '..' / 'puntitos.csv', index=False, sep=' ')

