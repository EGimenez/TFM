import pandas as pd

index_dir = '../data/celeba/index/'


def get_celeba_index():
    df_att = pd.read_csv(index_dir + 'list_attr_celeba_clean.txt', sep=' ')
    df_id = pd.read_csv(index_dir + 'identity_CelebA.txt', sep=' ', header=None, names=['img_id', 'person_id'])
    result = pd.merge(df_att, df_id, how="left", on=['img_id', 'img_id'])
    result.pop('Unnamed: 41')

    return result
