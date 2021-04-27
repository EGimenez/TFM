import pandas as pd
import config

index_dir = config.index_path


def get_celeba_index(the_index=None):
    if the_index:
        df_att = pd.read_csv(the_index, sep=' ')
    else:
        df_att = pd.read_csv(index_dir / 'list_attr_celeba_clean.txt', sep=' ')
    df_id = pd.read_csv(index_dir / 'identity_CelebA.txt', sep=' ', header=None, names=['img_id', 'person_id'])
    try:
        result = pd.merge(df_att, df_id, how="left", on=['img_id', 'img_id'])
        result.pop('Unnamed: 41')
    except:
        result = df_att

    return result
