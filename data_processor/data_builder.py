# coding=utf-8
import unicodedata

import pandas as pd

from parm import *


def build_dataset(unsupervised=False):
    if unsupervised:
        df_kg = pd.DataFrame()
        for filename in ['train', 'dev']:
            df = pd.read_json(os.path.join(PATH_DATA_RAW, filename + '.json'), lines=True)
            for col in df.columns.tolist():
                df = data_clean(df, col)

            df['label'] = df['label'].astype(int)
            mask = (df['label'] == 1)
            df_kg = df_kg.append(df[mask], ignore_index=True)

        df_kg = df_kg.drop_duplicates(['sentence1'])
        df_kg.to_csv(os.path.join(PATH_DATA_PRE, 'kg.csv'), index=False, encoding='utf-8')

        df_q = df_kg.rename(columns={'sentence1': COL_TXT})
        df_q[COL_TXT].to_csv(os.path.join(PATH_DATA_PRE, 'query.csv'), index=False, encoding='utf-8')

        df_c = df_kg.drop_duplicates(['sentence2'])
        df_c = df_c.rename(columns={'sentence2': COL_TXT})
        df_c[COL_TXT].to_csv(os.path.join(PATH_DATA_PRE, 'candidate.csv'), index=False, encoding='utf-8')

    else:
        for filename in ['train', 'dev', 'test']:
            df = pd.read_json(os.path.join(PATH_DATA_RAW, filename + '.json'), lines=True)
            for col in df.columns.tolist():
                df = data_clean(df, col)

            if 'label' not in df.columns.tolist():
                df['label'] = 0

            df.to_csv(os.path.join(PATH_DATA_PRE, filename + '.csv'), index=False, encoding='utf-8')


def data_clean(df, col):
    # 全角转半角
    df[col] = df[col].apply(lambda x: unicodedata.normalize('NFKC', str(x)))
    df[col] = df[col].str.replace(r'[\"\',]', '')

    # data clean by rule
    df[col] = df[col].str.replace('唄', '呗')
    df[col] = df[col].str.replace('花贝', '花呗')
    df[col] = df[col].str.replace('花吧', '花呗')
    df[col] = df[col].str.replace('花被', '花呗')

    df[col] = df[col].str.replace('借贝', '借呗')
    df[col] = df[col].str.replace('借吧', '借呗')
    df[col] = df[col].str.replace('借被', '借呗')

    return df


def data_loader(filename, path):
    df = pd.read_csv(os.path.join(path, filename + '.csv'), encoding='utf-8')
    return df


if __name__ == '__main__':
    # build_dataset(unsupervised=True)
    build_dataset()
