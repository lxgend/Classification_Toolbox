# coding=utf-8
import json
import unicodedata

import jieba.analyse
import pandas as pd

from parm import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

stop_words = [line.strip() for line in open(('stopword.txt'), 'r', encoding='utf-8').readlines()]


def data_clean(txt):
    txt_norm = unicodedata.normalize('NFKC', str(txt))
    return txt_norm


def get_labels():
    with open(os.path.join(PATH_DATA_TNEWS_PRE, 'labels.json'), 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return labels


def data_norm_for_wv(filename):
    id2label = get_labels()
    label2id = {v: k for k, v in id2label.items()}

    df = pd.read_json(filename, lines=True)

    print(df.columns.tolist())
    if filename != 'test.json':
        df['label'] = df['label'].apply(lambda x: label2id[str(x)])

    df['sentence'] = df['sentence'].apply(lambda x: data_clean(x))
    df['keywords'] = df['keywords'].apply(lambda x: data_clean(x))
    df['keywords'] = df['keywords'].apply(lambda x: x.split(','))

    # keep original order
    df['token'] = df['sentence'].apply(lambda x: sorted(set(jieba.cut(x)).difference(set(stop_words)), key=x.index))
    df['token'] = df['keywords'] + df['token']

    df['token'] = df['token'].apply(lambda x: ' '.join(x))
    df = df.drop(['sentence', 'keywords'], axis=1)

    print(df.head())

    df.to_csv(os.path.join(PATH_DATA_TNEWS_PRE, filename + '_norm.csv'), index=False, encoding='utf-8')


def data_norm(file_name):
    pass


if __name__ == '__main__':
    # print(get_labels())
    data_norm_for_wv('train.json')
    data_norm_for_wv('dev.json')
    data_norm_for_wv('test.json')
