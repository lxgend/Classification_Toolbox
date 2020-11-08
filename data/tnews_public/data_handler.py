# coding=utf-8
import json
import unicodedata

import jieba.analyse
import pandas as pd
import numpy as np
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

    df = pd.read_json(filename + '.json', lines=True)

    print(len(df))

    print(df.columns.tolist())
    if filename != 'test':
        df['label'] = df['label'].apply(lambda x: label2id[str(x)])

    df['sentence'] = df['sentence'].apply(lambda x: data_clean(x))
    df['keywords'] = df['keywords'].apply(lambda x: data_clean(x))
    df['keywords'] = df['keywords'].apply(lambda x: x.split(','))

    # keep original order
    df['token'] = df['sentence'].apply(lambda x: sorted(set(jieba.cut(x)).difference(set(stop_words)), key=x.index))
    df['token'] = df['keywords'] + df['token']
    df['token'] = df['keywords'] + df['token']
    df = df[df['token'].notnull()]

    df['token'] = df['token'].apply(lambda x: ' '.join(x))
    df['token'] = df['token'].str.strip()
    df = df[df['token'] != ' ']
    df = df[df['token'].notnull()]
    df = df.drop(['sentence', 'keywords'], axis=1)

    print(len(df))

    df.to_csv(os.path.join(PATH_DATA_TNEWS_PRE, filename + '.csv'), index=False, encoding='utf-8')


def data_norm_for_fasttext_fmt(filename):
    '''
    __label__2 中新网 日电 日前 上海 国际
    __label__0 两人 被捕 警方 指控 非法
    __label__3 中旬 航渡 过程 美军 第一
    __label__1 强强 联手 背后 品牌 用户 双赢
    '''
    df = pd.read_csv(os.path.join(PATH_DATA_TNEWS_PRE, filename + '.csv'), encoding='utf-8')

    df = df.drop(['label_desc'], axis=1)
    df['label'] = df['label'].apply(lambda x: '%s%s' % ('__label__', x))

    print(df.head())

    np.savetxt(os.path.join(PATH_DATA_TNEWS_PRE, filename + '_ft.txt'), df.values, fmt="%s")



def data_norm(file_name):
    pass


if __name__ == '__main__':
    # print(get_labels())
    # data_norm_for_wv('train')   # 53360
    # data_norm_for_wv('dev')     # 10000
    # data_norm_for_wv('test')    # 10000

    data_norm_for_fasttext_fmt('train')
    data_norm_for_fasttext_fmt('dev')
