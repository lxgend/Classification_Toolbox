# coding=utf-8
import jieba.analyse

from data_processor.data_builder import data_loader
from parm import *

stop_words = [line.strip() for line in
              open((os.path.join(PATH_DATA_PRE, 'stopword.txt')), 'r', encoding='utf-8').readlines()]

def get_token(df, col, drop_stop_word=False):
    jieba.suggest_freq('花呗', True)
    jieba.suggest_freq('借呗', True)
    jieba.suggest_freq('网商贷', True)
    jieba.suggest_freq('支付宝', True)
    jieba.suggest_freq('***', True)

    if drop_stop_word:
        df[COL_CUT] = df[col].apply(lambda x: ' '.join((set(jieba.cut(x)).difference(set(stop_words)))))
    else:
        df[COL_CUT] = df[col].apply(lambda x: ' '.join(jieba.cut(x)))
    return df, COL_CUT


def classify(df, col):
    mask = df[col].str.contains('花呗')
    df.loc[mask, COL_CLS] = 1

    mask = df[col].str.contains('借呗')
    df.loc[mask, COL_CLS] = 2

    df[COL_CLS] = df[COL_CLS].fillna(3)

    return df, COL_CLS


def build_dataset_preprocessed(filename, col):
    df = data_loader(filename=filename, path=PATH_DATA_PRE)
    df, col_cls = classify(df, col)
    # df = df.sort_values(by=col_cls)
    df.to_csv(os.path.join(PATH_DATA_PRE, filename + '.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    # build_dataset_preprocessed('query', COL_TXT)
    # build_dataset_preprocessed('candidate', COL_TXT)
    # build_dataset_preprocessed('kg', 'sentence1')

    for f in ['train', 'test', 'dev']:
        build_dataset_preprocessed(f, 'sentence1')
