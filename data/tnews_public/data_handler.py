# coding=utf-8
import json
import unicodedata

import jieba.analyse
import numpy as np
import pandas as pd

from parm import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

stop_words = [line.strip() for line in
              open(os.path.join(PATH_DATA_TNEWS, 'stopword.txt'), 'r', encoding='utf-8').readlines()]


def data_clean(txt):
    txt_norm = unicodedata.normalize('NFKC', str(txt))
    return txt_norm


def get_labels():
    with open(os.path.join(PATH_DATA_TNEWS_PRE, 'labels.json'), 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return labels


def data_norm_for_wv(filename, pretrain=None, dropduplicate=None):
    if pretrain:
        from classifier.nets.wv import MODEL_FILE
        model_path, vec_dim = MODEL_FILE[pretrain]
        jieba.load_userdict(os.path.join(model_path, 'vocab_wv.txt'))

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

    if dropduplicate:
        df['token'] = df['sentence'].apply(lambda x: sorted(set(jieba.cut(x)).difference(set(stop_words)), key=x.index))
    else:
        df['token'] = df['sentence'].apply(lambda x: jieba.lcut(x))

    df['token'] = df['keywords'] + df['token']
    df = df[df['token'].notnull()]

    df['token'] = df['token'].apply(lambda x: ' '.join(x))
    df['token'] = df['token'].str.strip()
    df = df[df['token'] != ' ']
    df = df[df['token'] != '']
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


def data_norm_for_tfidf():
    df1 = pd.read_csv(os.path.join(PATH_DATA_TNEWS_PRE, 'train' + '.csv'), encoding='utf-8')
    df2 = pd.read_csv(os.path.join(PATH_DATA_TNEWS_PRE, 'dev' + '.csv'), encoding='utf-8')
    df = df1.append(df2, ignore_index=True)

    all_text = df['token'].values

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    count_v0 = CountVectorizer()
    counts_all = count_v0.fit_transform(all_text)

    count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_train = count_v1.fit_transform(df1['token'].values)
    print("the shape of train is " + repr(counts_train.shape))

    count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_dev = count_v2.fit_transform(df2['token'].values)
    print("the shape of dev is " + repr(counts_dev.shape))

    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit(counts_train).transform(counts_train)
    dev_data = tfidftransformer.fit(counts_dev).transform(counts_dev)

    # np.savez_compressed(os.path.join(PATH_DATA_TNEWS_PRE, 'train_tfidf.npz'), x=train_data, y=df1['label'].values)
    # np.savez_compressed(os.path.join(PATH_DATA_TNEWS_PRE, 'dev_tfidf.npz'), x=dev_data, y=df2['label'].values)

    return train_data, df1['label'].values, dev_data, df2['label'].values


def data_norm_for_fasttext_vec(train=None):
    import fasttext
    # df1 = pd.read_csv(os.path.join(PATH_DATA_TNEWS_PRE, 'train' + '.csv'), encoding='utf-8')
    # df2 = pd.read_csv(os.path.join(PATH_DATA_TNEWS_PRE, 'dev' + '.csv'), encoding='utf-8')
    # df = df1.append(df2, ignore_index=True)
    #
    # all_text = df['token'].values

    if train:
        model = fasttext.train_unsupervised(
            os.path.join(PATH_DATA_TNEWS_PRE, 'train_ft.txt'),
            # model='cbow',
            lr=0.05,
            dim=200,
            epoch=40,
            ws=5,
            minCount=1,
            minn=1,
            maxn=3)
        model.save_model(
            os.path.join(PATH_MD_FT, 'model_ft_selftrain.pkl'))

    model = fasttext.load_model(os.path.join(PATH_MD_FT, 'model_ft_selftrain.pkl'))
    print(model.get_subwords('体育运动真好'))
    # print(len(model.get_sentence_vector('体育 运动 真好')))


def data_check(file_name):
    df = pd.read_csv(os.path.join(PATH_DATA_TNEWS_PRE, file_name + '.csv'), encoding='utf-8')
    print(df.columns.tolist())
    print(df['label'].value_counts(normalize=True) * 100)
    print(df['label'].value_counts())


if __name__ == '__main__':
    # print(get_labels())
    # data_norm_for_wv('train')  # 53360
    # data_norm_for_wv('dev')  # 10000
    # data_norm_for_wv('test')  # 10000

    # data_norm_for_fasttext_fmt('train')
    # data_norm_for_fasttext_fmt('dev')

    # data_norm_for_selftrain()

    data_norm_for_fasttext_vec(train=True)
