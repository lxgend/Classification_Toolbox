# coding=utf-8
import json
import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def model_norm_str(filename, path):
    id2word = dict()
    vec_all = dict()
    wv_model = KeyedVectors.load_word2vec_format(filename, binary=False, encoding='utf-8',
                                                 unicode_errors='ignore')
    vocab = wv_model.vocab
    word_list = vocab.keys()

    for idx, w in enumerate(word_list):
        id2word[w] = idx  # build index
        vec_all[str(idx)] = wv_model[w]
    np.savez_compressed(os.path.join(path, 'md_norm.npz'), **vec_all)
    with open(os.path.join(path, 'id2word.json'), 'w', encoding='utf-8') as f:
        json.dump(id2word, f, ensure_ascii=False)


def load_model_str(path):
    with open(os.path.join(path, 'id2word.json'), 'r', encoding='utf-8') as f:
        idx2word = json.load(f)

    vector_all = np.load(os.path.join(path, 'md_norm.npz'))

    return idx2word, vector_all


def model_norm(filename, path):
    word2id = dict()
    vec_all = list()
    wv_model = KeyedVectors.load_word2vec_format(filename, binary=False, encoding='utf-8',
                                                 unicode_errors='ignore')
    vocab = wv_model.vocab
    word_list = vocab.keys()

    for idx, w in enumerate(word_list):
        word2id[w] = idx  # build index
        vec_all.append(wv_model[w])

    np.savez_compressed(os.path.join(path, 'md_norm.npz'), vec_all)

    with open(os.path.join(path, 'word2id.json'), 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False)


def load_model(path):
    with open(os.path.join(path, 'word2id.json'), 'r', encoding='utf-8') as f:
        idx2word = json.load(f)

    vector_all = np.load(os.path.join(path, 'md_norm.npz'))
    vector_all = vector_all['arr_0']

    return idx2word, vector_all


def build_dict(path):
    with open(os.path.join(path, 'word2id.json'), 'r', encoding='utf-8') as f:
        id2word = json.load(f)
    df = pd.DataFrame(list(id2word.items()), columns=['a', 'b'])
    df['c'] = 999
    df['d'] = 'wv'
    df = df.drop(['b'], axis=1)

    df.to_csv(os.path.join(path, 'vocab_wv.txt'), index=False, sep=' ', header=False, encoding='utf-8')


if __name__ == '__main__':
    # model_norm(filename='/home/dc2-user/cc.zh.300.vec', path='/home/dc2-user/')
    filename = '/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt'
    path = '/Users/lixiang/Documents/Python/PycharmProjects/Workplace/Classification_Toolbox/classifier/vec_processor/vec_model/'
    # model_norm(filename, path)
    # word2id, vector_all = load_model(path)
    build_dict(path)

    # print(vector_all['pad'])