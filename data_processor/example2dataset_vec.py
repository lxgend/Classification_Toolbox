# coding=utf-8

import jieba.analyse
import numpy as np

from parm import *


# VOCAB = [word for word in WV.vocab]
# WV_UNK = np.mean(WV.wv[VOCAB], axis=0)
# vector_size = len(WV_UNK)

class InputFeatures(object):
    def __init__(self, sentence_vec, label):
        self.sentence_vec = sentence_vec
        self.label = label


def get_word_vector(word, idx2word, vector_all):
    if word in idx2word.keys():
        iid = idx2word[word]
    else:
        iid = idx2word['pad']
    wv = vector_all[str(iid)]
    return wv


def word2onehot(word, idx2word):
    if word in idx2word.keys():
        iid = idx2word[word]
    else:
        iid = idx2word['pad']
    wv = vector_all[str(iid)]
    return wv


def embedding_table(vector_all):
    wv = vector_all[str()]
    print(wv.shape)
    print(wv)

    # for i in range()
    #
    # wv_matrix = np.vstack(wv_sup)
    # if word in idx2word.keys():
    #     iid = idx2word[word]
    # else:
    #     iid = idx2word['pad']
    # wv = vector_all[str(iid)]
    # return wv


def convert_examples_to_features_df(args, examples_df):
    WV_UNK = np.zeros(args.vec_dim, dtype=np.float32)

    def txt2vec(row):
        print(row)
        print(row[2])
        tokens = jieba.lcut


    examples_df['vec'] = examples_df.apply(txt2vec, axis=1)

    vecs = examples_df['vec'].values
    labels = examples_df['label'].astype(int).values

    return vecs, labels


    # features = []
    # for (ex_index, example) in enumerate(examples_df):
    #
    #     print(example)
    #
    #     tokens = jieba.lcut(example.text_a)
    #
    #     if args.model_type != 'fasttext':
    #         # avoid oov
    #         # word_no_exist = [word for word in tokens if word not in vocab]
    #         word_exist = [word for word in tokens if word in args.wv_model.vocab]
    #         tokens = word_exist
    #         if not tokens:
    #             features.append(
    #                 InputFeatures(sentence_vec=WV_UNK, label=example.label))
    #             continue
    #
    #         wv_matrix = np.array(list(map(lambda x: args.wv_model[x], tokens)), dtype=np.float32)
    #     else:
    #         wv_matrix = np.array(list(map(lambda x: get_word_vector(x, args.idx2word, args.wv_model), tokens)),
    #                              dtype=np.float32)
    #
    #     if wv_matrix.size == 0:
    #         # 叠加几个subtoken
    #         wv_sup = np.broadcast_to(WV_UNK, (len(tokens), len(WV_UNK)))
    #         wv_matrix = np.vstack(wv_sup)
    #
    #     sentence_vec = np.sum(wv_matrix, axis=0, dtype=np.float32)
    #
    #     if np.all(sentence_vec == 0):
    #         sentence_vec = WV_UNK
    #
    #     features.append(
    #         InputFeatures(sentence_vec=sentence_vec, label=example.label))
    # return features


def convert_examples_to_features_df2(args, examples_df):
    WV_UNK = np.zeros(args.vec_dim, dtype=np.float32)

    features = []
    for (ex_index, example) in enumerate(examples_df):

        print(example)

        tokens = jieba.lcut(example.text_a)

        if args.model_type != 'fasttext':
            # avoid oov
            # word_no_exist = [word for word in tokens if word not in vocab]
            word_exist = [word for word in tokens if word in args.wv_model.vocab]
            tokens = word_exist
            if not tokens:
                features.append(
                    InputFeatures(sentence_vec=WV_UNK, label=example.label))
                continue

            wv_matrix = np.array(list(map(lambda x: args.wv_model[x], tokens)), dtype=np.float32)
        else:
            wv_matrix = np.array(list(map(lambda x: get_word_vector(x, args.idx2word, args.wv_model), tokens)),
                                 dtype=np.float32)

        if wv_matrix.size == 0:
            # 叠加几个subtoken
            wv_sup = np.broadcast_to(WV_UNK, (len(tokens), len(WV_UNK)))
            wv_matrix = np.vstack(wv_sup)

        sentence_vec = np.sum(wv_matrix, axis=0, dtype=np.float32)

        if np.all(sentence_vec == 0):
            sentence_vec = WV_UNK

        features.append(
            InputFeatures(sentence_vec=sentence_vec, label=example.label))
    return features




def load_and_cache_examples_df(args, processor, data_type='train'):
    # load wv

    # filename
    cached_features_file = os.path.join(args.data_dir, '{}_{}_{}.npz'.format(
        data_type,
        str(args.model_type),
        str(args.task_name)))

    if os.path.exists(cached_features_file) and args.overwrite_cache == 0:
        vector_all = np.load(cached_features_file, allow_pickle=True)
        vecs = vector_all['x']
        labels = vector_all['y']
    else:
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        else:
            examples = processor.get_test_examples()

        vecs, labels = convert_examples_to_features_df(args, examples)

        np.savez_compressed(cached_features_file, x=vecs, y=labels)

    vecs = vecs.tolist()
    labels = labels.tolist()

    return vecs, labels


if __name__ == '__main__':
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
