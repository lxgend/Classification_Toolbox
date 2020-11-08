# coding=utf-8
from typing import List

import jieba.analyse
import joblib
import numpy as np

from parm import *


# VOCAB = [word for word in WV.vocab]
# WV_UNK = np.mean(WV.wv[VOCAB], axis=0)
# vector_size = len(WV_UNK)

class InputFeatures(object):
    def __init__(self, sentence_vec, label):
        self.sentence_vec = sentence_vec
        self.label = label


def convert_examples_to_features(args, examples) -> List[InputFeatures]:
    WV_UNK = np.zeros(args.vec_dim, dtype=np.float32)

    features = []
    for (ex_index, example) in enumerate(examples):

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

        wv_to_sum = np.array(list(map(lambda x: args.wv_model[x], tokens)), dtype=np.float32)

        if wv_to_sum.size == 0:
            # 叠加几个subtoken
            wv_sup = np.broadcast_to(WV_UNK, (len(tokens), len(WV_UNK)))
            wv_to_sum = np.vstack(wv_sup)

        sentence_vec = np.sum(wv_to_sum, axis=0, dtype=np.float32)

        if np.all(sentence_vec == 0):
            sentence_vec = WV_UNK

        features.append(
            InputFeatures(sentence_vec=sentence_vec, label=example.label))
    return features


def load_and_cache_examples(args, processor, data_type='train'):
    # filename
    cached_features_file = os.path.join(args.data_dir, '{}_{}_{}.cache'.format(
        data_type,
        str(args.max_seq_length),
        str(args.task_name)))

    if os.path.exists(cached_features_file) and args.overwrite_cache == 0:
        with open(cached_features_file, mode='rb') as f:
            features = joblib.load(f)
    else:
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        else:
            examples = processor.get_test_examples()

        features = convert_examples_to_features(args, examples)

        with open(cached_features_file, mode='wb') as f:
            joblib.dump(features, f)

    return features


if __name__ == '__main__':
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
