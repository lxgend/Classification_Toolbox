# coding=utf-8
import jieba.analyse
import numpy as np
from tqdm import tqdm

from parm import *


# VOCAB = [word for word in WV.vocab]
# WV_UNK = np.mean(WV.wv[VOCAB], axis=0)
# vector_size = len(WV_UNK)

class InputFeatures(object):
    def __init__(self, sentence_vec, label):
        self.sentence_vec = sentence_vec
        self.label = label


def convert_examples_to_features_df(args, examples_df):
    if args.model_type == 'sg_tx':
        jieba.load_userdict(os.path.join(args.model_path, 'vocab_wv.txt'))
        args.unk = np.zeros(args.vec_dim, dtype=np.float32)

    def get_word_vector(word, args):
        if word in args.word2id.keys():
            idx = args.word2id[word]
            wv = args.wv_model[idx]
        else:
            wv = args.unk
        return wv

    def txt2vec(row):
        tokens = jieba.lcut(row['text_a'])
        wv_matrix = np.array(list(map(lambda w: get_word_vector(w, args), tokens)),
                             dtype=np.float32)

        if wv_matrix.size == 0:
            # 叠加几个subtoken
            wv_sup = np.broadcast_to(args.unk, (len(tokens), len(args.unk)))
            wv_matrix = np.vstack(wv_sup)

        sentence_vec = np.mean(wv_matrix, axis=0, dtype=np.float32)

        if np.all(sentence_vec == 0):
            sentence_vec = args.unk
        return sentence_vec

    tqdm.pandas(desc='mybar')

    if args.model_type == 'fasttext_selftrain':
        examples_df['vec'] = examples_df['text_a'].progress_apply(lambda x: args.model.get_sentence_vector(x))
    else:
        examples_df['vec'] = examples_df.progress_apply(txt2vec, axis=1)


    vecs = examples_df['vec'].values
    labels = examples_df['label'].astype(int).values

    return vecs, labels


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
