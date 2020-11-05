# coding=utf-8
from typing import List

import jieba.analyse
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from parm import *

# VOCAB = [word for word in WV.vocab]
# WV_UNK = np.mean(WV.wv[VOCAB], axis=0)
# vector_size = len(WV_UNK)

WV_UNK = np.zeros(100, dtype=np.float32)

class InputFeatures(object):
    def __init__(self, sentence_vec, label):
        self.sentence_vec = sentence_vec
        self.label = label

def convert_examples_to_features(examples,
                                 wv_model,
                                 ngram,
                                 task,
                                 label_list=None) -> List[InputFeatures]:
    label2id = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):

        tokens = jieba.lcut(example.text_a)

        if ngram is False:
            # word_no_exist = [word for word in tokens if word not in vocab]
            word_exist = [word for word in tokens if word in wv_model.vocab]
            tokens = word_exist

        wv_to_sum = np.array(list(map(lambda x: wv_model, tokens)), dtype=np.float32)

        if wv_to_sum.size == 0:
            # 叠加几个subtoken
            wv_sup = np.broadcast_to(WV_UNK, (len(tokens), len(WV_UNK)))
            wv_to_sum = np.vstack(wv_sup)

        sentence_vec = np.sum(wv_to_sum, axis=0, dtype=np.float32)

        if np.all(sentence_vec == 0):
            sentence_vec = WV_UNK

        features.append(
            InputFeatures(sentence_vec=sentence_vec,
                          label=label2id[example.label]))
    return features


def load_and_cache_examples(args, tokenizer, processor, data_type='train'):
    label_list = processor.get_labels()
    # filename
    cached_features_file = os.path.join(args.data_dir, '{}_{}_{}.cache'.format(
        data_type,
        str(args.max_seq_length),
        str(args.task_name)))

    if os.path.exists(cached_features_file) and args.overwrite_cache == 0:
        features = torch.load(cached_features_file)
    else:

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        # max_length = processor.get_max_length()
        max_length = 200
        features = convert_examples_to_features(examples,
                                                wv_model,
                                                ngram=False,
                                                task=args.task_name,
                                                label_list=label_list
                                                )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


class ModelProcessor(object):
    def __init__(self, wv_model):
        self.wv_model = wv_model
        self.vocab = [word for word in wv_model.wv.vocab]
        self.wv_unk = np.mean(wv_model.wv[self.vocab], axis=0)
        self.vector_size = len(self.wv_unk)

    def get_sts_vector(self, df, col):
        df, col_token = get_token(df, col, drop_stop_word=True)

        col_wv = 'wv'
        wv_init = np.zeros(self.vector_size)

        wv_model = self.wv_model
        wv_unk = self.wv_unk

        def use_wv_plm(row):
            # init
            wv_sum = wv_init
            for word in row[col_token]:
                try:
                    wv_sum = np.add(wv_sum, wv_model.wv[word])
                except:
                    # find unk word
                    # init
                    wv_sub_sum = wv_init
                    for char in word:
                        try:
                            # unk word use subtoken vector
                            wv_sub_sum = np.add(wv_sub_sum, wv_model.wv[char])
                        except:
                            # unk word use unk vector
                            wv_sub_sum = np.add(wv_sub_sum, wv_unk)
                    wv_sum = np.add(wv_sum, wv_sub_sum)
            if np.all(wv_sum == 0):
                wv_sum = wv_unk
            return wv_sum

        df.loc[:, col_wv] = df.apply(use_wv_plm, axis=1)
        return df, col_wv


def build(df, col, save=False):
    df, col_token = get_token(df, col, drop_stop_word=True)
    sents = [row.split(' ') for row in df[col_token]]

    model_trained = Word2Vec(sents, size=100, window=5, sg=0, min_count=1)
    if save:
        model_trained.save(os.path.join(PATH_MD_TMP, 'wv_trained.txt'))
    return model_trained


def load(filename, entire_model=False):
    if entire_model:
        model_trained = Word2Vec.load(filename)
    else:
        model_trained = KeyedVectors.load_word2vec_format(filename, binary=False)
    return model_trained


def match(df_kg, df_c, col_wv):
    col_max_value = 'max_sim_value'
    col_max_idx = 'max_sim_value_idx'

    matrix_query = np.vstack(df_kg[col_wv])
    matrix_candidate = np.vstack(df_c[col_wv])

    matrix_query_norm = matrix_query / (np.linalg.norm(matrix_query, axis=1, keepdims=True))
    matrix_candidate_norm = matrix_candidate / (np.linalg.norm(matrix_candidate, axis=1, keepdims=True))

    sim_calc = np.dot(matrix_query_norm, matrix_candidate_norm.T)

    max_sim_value = np.max(sim_calc, axis=1)
    max_sim_value_idx = np.argmax(sim_calc, axis=1)

    df_result = pd.DataFrame({col_max_value: max_sim_value, col_max_idx: max_sim_value_idx},
                             columns=[col_max_value, col_max_idx])

    df_kg = pd.concat([df_kg, df_result], axis=1)

    df_kg[['result']] = df_kg[col_max_idx].apply(lambda x: df_c.iloc[x][[COL_TXT]])

    from sklearn.metrics import accuracy_score
    print(accuracy_score(df_kg[COL_ST2], df_kg['result']))

    return df_kg


if __name__ == '__main__':
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # train
    from data_processor.data_builder import data_loader

    df = data_loader(filename='candidate', path=PATH_DATA_PRE)
    df[COL_CLS] = df[COL_CLS].astype(int)
    model_trained = build(df, COL_TXT, save=True)

    # use
    model_trained = load(os.path.join(PATH_MD_TMP, 'wv_trained.txt'), entire_model=True)

    # model_trained = load('/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt')

    wv_model = ModelProcessor(model_trained)
    df_kg = data_loader(filename='kg', path=PATH_DATA_PRE)
    # df_kg = df_kg.head()
    df_kg, col_wv = wv_model.get_sts_vector(df_kg, COL_ST1)

    df_c = data_loader(filename='candidate', path=PATH_DATA_PRE)
    df_c, col_wv = wv_model.get_sts_vector(df_c, COL_TXT)

    df_kg = match(df_kg, df_c, col_wv)
