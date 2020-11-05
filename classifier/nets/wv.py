# coding=utf-8
from parm import *

FASTTEXT_PATH = os.path.join(PATH_MD_FT, 'cc.zh.300.vec.gz')
WV_PATH = '/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt'

MODEL_FILE = {
    'fasttext': (FASTTEXT_PATH, 300),
    'sg_tx': (WV_PATH, 200)
}

if __name__ == '__main__':
    from gensim.models import KeyedVectors
    WV = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False)

    print(WV['你好'])
    print(len(WV['你好']))
