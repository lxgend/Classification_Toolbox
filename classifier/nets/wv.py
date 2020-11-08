# coding=utf-8
from parm import *

FASTTEXT_PATH = '/home/dc2-user/cc.zh.300.bin'
WV_PATH = '/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt'

MODEL_FILE = {
    'fasttext': (FASTTEXT_PATH, 300),
    'sg_tx': (WV_PATH, 200)
}

if __name__ == '__main__':
    # from gensim.models import KeyedVectors
    # WV = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False)
    #
    # print(WV['你好'])
    # print(len(WV['你好']))
    # from gensim.models import FastText
    # import fasttext
    # model = fasttext.load_model(FASTTEXT_PATH)

    from gensim.models import FastText

    # from gensim.models.fasttext_bin import load
    model = FastText.load(FASTTEXT_PATH)
    print(model['你好'])

    from gensim.models import FastText
    FASTTEXTFILE = "./fastText/cc.en.300.bin"
    FASTTEXT = FastText.load_fasttext_format(FASTTEXTFILE)

    # print(model.get_word_vector['你好'])
    # print(len(model['你好']))




