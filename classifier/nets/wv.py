# coding=utf-8

# from classifier.vec_processor.vecfile2npyfile import load_model
# id2word, vector_all = load_model('/home/dc2-user/')

FASTTEXT_PATH = '/home/dc2-user/'
WV_PATH = '/Users/lixiang/Documents/nlp_data/pretrained_model/tx_w2v/45000-small.txt'
# WV_PATH = '/home/dc2-user/45000-small.txt'

MODEL_FILE = {
    'fasttext': (FASTTEXT_PATH, 300),
    'sg_tx': (WV_PATH, 200)
}


if __name__ == '__main__':
    # from gensim.models import KeyedVectors
    #
    # WV = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False)
    #
    # print(WV['你好'])
    # print(len(WV['你好']))
    pass

    # print(get_word_vector('你好'))
