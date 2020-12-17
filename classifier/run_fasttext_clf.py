# coding=utf-8
import os

import fasttext

from parm import PATH_DATA_TNEWS_PRE


def train():
    model = fasttext.train_supervised(
        os.path.join(PATH_DATA_TNEWS_PRE, 'train_ft.txt'),
        lr=0.1,
        ws=5,
        dim=300,
        epoch=50,
        neg=5,
        label="__label__"
    )
    model.save_model('model_ft.pkl')


def eval():
    model = fasttext.load_model('model_ft.pkl')
    result = model.test(os.path.join(PATH_DATA_TNEWS_PRE, 'dev_ft.txt'))
    print('y_pred = ', result)
    print('size: ', result[0])
    print('precision: ', result[1])
    print('recall: ', result[2])


def predict():
    model = fasttext.load_model('model_ft.pkl')
    result = model.predict(['如何 办 卡'], k=5)
    # 前k的label和置信度
    print(result)
    print(model.get_subwords('体育运动'))  # wordNgrams=对中文无效


if __name__ == '__main__':
    train()
    eval()
    # predict()
