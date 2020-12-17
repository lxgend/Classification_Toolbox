# coding=utf-8
import joblib
import lightgbm as lgb
import numpy as np
from scipy.misc import derivative
from sklearn.metrics import classification_report

from classifier.nets.wv import MODEL_FILE
from data_processor.data2example import clf_data_processors
from data_processor.example2dataset_vec import load_and_cache_examples_df
from parm import *


def focal_loss_lgb_sk(y_true, y_pred, alpha, gamma, num_class):
    """
    Parameters:
    -----------
    alpha, gamma: float
    objective(y_true, y_pred) -> grad, hess

    y_truearray-like of shape = [n_samples]
    The target values.
    y_predarray-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
    The predicted values.
    """
    a, g = alpha, gamma

    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class)

    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return -(a * t + (1 - a) * (1 - t)) * ((1 - (t * p + (1 - t) * (1 - p))) ** g) * (
                t * np.log(p) + (1 - t) * np.log(1 - p))

    partial_fl = lambda x: fl(x, y_true)

    # 求导数
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)

    # return grad.flatten('F'), hess.flatten('F')
    return grad, hess


def focal_loss_lgb_eval_error_sk(y_true, y_pred, alpha, gamma, num_class):
    """
    Adapation of the Focal Loss for lightgbm to be used as evaluation loss
    """
    a, g = alpha, gamma
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')

    p = 1 / (1 + np.exp(-y_pred))
    loss = -(a * y_true + (1 - a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g) * (
            y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return 'focal_loss', np.mean(loss), False


def train(x_train, y_train):
    num_class =15
    focal_loss = lambda y_true, y_pred: focal_loss_lgb_sk(y_true, y_pred, 0.25, 2., num_class)
    eval_error = lambda x, y: focal_loss_lgb_eval_error_sk(x, y, 0.25, 2., num_class)

    params = {
        'boosting_type': 'gbdt',
        'max_depth': 6,
        'num_leaves': 60,
        'n_estimators': 200,
        'objective': focal_loss,
        # 'objective': 'multiclass',
        'max_bin': 150,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,

        # 'class_weight':weight
        'n_jobs': 8,
        'learning_rate': 0.1,
        #'num_class':15
        # 'silent': False
    }

    model = lgb.LGBMClassifier(**params)
    # model.fit(x_train, y_train,
    #           eval_set=[(x_dev, y_dev)],
    #           eval_metric=eval_error)

    model.fit(x_train, y_train)

    # from sklearn.model_selection import GridSearchCV
    # lg = lgb.LGBMClassifier(silent=False, verbose=-1)
    # # 评分函数
    # mll_scorer = make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
    # # max_depth : 树最大深度， 模型过拟合可以降低max_depth
    # # num_leaves: 取值应 <= 2 ^（max_depth）， 超过此值会导致过拟合
    # # min_data_in_leaf
    # param_dist = {"max_depth": [10, 25, 50, 75],
    #               "learning_rate": [0.01, 0.05, 0.1],
    #               "num_leaves": [300, 500, 900, 1200],
    #               "n_estimators": [150, 200, 250],
    #               }
    #
    # parameters = {
    #     'max_depth': [15, 20, 25, 30, 35],
    #     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    #     'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
    #     'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
    #     'bagging_freq': [2, 4, 5, 6, 8],
    #     'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
    #     'lambda_l2': [0, 10, 15, 35, 40],
    #     'cat_smooth': [1, 10, 15, 20, 35]
    # }
    #

    with open('model_lgbm.pkl', mode='wb') as f:
        joblib.dump(model, f)


def evaluate(x_dev, y_dev, model):
    # 模型预测
    y_pred = model.predict(x_dev, num_iteration=model.best_iteration_)

    # 查看各个类别的准召
    print(classification_report(y_dev, y_pred))


def main(args):
    # data init
    clf_data_processor = clf_data_processors[args.task_name](args.data_dir)
    args.id2label = clf_data_processor.get_labels()
    args.label2id = {label: i for i, label in enumerate(args.id2label)}
    num_labels = len(args.id2label)

    print('num_labels %d' % (num_labels))
    print('model %s' % args.model_type)

    if args.model_type == 'fasttext_selftrain':
        import fasttext
        args.model = fasttext.load_model(os.path.join(PATH_MD_FT, 'model_ft_selftrain.pkl'))
        args.vec_dim = 200
    else:
        args.model_path, args.vec_dim = MODEL_FILE[args.model_type]
        args.word2id, args.wv_model = load_model(args.model_path)

    if args.do_train:
        x_train, y_train = load_and_cache_examples_df(args, clf_data_processor, data_type='train')

        # print(len(x_train))
        # print(len(x_train[0]))
        # print(y_train.shape)

        print('train_dataset %d' % len(y_train))

        # x_dev, y_dev = load_and_cache_examples_df(args, clf_data_processor, data_type='dev')

        # train
        train(x_train, y_train)

    if args.do_eval:
        print('evaluate')
        x_dev, y_dev = load_and_cache_examples_df(args, clf_data_processor, data_type='dev')
        print('dev_dataset %d' % len(y_dev))
        with open('model_lgbm.pkl', mode='rb') as f:
            model = joblib.load(f)
        evaluate(args, x_dev, y_dev, model)


class Args(object):
    def __init__(self):
        self.task_name = 'tnews_vec'
        self.data_dir = PATH_DATA_TNEWS_PRE
        # self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuned')
        self.overwrite_cache = 1
        self.max_seq_length = 42
        # self.model_type = 'sg_tx'
        self.model_type = 'fasttext_selftrain'

        self.local_rank = -1
        self.use_cpu = 0

        self.do_train = 1
        self.do_eval = 0
        self.do_test = 0


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    import time

    a = time.time()

    args = Args()
    # main2(args)

    main(args)

    print(time.time() - a)
