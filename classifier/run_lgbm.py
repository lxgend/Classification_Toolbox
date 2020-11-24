# coding=utf-8
import joblib
import lightgbm as lgb
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import classification_report

from classifier.nets.wv import MODEL_FILE
from data_processor.data2example import clf_data_processors
from data_processor.example2dataset_vec import load_and_cache_examples_df
from parm import *


def BayesianSearch(clf, params):
    """贝叶斯优化器"""
    # 迭代次数
    num_iter = 25
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    # 输出结果
    params = bayes.res['max']
    print(params['max_params'])
    return params


def GBM_evaluate(min_child_samples, min_child_weight, colsample_bytree, max_depth, subsample, reg_alpha, reg_lambda):
    """自定义的模型评估函数"""

    # 模型固定的超参数
    param = {
        'objective': 'regression',
        'n_estimators': 275,
        'metric': 'rmse',
        'random_state': 2018}

    # 贝叶斯优化器生成的超参数
    param['min_child_weight'] = int(min_child_weight)
    param['colsample_bytree'] = float(colsample_bytree),
    param['max_depth'] = int(max_depth),
    param['subsample'] = float(subsample),
    param['reg_lambda'] = float(reg_lambda),
    param['reg_alpha'] = float(reg_alpha),
    param['min_child_samples'] = int(min_child_samples)

    # 5-flod 交叉检验，注意BayesianOptimization会向最大评估值的方向优化，因此对于回归任务需要取负数。
    # 这里的评估函数为neg_mean_squared_error，即负的MSE。
    val = cross_val_score(lgb.LGBMRegressor(**param),
                          train_X, train_y, scoring='neg_mean_squared_error', cv=5).mean()

    return val


def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    print('actual')
    print(actual)
    print('predicted')
    print(predicted)

    if len(actual) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


def train(args, x_train, y_train):
    params = {
        'boosting_type': 'gbdt',
        'max_depth': 6,
        'num_leaves': 60,
        'n_estimators': 200,
        'objective': 'multiclass',
        'max_bin': 150,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
        # 'class_weight':weight
        'n_jobs': 8,
        'learning_rate': 0.1,
        'silent': False
    }

    model = lgb.LGBMClassifier(**params)
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
    # model = GridSearchCV(estimator=lg, param_grid=param_dist, scoring='r2',
    #                      verbose=10, n_jobs=10, iid=True, refit=True, cv=2)
    # # idd: identically distributed across the folds ;the loss minimized is the total loss per sample, and not the mean loss across the folds
    # model.fit(x_train, y_train)
    # print("Best score: %0.3f" % model.best_score_)
    # print("Best parameters set:")
    # best_parameters = model.best_estimator_.get_params()
    # for param_name in sorted(param_dist.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

    with open('model_lgbm.pkl', mode='wb') as f:
        joblib.dump(model, f)


def evaluate(args, x_dev, y_dev, model):
    # 模型预测
    y_pred = model.predict(x_dev, num_iteration=model.best_iteration_)

    # 查看各个类别的准召
    print(classification_report(y_dev, y_pred))


def predict(args, test_dataset, model):
    results = []


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
        print('train_dataset %d' % len(y_train))

        # train
        train(args, x_train, y_train)

    if args.do_eval:
        print('evaluate')
        x_dev, y_dev = load_and_cache_examples_df(args, clf_data_processor, data_type='dev')
        print('dev_dataset %d' % len(y_dev))
        with open('model_lgbm.pkl', mode='rb') as f:
            model = joblib.load(f)
        evaluate(args, x_dev, y_dev, model)

    if args.do_test:
        print('test')
        test_dataset = load_and_cache_examples_df(args, clf_data_processor, data_type='test')
        predict(args, test_dataset)


def main2(args):
    # data init

    from data.tnews_public.data_handler import data_norm_for_tfidf

    x_train, y_train, x_dev, y_dev = data_norm_for_selftrain()

    if args.do_train:

        print('train_dataset %d' % len(y_train))
        # train
        train(args, x_train, y_train)

    if args.do_eval:
        print('dev_dataset %d' % len(y_dev))
        with open('model_lgbm.pkl', mode='rb') as f:
            model = joblib.load(f)
        evaluate(args, x_dev, y_dev, model)


class Args(object):
    def __init__(self):
        self.task_name = 'tnews_vec'
        self.data_dir = PATH_DATA_TNEWS_PRE
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finetuned')
        self.overwrite_cache = 1
        self.max_seq_length = 42
        # self.model_type = 'sg_tx'
        self.model_type = 'fasttext_selftrain'

        self.local_rank = -1
        self.use_cpu = 0

        self.do_train = 1
        self.do_eval = 1
        self.do_test = 0


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    import time

    a = time.time()

    args = Args()
    main(args)

    print(time.time() - a)
