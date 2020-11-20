# coding=utf-8
import lightgbm as lgb
import numpy as np

data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
label = np.random.randint(2, size=500)  # binary target
train_data = lgb.Dataset(data, label=label)


train_data.save_binary('train.bin')

validation_data = train_data.create_valid('validation.svm')


train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])

w = np.random.rand(500, )
train_data = lgb.Dataset(data, label=label, weight=w)


num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])


bst.save_model('model.txt')



# 7 entities, each contains 10 features
data = np.random.rand(7, 10)
ypred = bst.predict(data)

clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=10, learning_rate=0.1, n_estimators=100, \
                         subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, \
                         min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, \
                         random_state=None, n_jobs=-1, silent=True, importance_type='split')

clf.fit(xtrain_ctv,ytrain)

LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
        importance_type='split', learning_rate=0.1, max_depth=10,
        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
        n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
        random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
        subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

predictions = clf.predict_proba(xvalid_ctv)