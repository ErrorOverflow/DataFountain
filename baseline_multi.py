import pandas as pd
import lightgbm as lgb
import warnings
from sklearn.model_selection import StratifiedKFold
import numpy as np
import math

warnings.filterwarnings('ignore')
n_splits = 10
seed = 1030
train = pd.read_csv('train_2_fresh.csv')
train.pop('user_id')
test = pd.read_csv('test_2_fresh.csv')

params = {
    "learning_rate": 0.5,
    "lambda_l1": 0.0,
    "lambda_l2": 0.1,
    "max_depth": 10,
    "num_class": 5,
    "objective": "multiclass",
    "num_leaves": 63,
    'num_threads': 4,
    'max_bin': 511,
    'metric': None
}


def f1_score(predict, data):
    label = data.get_label()
    predict = np.argmax(predict.reshape(5, -1), axis=0)
    k, f1 = 0, 0
    precision, recall, TP, FP, FN = [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5

    for kl in label:
        kp = predict[k]
        if kp == kl:
            TP[kp] += 1
        else:
            FP[kp] += 1
            FN[kl] += 1
        k += 1

    for j in range(0, 5):
        precision[j] = TP[j] / (TP[j] + FP[j])
        recall[j] = TP[j] / (TP[j] + FN[j])
        f1 += (2 * precision[j] * recall[j]) / (precision[j] + recall[j])

    score = math.pow((1.00 / 5.00) * f1, 2)

    return 'f1_score', score, True


def lgb_train(train_sample, test_sample, param):
    # 对标签编码 映射关系
    label2current_service = dict(
        zip(range(0, len(set(train_sample['current_service']))), sorted(list(set(train_sample['current_service'])))))
    current_service2label = dict(
        zip(sorted(list(set(train_sample['current_service']))), range(0, len(set(train_sample['current_service'])))))
    # 原始数据的标签映射
    train_sample['current_service'] = train_sample['current_service'].map(current_service2label)
    # 构造原始数据
    train_real = train_sample.pop('current_service')
    train_X = train_sample
    train_col = train_sample.columns
    test_id = test_sample['user_id']
    train_X, test_X = train_X.values, test_sample[train_col].values
    cv_predict = []
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)  # sklearn divide 10
    for index, (train_index, test_index) in enumerate(skf.split(train_X, train_real)):  # 后一项为元组

        X_train, X_valid, y_train, y_valid = train_X[train_index], train_X[test_index], train_real[train_index], \
                                             train_real[test_index]
        train_data = lgb.Dataset(X_train, label=y_train)
        validation_data = lgb.Dataset(X_valid, label=y_valid)

        clf = lgb.train(param, train_data, num_boost_round=100000, valid_sets=[validation_data],
                        early_stopping_rounds=100,
                        feval=f1_score, verbose_eval=10)
        print(clf.best_iteration)
        y_test = clf.predict(test_X, num_iteration=clf.best_iteration)
        y_test = [np.argmax(x) for x in y_test]

        if index == 0:
            cv_predict = np.array(y_test).reshape(-1, 1)
        else:
            cv_predict = np.hstack((cv_predict, np.array(y_test).reshape(-1, 1)))
    # 投票
    submit = []
    for line in cv_predict:
        submit.append(np.argmax(np.bincount(line)))
    # 保存结果
    df_test = pd.DataFrame()
    df_test['id'] = list(test_id.unique())
    df_test['predict'] = submit
    df_test['predict'] = df_test['predict'].map(label2current_service)
    return df_test
    # df_test.to_csv('result_2.csv', index=False)


train1 = train
train1 = train1.replace(89950166, 1)
train1 = train1.replace(89950167, 1)
train1 = train1.replace(89950168, 1)
train1 = train1.replace(99999825, 2)
train1 = train1.replace(99999826, 2)
train1 = train1.replace(99999827, 2)
train1 = train1.replace(99999828, 2)
train1 = train1.replace(99999830, 2)
round1 = lgb_train(train1, test, params)
round1.to_csv('round1.csv', index=False)
