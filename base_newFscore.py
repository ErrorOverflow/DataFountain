import pandas as pd
import lightgbm as lgb
import warnings
from sklearn.model_selection import StratifiedKFold
import numpy as np
import math

warnings.filterwarnings('ignore')
n_splits = 10
seed = 1030
train = pd.read_csv('train_2_dup_del.csv')
test = pd.read_csv('test_2.csv')


def f1_score(pred, data):
    label = data.get_label()
    pred = np.argmax(pred.reshape(11, -1), axis=0)
    k, f1 = 0, 0
    precision, recall, TP, FP, FN = [0] * 11, [0] * 11, [0] * 11, [0] * 11, [0] * 11

    for kl in label:
        kp = pred[k]
        if kp == kl:
            TP[kp] += 1
        else:
            FP[kp] += 1
            FN[kl] += 1
        k += 1

    for j in range(0, 11):
        precision[j] = TP[j] / (TP[j] + FP[j])
        recall[j] = TP[j] / (TP[j] + FN[j])
        f1 += (2 * precision[j] * recall[j]) / (precision[j] + recall[j])

    score = math.pow((1.00 / 11.00) * f1, 2)

    return 'f1_score', score


params = {
    "learning_rate": 0.1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "max_depth": 10,
    "num_class": 11,
    "objective": "multiclass",
    "num_leaves": 63,
    'num_threads': 4,
    'max_bin': 511,
    'metric': None
}


def f1_score(pred, data):
    label = data.get_label()
    pred = np.argmax(pred.reshape(11, -1), axis=0)
    k, f1 = 0, 0
    precision, recall, TP, FP, FN = [0] * 11, [0] * 11, [0] * 11, [0] * 11, [0] * 11

    for kl in label:
        kp = pred[k]
        if kp == kl:
            TP[kp] += 1
        else:
            FP[kp] += 1
            FN[kl] += 1
        k += 1

    for j in range(0, 11):
        precision[j] = TP[j] / (TP[j] + FP[j])
        recall[j] = TP[j] / (TP[j] + FN[j])
        f1 += (2 * precision[j] * recall[j]) / (precision[j] + recall[j])

    score = math.pow((1.00 / 11.00) * f1, 2)

    return 'f1_score', score, True


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


# 对标签编码 映射关系
label2current_service = dict(
    zip(range(0, len(set(train['current_service']))), sorted(list(set(train['current_service'])))))
current_service2label = dict(
    zip(sorted(list(set(train['current_service']))), range(0, len(set(train['current_service'])))))
# 原始数据的标签映射
train['current_service'] = train['current_service'].map(current_service2label)
# 构造原始数据
y = train.pop('current_service')
train_id = train.pop('user_id')
# 好像有点问题
X = train
train_col = train.columns
X_test = test[train_col]
test_id = test['user_id']
# 有问题数据
for i in train_col:
    X[i] = X[i].replace("\\N", -1)
    X_test[i] = X_test[i].replace("\\N", -1)

X, y, X_test = X.values, y, X_test.values

cv_pred = []

skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)  # sklearn divide 10
for index, (train_index, test_index) in enumerate(skf.split(X, y)):  # 后一项为元组

    X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, train_data, num_boost_round=100000, valid_sets=[validation_data], early_stopping_rounds=100,
                    feval=f1_score, verbose_eval=10)

    xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
    xx_pred = [np.argmax(x) for x in xx_pred]

    y_test = clf.predict(X_test, num_iteration=clf.best_iteration)
    y_test = [np.argmax(x) for x in y_test]

    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

# 投票
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

# 保存结果
df_test = pd.DataFrame()
df_test['id'] = list(test_id.unique())
df_test['predict'] = submit
df_test['predict'] = df_test['predict'].map(label2current_service)

df_test.to_csv('result_2.csv', index=False)
