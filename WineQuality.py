import pandas as pd
import lightgbm as lgb
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

warnings.filterwarnings('ignore')
n_splits = 5
seed = 104
train = pd.read_csv('WineQualityTrain.csv')
test = pd.read_csv('WineQualityTest.csv')

params = {
    "learning_rate": 0.01,
    "lambda_l1": 1,
    "lambda_l2": 2,
    "max_depth": 5,
    "num_class": 2,
    "application": 'multiclass',
    # "num_leaves": 40,
    'num_threads': 4,
    # 'max_bin': 63,
    'boost_from_average': False
}


def f1_score(predict, data):
    label = data.get_label()
    predict = np.argmax(predict.reshape(2, -1), axis=0)
    k, f1 = 0, 0
    precision, recall, TP, FP, FN = [0] * 2, [0] * 2, [0] * 2, [0] * 2, [0] * 2

    for kl in label:
        kp = predict[k]
        if kp == kl:
            TP[kp] += 1
        else:
            FP[kp] += 1
            FN[kl] += 1
        k += 1
    precision = (TP[1] + TP[0]) / (TP[1] + FP[1] + TP[0] + FP[0])

    return 'f1_score', precision, True


# 对标签编码 映射关系
label2current_service = dict(
    zip(range(0, len(set(train['type']))), sorted(list(set(train['type'])))))
current_service2label = dict(
    zip(sorted(list(set(train['type']))), range(0, len(set(train['type'])))))
# 原始数据的标签映射
train['type'] = train['type'].map(current_service2label)
# 构造原始数据
y = train.pop('type')
# 好像有点问题
X = train
train_col = train.columns
X_test = test[train_col]

X, y, X_test = X.values, y, X_test.values

cv_pred = []

skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)  # sklearn divide 10
for index, (train_index, test_index) in enumerate(skf.split(X, y)):  # 后一项为元组

    X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)
    clf = lgb.train(params, train_data, num_boost_round=10000, valid_sets=[validation_data], early_stopping_rounds=50,
                    feval=f1_score, verbose_eval=1)
    print(233)

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
df_test['type'] = submit
df_test['type'] = df_test['type'].map(label2current_service)

df_test.to_csv('submission.csv', index=False)
