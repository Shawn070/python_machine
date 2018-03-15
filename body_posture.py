import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer               # 导入预处理模块Imputer
from sklearn.model_selection import train_test_split    # 导入自动生成训练集和测试集的模块train_test_split
from sklearn.metrics import classification_report       # 导入预测结果评估模块classification_report#导入分类器

from sklearn.neighbors import KNeighborsClassifier      # K近邻分类器
from sklearn.tree import DecisionTreeClassifier         # 决策树分类器
from sklearn.naive_bayes import GaussianNB              # 高斯朴素贝叶斯函数

def load_dataset(feature_paths, label_paths):
    '''读取特征文件列表和标签文件列表中的内容，归并后返回
    '''
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0, 1))
    for file in feature_paths:
        # 使用都好分隔符读取特征数据，将问号替换标记为缺失值，文件中不包含表头
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        # 使用平均值补全缺失值，然后将数据进行补全
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)             # 训练预处理器
        df = imp.transform(df)  # 生成预处理结果
        # 将新读入的数据合并到特征集合中
        feature = np.concatenate((feature, df))
    for file in label_paths:
        # 读取标签数据，文件中不包含表头
        df = pd.read_table(file, header=None)
        # 将新读入的数据合并到标签集合中
        label = np.concatenate((label, df))
    # 将标签归整为一维向量
    label = np.ravel(label)
    return feature, label

if __name__ == '__main__':
    # 设置数据路径
    feature_paths = ['A/A.feature', 'A/B.feature', 'A/C.feature', 'A/D.feature', 'A/E.feature']
    label_paths = ['A/A.label', 'A/B.label', 'A/C.label', 'A/D.label', 'A/E.label']

    # 将前4个数据作为训练集读入
    x_train, y_train = load_dataset(feature_paths[:4], label_paths[:4])
    # 将最后一个数据作为测试集读入
    x_test, y_test = load_dataset(feature_paths[4:], label_paths[4:])

    # 使用全量数据作为训练集， 借助train_test_split函数将训练数据打乱
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)

    # 创建k近邻分类器， 并在测试集上进行预测
    print('Start training knn!')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done!')
    answer_knn = knn.predict(x_test)
    print('Prediction done!')

    # 创建决策树分类器，并在测试集上进行预测
    print('Start training DT!')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done!')
    answer_dt = dt.predict(x_test)
    print('Prediction done!')

    # 创建贝叶斯分类器， 并在测试集上进行预测
    print("Start training Bayes!")
    gnb = GaussianNB().fit(x_train, y_train)
    print("Training done!")
    answer_gnb = gnb.predict(x_test)
    print("Prediction done!")

    # 计算准确率与召回率
    print("\n\nThe classification report for knn:")
    print(classification_report(y_test, answer_knn))

    print("\n\nThe classification report for dt:")
    print(classification_report(y_test, answer_dt))

    print("\n\nThe classification report for gnb:")
    print(classification_report(y_test, answer_gnb))

'''
时间：2018年3月15日 21:51:44
Result:

Start training knn!
Training done!
Prediction done!
Start training DT!
Training done!
Prediction done!
Start training Bayes!
Training done!
Prediction done!


The classification report for knn:
             precision    recall  f1-score   support

        0.0       0.56      0.60      0.58    102341
        1.0       0.92      0.93      0.93     23699
        2.0       0.94      0.78      0.85     26864
        3.0       0.83      0.82      0.82     22132
        4.0       0.85      0.88      0.87     32033
        5.0       0.39      0.21      0.27     24646
        6.0       0.77      0.89      0.82     24577
        7.0       0.80      0.95      0.87     26271
       12.0       0.32      0.33      0.33     14281
       13.0       0.16      0.22      0.19     12727
       16.0       0.90      0.67      0.77     24445
       17.0       0.89      0.96      0.92     33034
       24.0       0.00      0.00      0.00      7733

avg / total       0.69      0.69      0.68    374783



The classification report for DT:
             precision    recall  f1-score   support

        0.0       0.50      0.79      0.62    102341
        1.0       0.79      0.96      0.87     23699
        2.0       0.86      0.86      0.86     26864
        3.0       0.94      0.75      0.83     22132
        4.0       0.23      0.16      0.19     32033
        5.0       0.71      0.52      0.60     24646
        6.0       0.77      0.66      0.71     24577
        7.0       0.32      0.15      0.20     26271
       12.0       0.59      0.64      0.61     14281
       13.0       0.63      0.47      0.54     12727
       16.0       0.57      0.07      0.13     24445
       17.0       0.86      0.85      0.86     33034
       24.0       0.40      0.31      0.35      7733

avg / total       0.61      0.61      0.58    374783



The classification report for Bayes:
             precision    recall  f1-score   support

        0.0       0.62      0.81      0.70    102341
        1.0       0.97      0.91      0.94     23699
        2.0       1.00      0.65      0.79     26864
        3.0       0.60      0.66      0.63     22132
        4.0       0.91      0.77      0.83     32033
        5.0       1.00      0.00      0.00     24646
        6.0       0.87      0.72      0.79     24577
        7.0       0.31      0.47      0.37     26271
       12.0       0.52      0.59      0.55     14281
       13.0       0.61      0.50      0.55     12727
       16.0       0.89      0.72      0.79     24445
       17.0       0.75      0.91      0.82     33034
       24.0       0.59      0.24      0.34      7733

avg / total       0.74      0.68      0.67    374783

'''