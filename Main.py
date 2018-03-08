#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from getDbData import getTrainData

import warnings
import seaborn as sns
pd.set_option('display.large_repr', 'truncate')
pd.set_option('display.max_columns', 0)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.set_printoptions(threshold=5, edgeitems=4)
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
# %matplotlib inline


# 自定义函数：求差分绝对值之和。
def sumdiff(series):
    if len(series) > 1:
        diff_results = np.abs(np.diff(series))
        results = diff_results.sum()
    else:
        results = 0
    return results


if __name__ == '__main__':
    # 设置字体为微软雅黑
    font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc')

    # 读取数据库中的数据
    preprare_data = pd.read_csv("resources/prepare_db_data.csv")  # csv导出读取
    # preprare_data = getTrainData()  # 数据库读取

    # 计算mean， sd 和sumdiff
    print(preprare_data.columns.values)
    feature_src = preprare_data.loc[:, ['storeProdId', 'netSalesUnits', 'weekno']]
    grouped = feature_src['netSalesUnits'].groupby(feature_src['storeProdId'])
    means = grouped.mean()
    sd = grouped.std()
    diffsum_value = []
    index = []

    for grouped_id, group in feature_src.groupby(feature_src['storeProdId']):
        group = group.sort_values(['weekno'])
        # if group.size() > 8:
        # print(grouped_id, sumdiff(group['netSalesUnits']))
        diffsum = sumdiff(group['netSalesUnits'])
        diffsum_value.append(diffsum)
        index.append(grouped_id)

    diffsum = pd.Series(diffsum_value, index=index)
    feature = pd.concat([means, sd, diffsum], axis=1)
    # print(feature.head(5))


    # 根据WEEKNO 和 NET_SALES_UNITS 以外的列去重。
    # print(preprare_data.describe())
    print("preprare_data: shape[0]= %d, shape[1]= %d" % (preprare_data.shape[0], preprare_data.shape[1]))
    preprare_data.set_index(keys='storeProdId', inplace=True)
    columns_list = preprare_data.columns.values.tolist()
    columns_dicts = dict.fromkeys(columns_list)
    columns_dicts.pop("netSalesUnits")
    columns_dicts.pop("weekno")
    # columns_dicts.pop("posId")
    columns = list(i for i in columns_dicts)
    unique_data = preprare_data.drop_duplicates(columns)
    print("unique_data: shape[0]= %d, shape[1]= %d" % (unique_data.shape[0], unique_data.shape[1]))
    # print(unique_data.head(5))
    print("unique_data.columns", unique_data.columns.values)
    # test = unique_data.ix[:, 2:unique_data.shape[1]]
    # print(test.head(5))
    # print("test.columns", test.columns.values)

    # print(preprare_data.drop_duplicates(['Store_Prod_Id']).shape[0])  # 33853


    # 读取csv中的sd mean sumdiff 特征数据 #
    feature = pd.read_csv("resources/features.csv")
    print("feature: shape[0]= %d, shape[1]= %d" % (feature.shape[0], feature.shape[1]))
    # store_prod_id = feature.columns.values[0]
    # print(feature.columns.values)
    # datasets = pd.merge(feature, unique_data, left_on=[store_prod_id], right_on=['Store_Prod_Id'], how='inner')
    # 特征重命名
    # feature.rename(columns={store_prod_id: 'StoreProdId'}, inplace=True)
    # 设置行索引
    feature.set_index(keys='storeProdId', inplace=True)
    # print(feature)

    # 根据肘部图确定cluster的个数
    show = 0
    if show:
        k = range(2, 10)
        clusters = []
        for i in k:
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(feature)
            print("cluster_centers_",  kmeans.cluster_centers_)
            # 样本到所属簇重心的平均距离
            print(kmeans.inertia_)  # 判断簇个数， 越小越好。
            clusters.append(sum(np.min(cdist(feature, kmeans.cluster_centers_, 'euclidean'), axis=1)) / feature.shape[0])
        plt.plot(k, clusters, 'bx-')
        plt.xlabel('k')
        plt.ylabel('平均畸变程度', fontproperties=font)
        # plt.ylabel('平均畸变程度')

        plt.show()

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(feature)
    feature['cluster_temp'] = kmeans.labels_

    remain_sample = feature.copy()
    # print(kmeans.labels_)
    # print(feature)

    # 去除噪声
    remove_cluster = []
    for i in range(6):
        data_cluster_percent = len(feature[feature['cluster_temp'] == i])/feature.shape[0]
        # print(data_cluster_percent)
        if data_cluster_percent < 0.01:
            # print("cluster no.", i)
            remain_sample = remain_sample[remain_sample['cluster_temp'] != i]
            # print(len(remain_sample[remain_sample['cluster'] == i]) / feature.shape[0])
            remove_cluster.append(i)
    print(remove_cluster)  # 输出
    # print(remain_sample.head(5))

    # 重新聚类,只根据sd进行聚类
    # remain_feature = remain_sample['sd']
    # print(remain_feature.head(5))
    cluster_num = 4
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(remain_sample['sd'].values.reshape(-1, 1))
    remain_sample['cluster'] = kmeans.labels_
    # print("kmeans 系数：", kmeans.inertia_)
    # print(kmeans.cluster_centers_)

    # 计算质心, 之后用apply 函数将簇命名， 簇0 对应的中心是不是就是center[0]？
    clusters_meta = zip([x for x in range(0, cluster_num)], kmeans.cluster_centers_)
    for meta in clusters_meta:
        print(meta)

    # print(remain_sample['cluster'].value_counts())

    # 得到聚类中心，根据聚类sd大小排序， 重命名。
    print("*********************************************")
    # print(remain_sample)
    remain_sample.drop(['mean', 'cluster_temp', 'mean', 'diffsum'], axis=1, inplace=True)
    # print(remain_sample)

    # 索引变为列
    remain_sample.reset_index(inplace=True)
    unique_data.reset_index(inplace=True)
    # print(remain_feature.head(5))
    # 数据merge
    # print(feature.columns.values[0])
    datasets = pd.merge(remain_sample, unique_data, on='storeProdId', how="left")  # inner 只有5228. left有 21496
    datasets.set_index(keys='storeProdId', inplace=True)
    # print("merge data shape:", datasets.shape[0], datasets.shape[1])
    # print(datasets.info())  # 查看数据类型，缺失请情况。
    # https://www.cnblogs.com/sirkevin/p/5767532.html
    print("Null value processing...")
    # print(datasets['tradeZone'].isnull())
    datasets['tradeZone'].fillna('OTHER', inplace=True)
    # print(datasets['tradeZone'].isnull())
    # x_feature = ['abbrevOwnerGroupName', 'clcStatus', 'colorMain', 'ctgyPtfm', 'ftwPlatform', 'salesAreaNames', 'tradeZone', 'storeRecordType', 'storeCityTierNumber']
    # x_feature = ['abbrevOwnerGroupName', 'colorMain', 'storeCityTierNumber', 'tradeZone', 'storeType', 'regMsrp', 'gndrGroupNm', 'ctgyPtfm', 'salesAreaNames', 'storeRecordType']
    x_feature = ['abbrevOwnerGroupName', 'storeCityTierNumber', 'tradeZone', 'storeType', 'regMsrp', 'gndrGroupNm', 'ctgyPtfm', 'salesAreaNames']

    # ### 6大属性：性别GNDR_GROUP_NM，
    # ### 价格：REG_MSRP，
    # ### 鞋子的系列Category：Ctgy_Ptfm，
    # ### 颜色：COLOR_MAIN，
    # ### 店铺类型：Store_type，
    # ### 商圈：Trade_Zone，
    # ### 店铺销售区域：Sales.Area.Names
    # print(X.head(5))
    # print(Y.head(5))
    # print(datasets.head(5))
    # print(datasets.columns.values),
    # print(a.shape[0], a.shape[1])
    # print(datasets.info())
    # print(datasets.describe())
    # train_test_split(datasets, )

    label_encoder = LabelEncoder()  # 数值化
    onehot_encoder = OneHotEncoder(sparse=False)  # 不需要稀松矩阵表示。 ont hot编码去除排序
    names = datasets[x_feature].columns

    #################################################################################
    # 价格可视化与分箱操作
    #################################################################################
    # sns.set(palette="muted", color_codes=True)
    # fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # sns.distplot(datasets['regMsrp'], hist=False, color="g", kde_kws={"shade": True}, ax=axes[0])
    # sns.distplot(datasets['regMsrp'], color="m", ax=axes[1])
    # plt.show()
    # group_name = ['Low', 'Middle', 'High', 'Extra']
    # # bins = [0, 750, 1000, 1500, 10000]
    # prices = pd.qcut(datasets['regMsrp'], 4, labels=group_name)
    # datasets['regMsrp'] = prices.astype('object')
    # print(datasets['regMsrp'].head(5))
    # print(datasets['regMsrp'].dtypes)

    for col in names:
        print(datasets[col].value_counts(sort=True))
        # print(datasets[col].dtypes)
        if datasets[col].dtypes == 'object':
            # label encoder编码
            integer_encoded = label_encoder.fit_transform(datasets[col])
            # print(integer_encoded)
            datasets[col] = integer_encoded
            # #one hot编码
            # integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            # onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            # datasets[col] = onehot_encoded
            # print(onehot_encoded)

    # 目标变量分布可视化
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    sns.countplot(x='cluster', data=datasets, ax=axs[0])
    axs[0].set_title("Frequency of each Class")
    datasets['cluster'].value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
    axs[1].set_title("Percentage of each Class")
    plt.show()


    X = datasets[x_feature]
    Y = datasets['cluster']
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    # print(inverted)
    clf = RandomForestClassifier(n_estimators=100, random_state=123)  # 构建分类随机森林分类器 100：0.67， 添加属性可以达到0.7， 另外可以进行调优
    #################################################################################
    # 类别不均衡的处理需要添加
    # 参数调优，提高精度
    # 价格数据的处理， 进行自动分箱，不过精度会降低
    #################################################################################
    clf.fit(X, Y)  # 对自变量和因变量进行拟合
    names, clf.feature_importances_
    feature_importances = zip(names, clf.feature_importances_)
    for feature in feature_importances:
        print(feature)
    y_head = clf.predict(X)
    print("*** test accuracy ***")
    test_accuracy = accuracy_score(Y, y_head)
    print(test_accuracy)
    # scores = cross_val_score(clf, X, Y)
    # print("score:", scores.mean())

    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (12, 6)


    #################################################################################
    # feature importances 可视化 ## 可以换种图形展示。
    #################################################################################
    importances = clf.feature_importances_
    feat_names = names
    indices = np.argsort(importances)[::-1]
    # print(indices) # 索引排序
    fig = plt.figure(figsize=(30, 10))
    plt.title("Feature importances by RandomTreeClassifier")
    plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
    plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
    plt.xticks(range(len(indices)), feat_names[indices], rotation=0, fontsize=10)
    plt.xlim([-1, len(indices)])
    plt.show()

    # 使用线性回归进行变量重要性判断

    # http://blog.csdn.net/niuniuyuh/article/details/77102442
