#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from getDbData import getTrainData
from matplotlib.font_manager import FontProperties
from _interface_mysql.CRUD import *
from _interface_mysql.db import *


import warnings
import seaborn as sns
pd.set_option('display.large_repr', 'truncate')
pd.set_option('display.max_columns', 0)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
np.set_printoptions(threshold=5, edgeitems=4)
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc' )

# %matplotlib inline

# 定义季节字典，用于提取每一年的数据。
season2number = {"FA16": 1,
                 "HO16": 2,
                 "SP17": 3,
                 "SU17": 4,
                 "FA17": 5,
                 "HO17": 6,
                 "SP18": 7,
                 "SU18": 8,
                 "FA18": 9,
                 "HO18": 10,
                 "SP19": 11,
                 "SU19": 12,
                 "FA19": 13,
                 "HO19": 14,
                 "SP20": 15,
                 "SU20": 16,
                 "FA20": 17,
                 "HO20": 18}


# 自定义函数：求差分绝对值之和。
def sumdiff(series):
    """

    :param series: Series
    :return: 返回差分的绝对值之后
    """

    if len(series) > 1:
        diff_results = np.abs(np.diff(series))
        results = diff_results.sum()
    else:
        results = 0
    return results


def getUniquePrepareData(preprare_data):
    """

    :param preprare_data: 原始样本数据
    :return: 去重后的数据
    """

    # 根据WEEKNO 和 NET_SALES_UNITS 以外的列去重。
    # print(preprare_data.describe())
    # print("preprare_data: shape[0]= %d, shape[1]= %d" % (preprare_data.shape[0], preprare_data.shape[1]))
    preprare_data.set_index(keys='storeProdId', inplace=True)
    columns_list = preprare_data.columns.values.tolist()
    columns_dicts = dict.fromkeys(columns_list)
    columns_dicts.pop("netSalesUnits")
    columns_dicts.pop("weekno")
    # columns_dicts.pop("posId")
    columns = list(i for i in columns_dicts)
    unique_data = preprare_data.drop_duplicates(columns)

    preprare_data = None
    gc.collect()

    return unique_data


def cal_feature(preprare_data):
    """

    :param preprare_data: 原始样本数据
    :return: 计算sd，mean，sumdiff的数据
    """
    # 计算mean， sd 和sumdiff
    # print(preprare_data.columns.values)
    feature_src = preprare_data.loc[:, ['storeProdId', 'netSalesUnits', 'weekno']]
    # print(feature_src.head(100))
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
    feature.columns = ['mean', 'sd', 'diffsum']
    feature = feature.dropna()
    # print(feature.info())

    preprare_data = None
    gc.collect()
    return feature


def remove_noise_data(feature):
    """

    :param feature: 样本数据
    :return: 返回去噪后的样本数据
    """
    # feature.set_index(keys='storeProdId', inplace=True)
    # print(feature)

    # 根据肘部图确定cluster的个数
    show = 0
    if show:
        k = range(2, 10)
        clusters = []
        for i in k:
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(feature )
            print("cluster_centers_", kmeans.cluster_centers_ )
            # 样本到所属簇重心的平均距离
            print(kmeans.inertia_)  # 判断簇个数， 越小越好。
            clusters.append(
                sum(np.min(cdist(feature , kmeans.cluster_centers_, 'euclidean'), axis=1))/feature.shape[0])
        plt.plot(k, clusters, 'bx-')
        plt.xlabel('k')
        # plt.ylabel( '平均畸变程度' , fontproperties=font )
        plt.ylabel('平均畸变程度')

        plt.show()

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(feature)
    feature['cluster_temp'] = kmeans.labels_

    remain_sample = feature.copy()
    # print(kmeans.labels_)
    # print(feature)

    # 去除噪声
    for i in range(6):
        data_cluster_percent = len(feature[feature['cluster_temp'] == i]) / feature.shape[0]
        # print(data_cluster_percent)
        if data_cluster_percent < 0.01:
            # print("cluster no.", i)
            remain_sample = remain_sample[remain_sample['cluster_temp'] != i]

    feature = None
    gc.collect()
    return remain_sample


def clusterData(sample):
    """

    :param sample: 样本数据
    :return: 返回带cluster 标签的样本数据，以及cluster中心值
    """
    # 重新聚类,只根据sd进行聚类
    # remain_feature = remain_sample['sd']
    # print(remain_feature.head(5))
    cluster_num = 4
    kmeans = KMeans(n_clusters=cluster_num, max_iter=2000, random_state=123)
    kmeans.fit(sample['sd'].values.reshape(-1, 1))
    sample['cluster'] = kmeans.labels_
    # print("kmeans 系数：", kmeans.inertia_)
    # print(kmeans.cluster_centers_)

    # 计算质心, 之后用apply 函数将簇命名， 簇0对应的中心就是center[0]？
    clusters_meta = zip([x for x in range(0, cluster_num)], kmeans.cluster_centers_)
    # for meta in clusters_meta:
    #     print ( meta )

    return sample, clusters_meta
    # print(remain_sample['cluster'].value_counts())


def generateTrainData(remain_sample, unique_data):

    """

    :param remain_sample: 特征数据， 包含sd，mean，sumdiff
    :param unique_data: 特征数据， 包含产品属性，店铺属性等
    :return:  返回拼接的数据
    """

    # 得到聚类中心，根据聚类sd大小排序， 重命名。
    # print(remain_sample)
    remain_sample.drop(['mean', 'cluster_temp', 'diffsum'], axis=1, inplace=True )
    # print(remain_sample)

    # 索引变为列
    remain_sample.reset_index (inplace=True )
    unique_data.reset_index ( inplace=True )
    # print(remain_feature.head(5))
    # 数据merge
    # print(feature.columns.values[0])
    datasets = pd.merge(remain_sample, unique_data, on='storeProdId', how="left")  # inner 只有5228. left有 21496
    datasets.set_index(keys='storeProdId', inplace=True)
    print("merge data shape:", datasets.shape[0], datasets.shape[1])
    print(datasets.info())  # 查看数据类型，缺失请情况。
    # print("Null value processing...")
    # print(datasets['tradeZone'].isnull())
    # datasets['tradeZone'].fillna('OTHER', inplace=True)
    # print(datasets['tradeZone'].isnull())
    remain_sample = None
    unique_data = None
    gc.collect()

    return datasets


def getImportance(datasets, feature_list=[]):
    """

    :param datasets: 特征数据， 包含sd，mean，sumdiff, 特征数据， 包含产品属性，店铺属性等
    :param feature_list: 选择的特征列表
    :return:  返回属性重要都得map
    """

    if not feature_list:
        x_feature = ['abbrevOwnerGroupName', 'storeCityTierNumber', 'tradeZone', 'storeType', 'regMsrp', 'gndrGroupNm', 'ctgyPtfm', 'salesAreaNames']
    else:
        x_feature = feature_list
    label_encoder = LabelEncoder()  # 数值化
    # onehot_encoder = OneHotEncoder(sparse=False)  # 不需要稀松矩阵表示。 ont hot编码去除排序
    names = datasets[x_feature].columns

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

    # 目标变量分布可视化， show为0时不显示。
    show = 0
    if show:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        sns.countplot(x='cluster', data=datasets, ax=axs[0])
        axs[0].set_title("Frequency of each Class")
        datasets['cluster'].value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
        axs[1].set_title("Percentage of each Class")
        plt.show()

    X = datasets[x_feature]
    Y = datasets['cluster']
    # 划分训练集与测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    # print(inverted)
    clf = RandomForestClassifier(n_estimators=100, random_state=123)  # 构建分类随机森林分类器 100：0.67， 添加属性可以达到0.7， 另外可以进行调优
    #################################################################################
    # 如果做预测，类别不均衡的处理需要添加,此处不是用于预测。
    # 参数调优，提高精度
    # 价格数据的处理， 进行自动分箱，不过精度会降低
    #################################################################################
    clf.fit(X, Y)  # 对自变量和因变量进行拟合
    feature_importance = {}
    for i in range(len(names)):
        feature_importance[names[i]] = clf.feature_importances_[i]

    print(feature_importance)
    print(clf.feature_importances_.tolist())
    # property = np.array(names)
    feature_importance_df = pd.DataFrame({'property': names, 'importance': clf.feature_importances_.tolist()})

    return feature_importance_df


def getTrainDataByYear(feature_list=[], calculate_season=["FA17"], category="FTW"):
    """

    :param feature_list: 特征列表
    :param calculate_season: 计算的季节与年份的组合如ＦＡ17
    :param category: 类型，APP或者FTW
    :return:datasets: 返回的数据集
    """
    print(calculate_season)
    print(feature_list)
    startTime = time.time()
    print("start time:", startTime)

    global season2number
    number2season = dict(zip(season2number.values(), season2number.keys()))
    print(number2season)
    season_num = getMaxSeason()
    success_try = 0
    count = season_num - 4
    datasets = pd.DataFrame()
    while season_num > count:
        try:
            season = number2season[season_num]
            success_try += 1
            print(season)
        except KeyError:
            continue

        # preprare_data = pd.read_csv("resources/prepare_db_data.csv")  # csv导出读取
        preprare_data = getTrainData(season, category)  # 数据库读取
        # if preprare_data.size() == 0:
        #     continue
        print("preprare_data: shape[0]= %d, shape[1]= %d" % (preprare_data.shape[0], preprare_data.shape[1]))

        # 读取csv中的sd mean sumdiff 特征数据 #
        # feature = pd.read_csv("resources/features.csv")
        feature = cal_feature(preprare_data)
        print("feature: shape[0]= %d, shape[1]= %d" % (feature.shape[0], feature.shape[1]))

        remain_sample = remove_noise_data(feature)
        print("remain_sample: shape[0]= %d, shape[1]= %d" % (remain_sample.shape[0], remain_sample.shape[1]))

        unique_data = getUniquePrepareData(preprare_data)
        print("unique_data: shape[0]= %d, shape[1]= %d" % (unique_data.shape[0], unique_data.shape[1]))

        sub_datasets = generateTrainData(remain_sample, unique_data)
        # # 第一次存入csv需要列名
        # if success_try == 1:
        #     sub_datasets.to_csv("./resources/datasets.csv", index=False, header=True, mode="w")
        # else:
        #     sub_datasets.to_csv("./resources/datasets.csv", index=False, header=False, mode="a")

        datasets = datasets.append(sub_datasets)
        season_num -= 1

        sub_datasets = None
        preprare_data = None
        remain_sample = None
        unique_data = None
        feature = None
        del sub_datasets
        del preprare_data
        del remain_sample
        del unique_data
        del feature
        gc.collect()

    endTime = time.time()
    print("load 4 season data to csv used time: %f" % (endTime - startTime))
    return datasets


def getMaxSeason():
    rs = query_with_raw_sql('select distinct quart from nk_prepared_data')
    season_num = 0
    for item in rs:
        season_temp = season2number[item[0]]
        if season_temp > season_num:
            season_num = season_temp
    print(season_num)
    return season_num


if __name__ == '__main__':
    x_feature = ['ctgyPtfm',  'colorMain', 'gndrGroupNm',
                 'storeCityTierNumber', 'storeType',
                 'price', 'subTerritory', 'clcStatus',
                 'salesAreaNames', 'storeRecordType', 'gblSilhLongDesc',
                 'storeLeadCategory', 'storeEnvironmentDescription', 'tradeZone'
                 ]

    type_list = ["FTW", "APP"]
    first_run = True
    for group in type_list:
        year_datasets = getTrainDataByYear(x_feature, [], group)
        year_datasets.dropna(inplace=True)
        gc.collect()
        # 从csv里头取数据
        # year_datasets = pd.read_csv("./resources/datasets.csv")
        # if len(year_datasets) == 0:
        #     print("no train data to process, exit 1")
        #     exit(0)
        print(year_datasets.shape[0], year_datasets.shape[1])
        print(year_datasets.info())

        # 聚类数据
        total_sample, clusters_meta = clusterData(year_datasets)

        print("****************clusters_meta**************")
        for item in clusters_meta:
            print(item)

        # 样本太多的化采用随机抽样，抽取400000数据进行重要度计算。
        # year_datasets.sample(n=400000, replace=True, random_state=123)
        importances = getImportance(total_sample, x_feature)
        importances['type'] = group
        # print(importances)
        engine = get_engine()
        if first_run:
            # importances.to_sql('nk_prop_importance', engine, if_exists='replace', index=False)
            first_run = False
        else:
            # importances.to_sql('nk_prop_importance', engine, if_exists='append', index=False)
            pass

