#!/usr/bin/env python
# encoding: utf-8
import gc

from _interface_mysql.db import *
from _interface_mysql.tables import *
from _interface_mysql.CRUD import *
import pandas as pd


def underline_to_camel(underline_format):

    """
    下划线命名格式驼峰命名格式
    """

    camel_format = ''
    number = 1
    if isinstance(underline_format, str):
        for _s_ in underline_format.split('_'):
            if number == 1:
                camel_format += _s_.lower()
            else:
                camel_format += _s_.capitalize()
            number += 1
    return camel_format


def getTrainData(quart, category):
    engine = get_engine()
    with engine.connect() as con:

        # select * from nk_prepared_data where quart = 'FA17'
        try:
            rs = con.execute("select distinct STORE_PROD_ID"
                             ", WEEKNO"
                             ", NET_SALES_UNITS"
                             ", PROD_ID"
                             ", COLOR_MAIN"
                             ", GBL_CAT_SUM_LONG_DESC"
                             ", Ctgy_Ptfm"
                             ", GNDR_GROUP_NM"
                             ", GBL_SILH_LONG_DESC"
                             ", REG_MSRP"
                             ", PRICE, FTW_PLATFORM"
                             ", ICON_FRANCHISE"
                             ", POS_ID"
                             ", STORE_TYPE"
                             ", ABBREV_OWNER_GROUP_NAME"
                             ", STORE_LEAD_CATEGORY"
                             ", SALES_AREA_NAMES"
                             ", STORE_ENVIRONMENT_DESCRIPTION"
                             ", STORE_CITY_TIER_NUMBER"
                             ", STORE_RECORD_TYPE"
                             ", CLC_STATUS"
                             ", SUB_TERRITORY"
                             ", TRADE_ZONE "
                             "FROM nk_prepared_data "
                             "where quart = '" + quart + "' and PROD_ENGN_DESC = '" + category + "'")
            df = pd.DataFrame(rs.fetchall())
            columns = [underline_to_camel(item) for item in rs.keys()]
            df.columns = columns
        except Exception:
            print("exception in fetch data")
    # print(df.shape)
    # print(df.head(1))
    # print(df.columns.values)
    rs = None
    gc.collect()

    return df


if __name__ == '__main__':
    session = get_session()
    # feature = session.query(Feature).filter(Feature.Store_Prod_Id == '9541_310811-407').all()
    # print('id', feature.Store_Prod_Id)
    # print('color', feature.COLOR_MAIN)
    # lists = session.query(Feature).all()
    # for i in lists:
    #     print(i)
    # feature = session.query(Feature).filter(Feature.QUART == 'FA17').all()
    # feature = session.query(Feature).all()
    # print(type(feature))
    # feature2 = session.query(Feature).all()
    # # print(type(feature2))
    # table_name = "nk_prepared_data"
    # quart_time = "FA17"
    # account_name = "GOODBABY"
    # # 取成年FTW的数据进行分析
    # ftw_datasets = []
    # sql = "select * from " + table_name + " where quart =  '" + quart_time + "' and GNDR_GROUP_NM in('男','女');"
    # records = query_with_raw_sql(sql)
    # # print(type(records))
    # for record in records:
    #     print(list(record))
    #     ftw_datasets.append(list(record))

    # http://shichaoji.com/2016/10/10/database-python-connection-basic/ how to read data from mysql to pandas
    getTrainData("FA17", "FTW")
