#!/usr/bin/env python
# encoding: utf-8

from sqlalchemy import Column, create_engine, MetaData, Table
from sqlalchemy import Integer, Float, DateTime, String, Numeric
from _interface_mysql.db import Base

class Feature(Base):
    __tablename__ = 'nk_prepared_data'
    Store_Prod_Id = Column(String(50), primary_key=True)
    Weekno = Column(Integer, primary_key=True)
    NET_SALES_UNITS = Column(Integer)
    Prod_id = Column(String(50))
    COLOR_MAIN = Column(String(50))
    GBL_CAT_SUM_LONG_DESC = Column(String(50))
    CTGY_PTFM = Column(String(50))
    GNDR_GROUP_NM = Column(String(50))
    GBL_SILH_LONG_DESC = Column(String(50))
    REG_MSRP = Column(Float)
    PRICE = Column(String(50))
    FTW_PLATFORM = Column(String(50))
    ICON_FRANCHISE = Column(String(50))
    POS_ID = Column(Integer)
    STORE_RECORD_TYPE = Column(String(100))
    SUB_TERRITORY = Column(String(50))
    STORE_ENVIRONMENT_DESCRIPTION = Column(String(50))
    SALES_AREA_NAMES = Column(String(50))
    STORE_CITY_TIER_NUMBER = Column(Integer)
    STORE_LEAD_CATEGORY = Column(String(50))
    ABBREV_OWNER_GROUP_NAME = Column(String(50))
    TRADE_ZONE = Column(String(50))
    STORE_TYPE = Column(String(50))
    CLC_STATUS = Column(String(50))
    QUART = Column(String(50))
    PROD_ENGN_DESC = Column(String(50))

    def __repr__(self):
        return '%s, %s, %s, %s, %s' % (self.__class__.__name__, self.Store_Prod_Id, self.Weekno, self.ABBREV_OWNER_GROUP_NAME, self.GBL_CAT_SUM_LONG_DESC)


class PropertyImportance(Base):
    __tablename__ = 'nk_prop_importance'
    # id = Column(Integer, primary_key=True)
    PROPERTY = Column(String(50), primary_key=True)
    IMPORTANCE = Column(Float)


# select distinct Ctgy_Ptfm  -- Y
# , FTW_PLATFORM  -- Ctgy_Ptfm
# , GBL_CAT_SUM_LONG_DESC  -- Ctgy_Ptfm
# , COLOR_MAIN  -- 颜色 2  Y
# , GNDR_GROUP_NM  -- 性别 3  Y
# , REG_MSRP  -- 售价  Y
# , GBL_SILH_LONG_DESC  -- 三个值  Y
# , store_type -- 大小2个  Y
# , STORE_LEAD_CATEGORY  -- jordan ,basketball other null 共9个值   Y
# , SALES_AREA_NAMES  -- 16个值  Y
# , STORE_ENVIRONMENT_DESCRIPTION  -- 7个， other 和null  Y
# , STORE_CITY_TIER_NUMBER  -- 8个值   Y
# , STORE_RECORD_TYPE  -- 2个属性  Y
# , clc_status  -- 一个属性   Y
# , SUB_TERRITORY -- 3个属性
# , TRADE_ZONE   -- R 第三个模块用到了  Y
# from nk_prepared_data;


