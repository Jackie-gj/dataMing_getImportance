#!/usr/bin/env python
# encoding: utf-8
from sqlalchemy import Column, create_engine, MetaData, Table
from sqlalchemy import Integer, Float, DateTime, String, Numeric
from db import metadata
from db import create_metadata


metadata = create_metadata()  # THIS IS THE WAY OF CORE.

future = Table("future", metadata,
               Column("id", Integer, primary_key=True),
               Column("name", String(20)),
               Column("category", Integer),
               Column("tableName", String(50)))


FutureCategory = Table("future_category", metadata,
                       Column("id", Integer, primary_key=True),
                       Column("name", String(50)))


FutureCategoryDetail = Table("future_category_detail", metadata,
                             Column("futureCategoryId", Integer, primary_key=True, autoincrement=False),
                             Column("futureId", Integer, primary_key=True, autoincrement=False))

SingleFactorInfo = Table("single_factor_info", metadata,
                         Column("id", Integer, primary_key=True),
                         Column("name", String(100)),
                         Column("tableName", String(100)),
                         Column("vType", Integer),
                         Column("type", Integer),
                         Column("enable", Integer),
                         Column("updateRate", String(10)))


FactorFutureConfig = Table("factor_future_config", metadata,
                           Column("futureId", Integer, primary_key=True, autoincrement=False),
                           Column("singleFactorId", Integer, primary_key=True, autoincrement=False))


Factor = Table("factor", metadata,
               Column("id", Integer, primary_key=True),
               Column("fCate", Integer),
               Column("futureId", Integer),
               Column("name", String(100)),
               Column("modelId", Integer),
               Column("singleFactorId", Integer),
               Column("tableName", String(100)),
               Column("vType", Integer),
               Column("type", Integer))

FactorDetail = Table("factor_detail", metadata,
                     Column("futureFactorId", Integer, primary_key=True, autoincrement=False),
                     Column("factorId", Integer, primary_key=True, autoincrement=False))

Model = Table("model", metadata,
              Column("id", Integer, primary_key=True),
              Column("name", String(100)))

FutureFactor = Table("future_factor", metadata,
                     Column("id", Integer, primary_key=True),
                     Column("futureId", Integer),
                     Column("factorId", Integer),
                     Column("tableNames", String(500)),  # perhaps exceed the limitation
                     Column("modelId", Integer),
                     Column("name", String(100)),
                     Column("version", String(50)),
                     Column("component", String(1000)),
                     Column("fCate", Integer),
                     Column("generateTime", DateTime),
                     Column("onlineTime", DateTime),
                     Column("predict", Integer),
                     Column("predictValue", Float(precision=6)),
                     Column("confidence", Float(precision=3)),
                     Column("accuracy1m", Float(precision=3)),
                     Column("accuracy3m", Float(precision=3)),
                     Column("accuracy6m", Float(precision=3)),
                     Column("accuracy12m", Float(precision=3)),
                     Column("trainInfo", String(1000)),
                     Column("testInfo", String(1000)),
                     Column("loadIndex", Float(precision=3)),
                     Column("corrIndex", Float(precision=3)),
                     Column("corr", String(255)),
                     Column("mi", String(255)),
                     Column("heatMap", String(255)),
                     Column("enable", Integer),
                     Column("status", Integer)
                     )

History = Table("history", metadata,
                Column("id", Integer, primary_key=True),
                Column("futureFactorId", Integer),
                Column("time", DateTime),
                Column("accuracy", Float(precision=3)))

Corr = Table("correlation", metadata,
             Column("id", Integer, primary_key=True),
             Column("futureFactorId", Integer),
             Column("time", DateTime),
             Column("value", Float(precision=3)))

MInfo = Table("mutual_info", metadata,
             Column("id", Integer, primary_key=True),
             Column("futureFactorId", Integer),
             Column("time", DateTime),
             Column("value", Float(precision=3)))

ICIR = Table("ic_ir", metadata,
           Column("id", Integer, primary_key=True),
           Column("time", DateTime),
           Column("futureFactorId", Integer),
           Column("icValue1", Float(precision=3)),
           Column("icValue5", Float(precision=3)),
           Column("icValue10", Float(precision=3)),
           Column("icValue20", Float(precision=3)),
           Column("icValue40", Float(precision=3)),
           Column("icValue60", Float(precision=3)),
           Column("irValue1", Float(precision=3)),
           Column("irValue5", Float(precision=3)),
           Column("irValue10", Float(precision=3)),
           Column("irValue20", Float(precision=3)),
           Column("irValue40", Float(precision=3)),
           Column("irValue60", Float(precision=3)))

Factordata = Table("factor_data", metadata,
                   Column("id", Integer, primary_key=True),
                   Column("time", DateTime),
                   Column("futureFactorId", Integer),
                   Column("updateRate", String(10)),
                   Column("value", Numeric(15, 6)),
                   Column("vType", Integer))

Price = Table("price_data", metadata,
              Column("id", Integer, primary_key=True),
              Column("futureid", Integer),
              Column("price", Float(precision=3)),
              Column("time", DateTime))

TradeDate = Table("trade_date", metadata,
                  Column("id", Integer, primary_key=True),
                  Column("date", DateTime),
                  Column("on", Integer),
                  Column("type", Integer))


