#!/usr/bin/env python
# encoding: utf-8

from _interface_mysql.db import get_session, get_engine, get_connect
from _interface_mysql.tables import *
from sqlalchemy import text


def create_table():
    db.Base.metadata.create_all(get_engine())

#
# def select_by_id(Bean, bean):
#     rs = get_session().query(Bean).get(bean.id)
#     return rs


def select_many(Bean, condition):
    """
        select by condition
        :return
            list of records
    """
    rs = get_session().query(Bean).filter(text(condition)).all()
    return rs


def add(bean):
    get_session().add(bean)
    get_session().commit()


def batch_add(beans):
    get_session().add_all(beans)
    get_session().commit()


def delete(Bean, bean):
    get_session().query(Bean).filter(Bean.id == bean.id).delete()
    get_session().commit()


def query_with_raw_sql(sql):
    """
        execute raw sql
    """
    con = get_connect()
    rs = con.execute(sql)
    return rs


if __name__ == '__main__':
    price = Feature(id=1, name="豆粕", category=0, tableName="price_data")
    # update_by_id(Future, price)
    # price = PriceData(id=1)
    # delete(PriceData, price)
    # rs = select_by_id(Future, price)
    # print type(rs)