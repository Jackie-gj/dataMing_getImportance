#!/usr/bin/env python
# encoding: utf-8

from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from config.config_default import config_mysql
from sqlalchemy import create_engine
from sqlalchemy import MetaData

Base = declarative_base()  # USE THE WAY OF ORM
engine = None
session = None


def _create_engine():
    global engine
    if engine is None:
        engine = create_engine('mysql://%s:%s@%s/%s?charset=%s'%(config_mysql['user'],
                                                         config_mysql['password'],
                                                         config_mysql['host'],
                                                         config_mysql['database'],
                                                         config_mysql['charset']), echo=True)
    return engine


def get_engine():
    return _create_engine()


def _create_session():
    global session
    if session is None:
        Session = sessionmaker(bind=_create_engine())
        session = Session()
    return session


def get_session():
    return _create_session()


def get_connect():
    return _create_engine().connect()


def create_metadata():
    return MetaData(bind=_create_engine())

