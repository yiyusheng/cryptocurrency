#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Filename: getData.py
#
# Description: 
#
# Copyright (c) 2017, Yusheng Yi <yiyusheng.hust@gmail.com>
#
# Version 1.0
#
# Initial created: 2017-05-11 10:46:11
#
# Last   modified: 2017-12-29 10:34:17
#
#
#

import time
import pymysql
import ast
from urllib2 import Request, urlopen
from datetime import datetime as dt

#%% Connect to MySQL
def dbHandle():
    conn = pymysql.connect(
        host = "localhost",
        user = "root",
        passwd = "qwer1234",
        charset = "utf8",
        use_unicode = True
    )
    return conn

#%% create table
def createTable(cur,dbName,tbName): 
    cur.execute("USE "+dbName)
    cur.execute('''create table if not exists '''+tbName+'''
                (id int(20),
                exchanger varchar(32),
                pair varchar(32),
                price decimal,
                amount decimal,
                type varchar(32),
                time int(20),
                create_time int(20),
                primary key (id,exchanger)
                )''')

#%% drop table
def dropTable(cur,dbName,tbName):
    cur.execute("USE "+dbName)
    cur.execute("drop table "+tbName)
    
#%% get sqlite infomation
def getSqliteInfo(dbName,tbName):
    conn = sqlite3.connect(dir_SQL+dbName+".sqlite")
    cur = conn.cursor()
    a = cur.execute('select count(*) from '+tbName+' group by pair')
    a = a.fetchall()
    su = a[0][0] + a[1][0] + a[2][0] + a[3][0] + a[4][0]
    print('[%s][sqlite-%s]\tbu:%d\tlu:%d\tlb:%d\teu:%d\teb:%d\ttotal:%d' %(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),dbName,a[0][0],a[4][0],a[3][0],a[2][0],a[1][0],su))
    cur.close()
    conn.close()

#%% getTradeHistory per hours
def getBitfinex(pair, cur, dbName, tbName, t, lmt=5000):
    try:
        cur.execute("USE "+dbName)
        url = 'https://api.bitfinex.com/v2/trades/t'+pair+'/hist'
        response = urlopen(Request(url)).read()
        DT = ast.literal_eval(response)
        ct = int(dt.utcnow().strftime("%s"))*1000
        exchanger = 'bitfinex'
        type = 'None'
        
        for h in DT:
            item = list([h[0],exchanger,pair,h[3],h[2],type,h[1],ct])
            cur.execute('insert ignore into '+tbName+' values(%s,%s,%s,%s,%s,%s,%s,%s)',item)
            #cur.execute('insert or ignore into '+tbName+' values(%s,%s,%s,%s,%s,%s,%s,%s)',item)
            cur.commit()
        cur.close()
        conn.close()
        return(1)
    except Exception,e:
        print('[%s]\tException:%s' %(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),e))
        return(0)
        
#%% build connection
dbObject = dbHandle()
cur = dbObject.cursor()
dbName = 'CryptoCurrencyExchanger'
tbName = 'bitfinex'
pair = 'BTCUSD'

#%%

cursor.execute("USE scrapy")
cursor.execute("set names 'utf8';")
cursor.execute("set character set utf8;")
cur_time = datetime.utcnow() + timedelta(hours=8)
utctime_3daysago = datetime.utcnow() +timedelta(days=-3)
time_7daysago = cur_time + timedelta(days=-7)
time_30daysago = cur_time + timedelta(days=-30)
advertiser_extract()
update_ad_for_secondhand()
cursor.close()




my_pair = ['btc_usd','ltc_usd','eth_usd','ltc_btc','eth_btc']
dbName = 'btce'
tbName = 'btce'
dir_SQL = '/home/yiyusheng/Data/Trade_Visualization/'
#dir_SQL = os.path.expanduser('~')+'/Data/Trade_Visualization/'
t = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

#dropSqlite(dbName,tbName)                                           
createSqlite(dbName,tbName)
[getPerhour(p,dbName,tbName,t,5000) for p in my_pair]
getSqliteInfo(dbName,tbName)
