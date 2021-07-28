import pandas as pd
import data_organize as do
import numpy as np
import matplotlib.pyplot as plt 
from ReportGenerator import std_series,winsorize_series
import datetime as dt
import talib
global df
plt.style.use({'figure.figsize':(10, 4)})
plt.rcParams['font.family']=['STKaiti']
plt.rcParams['axes.unicode_minus'] = False
from jqdatasdk import *



# future = pd.read_excel('/Users/wdt/Desktop/tpy/Quanti/交易信号/macross_boll_V1/future.xlsx')
# future.index = future.date

#####********* based on WindData

data_oi = do.get_data('all_treasure_futures_oi')
data_oi.index=data_oi.date

global data_oi
global data


data = do.get_data('all_treasure_futures_close')
data.index = data.date
sec_names = data.columns[:-1]
last_dates = getLastDay(data,data_oi)
s = pd.DataFrame([],index=last_dates)
s['name'] = sec_names[::-1][-1]
#### 
df = Merge(data,last_dates)
df_close = df.copy() # 复权前收盘
df = fill_gapday_pct(df, last_dates)
df = cal_backward(df)

data = do.get_data('all_treasure_futures_open')
data.index=data.date
df_open = Merge(data,last_dates) # 复权前开盘价

df['pct_inday'] = df_close/df.open - 1

def getLastDay(data,data_oi):
    """data收盘价数据；data_oi持仓量数据"""
    
    data_oi = data_oi.fillna(0)
    last_dates = []
    k=1
    # 找到每个主力合约的lastday
    for idx in data_oi.index:
        oi_max = max(data_oi.loc[idx,sec_names])
        if oi_max==data_oi.loc[idx,sec_names[-k]]:
            continue
        else:
            print(idx)
            last_dates.append(idx)
            k+=1
            continue
    return last_dates
def Merge(data, last_dates):
    '''
    data包含所有历史合约, 
    通过last_dates提取主连
    '''

    df = pd.DataFrame([],index=data[:last_dates[-1]].index) 
    # 现有新合约需要第二次合并（因为没有lastday）

    l = []
    for i in range(0,len(last_dates)):
        last = last_dates[i]
        sec = sec_names[-i-1]
        idxs=data.loc[:last,sec].index
        l+=(data.loc[:last,sec].dropna().tolist())
        data.drop(idxs,inplace=True,axis=0)
    df['close'] = l
    # 加上现有未跨月的主力合约
    for idx in data.index:
        df.loc[idx,'close'] = data.loc[idx,'T2109.CFE']
    return df
def fill_gapday_pct(df, last_dates):
    '''
    为df计算换月后第一天的真实涨跌幅度
    '''

    data = do.get_data('all_treasure_futures_close')
    data.index=data.date
    # 换新后第一天新合约的收益率（用于前复权）
    # 因为gap会导致计算涨跌幅时出错，因此需要处理换新合约后第一天的pct
    for i in range(0,len(last_dates)):
        last = last_dates[i]
        print(last)
        sec = sec_names[-i-2]
        # break
        pct_fs = data.loc[last:, sec].pct_change().dropna()[0]
        idx_fs = data.loc[last:, sec].pct_change().dropna().index[0]
        df.loc[idx_fs,'pct'] = pct_fs
    return df
def cal_backward(df):
    # 填充收益率
    se_pct = df['close'].pct_change()
    for idx in df.index:
        if np.isnan(df.loc[idx,'pct']):
            df.loc[idx,'pct'] = se_pct[idx]

    # T2109 2021-05-18起
    # 逐个前复权
    tmp = list(df.loc[:'2021-05-18'].index)[::-1]
    for i in range(1,len(tmp)):
        df.loc[tmp[i], 'close'] = df.loc[tmp[i-1],'close'] / \
            (1+df.loc[tmp[i-1],'pct'])
    return df


# 30min data
data = do.get_data('t10_30min')
data.index = data.date
data = data.dropna()

data_2 = pd.DataFrame(index=data.index.unique(),columns=data.windcode.unique())
for c in data_2.columns:
    data_2[c] = data.loc[data.windcode==c,'close']
data_2


# merge
d = pd.DataFrame()
for name in s.name.tolist()[12:]:
    d=d.append(data.loc[data.windcode == name]\
        [:s.loc[s.name==name].index[0].date()+dt.timedelta(1)])
    idxs=(data.loc[data.windcode == name]\
        [:s.loc[s.name==name].index[0].date()+dt.timedelta(1)]).index
    data.drop(idxs, inplace=True)
    # break
 d = d.append(data.loc[data.windcode == 'T2109.CFE'])
# fillgap
for date in s.index[12:]:
    idx = d[date+dt.timedelta(1):].index[0]
    sec = d.loc[idx,'windcode']
    d.loc[ idx, 'pct'] = \
        data_2.loc[:,sec].pct_change()[idx]
    break
# fillpct
for idx in d.index:
    if np.isnan(d.loc[idx,'pct']):
        d.loc[idx,'pct'] = d.close.pct_change()[idx]
# (d.pct+1).cumprod().plot()
# (future.pct+1)['2018-07-23':].cumprod().plot()
# (d.close.pct_change()+1).cumprod().plot()

# 逐个前复权
tmp = d.index[::-1]
d.loc[tmp[0],'close1'] = d.loc[tmp[0],'close']
for i in range(1, len(tmp)):
    # 从当下位置前复权
    d.loc[tmp[i],'close1'] = \
        d.loc[tmp[i-1] ,'close1'] / (1+d.loc[tmp[i-1],'pct'])
d['pct_in'] = (d.close/d.open) - 1
d.to_excel('t10_30min.xlsx')


#####********* based on joinQ

# JQ数据
data_jq2 =get_price('T9999.CCFX', start_date='2015-01-01', end_date='2021-7-22', \
    frequency='daily', fields=None, skip_paused=False, fq='pre').dropna()


# 确定jq数据中每期主力合约的lastday
data_vol = do.get_data('all_treasure_futures_vol')
data_vol.index=data_vol.date
sec_names = data_vol.columns[:-1]
data_vol['vol_'] = data_jq['volume']
for idx in data_vol.index:
    jq_vol = data_vol.loc[idx, 'vol_']
    for c in sec_names:
        if data_vol.loc[idx,c] == jq_vol:
            data_vol.loc[idx,'sec'] = c

last=[]
for name in sec_names:
    last.append (data_vol.loc[data_vol.sec==name].index[-1])
allclose = do.get_data('all_treasure_futures_close')
allclose.index =allclose.date

sec_last = pd.DataFrame(index=sec_names)
sec_last['last_date'] = last

for i in range(len(last)-1):
    l = last[::-1][i]
    sec = sec_names[-i-2]
    
    new_first = allclose[l:].index[1]
    gap_pct= allclose[sec].dropna().pct_change()[new_first]
    data_jq.loc[new_first,'pct'] = gap_pct

# fillpct
for idx in data_jq.index:
    if np.isnan(data_jq.loc[idx,'pct']):
        data_jq.loc[idx,'pct'] = data_jq.close.pct_change()[idx]
tmp = data_jq.index[::-1]
data_jq.loc[tmp[0],'close1'] = data_jq.loc[tmp[0],'close']
for i in range(1, len(tmp)):
    # 从当下位置前复权
    data_jq.loc[tmp[i],'close1'] = \
        data_jq.loc[tmp[i-1] ,'close1'] / (1+data_jq.loc[tmp[i-1],'pct'])
# data_jq['pct_in'] = (data_jq.close/data_jq.open) - 1

daily_prefq = pd.DataFrame(index=data_jq.index)
daily_prefq['close'] = data_jq.close1
daily_prefq['open'] = data_jq.open/data_jq.close * data_jq.close1
daily_prefq['low'] = data_jq.low/data_jq.close * data_jq.close1
daily_prefq['high'] = data_jq.high/data_jq.close * data_jq.close1
daily_prefq['volume'] = data_jq.volume
daily_prefq

#######* 30min
# JQ数据
allclose_min = do.get_data('t10_30min')
allclose_min.index = allclose_min.date
allclose_min = allclose_min.dropna()
data_jq =get_price('T9999.CCFX', start_date='2016-01-01', end_date='2021-7-22', \
    frequency='30m', fields=None, skip_paused=False, fq='pre').dropna()
data_jq['open_pct'] = np.nan
for i in range(len(last)-1):
    l = last[::-1][i]
    if l < data_jq.index[0]:
        continue
    sec = sec_names[-i-2]
    new_first = data_jq[l + dt.timedelta(1):].index[0]
    new_first_day = daily_prefq[l:].index[1]
    # open/较前日
    data_jq.loc[new_first,'open_pct'] = \
        daily_prefq.loc[new_first_day,'open']/daily_prefq['close'].shift(1)[new_first_day]
    data_jq.loc[new_first,'pct'] = \
        data_jq.loc[new_first, 'open_pct']*data_jq.loc[new_first, 'close']/data_jq2.loc[new_first_day, 'open']-1
# fillpct
for idx in data_jq.index:
    if np.isnan(data_jq.loc[idx,'pct']):
        data_jq.loc[idx,'pct'] = data_jq.close.pct_change()[idx]

tmp = data_jq.index[::-1]
data_jq.loc[tmp[0],'close1'] = data_jq.loc[tmp[0],'close']
for i in range(1, len(tmp)):
    # 从当下位置前复权
    data_jq.loc[tmp[i],'close1'] = \
        data_jq.loc[tmp[i-1] ,'close1'] / (1+data_jq.loc[tmp[i-1],'pct'])
# data_jq['pct_in'] = (data_jq.close/data_jq.open) - 1

hfhour_prefq = pd.DataFrame(index= data_jq.index)
hfhour_prefq['close'] = data_jq.close1
hfhour_prefq['open'] = data_jq.open/data_jq.close * data_jq.close1
hfhour_prefq['high'] = data_jq.high/data_jq.close * data_jq.close1
hfhour_prefq['low'] = data_jq.low/data_jq.close * data_jq.close1
hfhour_prefq['volume'] = data_jq.volume

daily_prefq

from sqlalchemy.types import String, Float, Integer,DECIMAL,VARCHAR
from sqlalchemy import DateTime
from sqlalchemy import create_engine
hfhour_prefq['date'] = hfhour_prefq.index
# name = 't10_prefq_daily'
name = 't10_prefq_30min'
columns_type = [DECIMAL(10,4), DECIMAL(10,4), DECIMAL(10,4),\
                DECIMAL(10,4), DECIMAL(10,4), \
                DateTime()]
dtypelist = dict(zip(hfhour_prefq.columns,columns_type))
do.upload_data(hfhour_prefq,name,dtypelist,'replace')

do.get_data(name).close.plot()


