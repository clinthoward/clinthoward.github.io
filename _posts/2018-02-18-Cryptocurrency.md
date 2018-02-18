---
layout:     post
title:      "A Primitive CryptoCurrency Trading Strategy"
subtitle:   "We investigate some mildly outdated cryptocurrency data and implement a primitive trading strategy."
date:       2018-02-18 12:00:00
author:     "Clint Howard"
category: Portfolio
tags: [python, data, finance]
comments: True
---
# CryptoCurrency Data Analysis & Primitive Trading

Dataset: https://www.kaggle.com/natehenderson/top-100-cryptocurrency-historical-data

This post is purely for fun/demonstrative purposes... nobody should actually take any investment advice or purchase cryptocurrency based on primitive trading strategies/analysis. 



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime as dt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```


```python
path =r'D:\Downloads\Coding\Datasets\CryptoCurrencyHistoricalData' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, delimiter=";")
    list_.append(df)
frame = pd.concat(list_)
```


```python
close_df = pd.merge(list_[0].iloc[:,[0,4]], list_[1].iloc[:,[0,4]], how = "outer", on="Date")

for i in range(2,len(list_)):
    close_df = pd.merge(close_df, list_[i].iloc[:,[0,4]], how = "outer", on="Date")

names = []
for i in range(0, len(allFiles)):
    names.append(allFiles[i].split(path+"\\")[1].split(".csv")[0])
names.insert(0, "Date")
close_df.columns = names
close_df = close_df.fillna(0)  

close_df.Date[96] = '18/08/2017'
close_df.Date[672] = '18/08/2016'
close_df.Date[827] = '18/08/2015'
close_df.Date[1192] = '18/08/2014'
close_df.Date[1557] = '18/08/2013'

close_df.set_index("Date", inplace=True)
close_df.index = pd.to_datetime(close_df.index, format="%d/%m/%Y")
close_df.sort_index(ascending=True, inplace=True)
```


```python
mcap_df = pd.merge(list_[0].iloc[:,[0,6]], list_[1].iloc[:,[0,6]], how = "outer", on="Date")

for i in range(2,len(list_)):
    mcap_df = pd.merge(mcap_df, list_[i].iloc[:,[0,6]], how = "outer", on="Date")

mcap_df.columns = names

mcap_df.Date[96] = '18/08/2017'
mcap_df.Date[672] = '18/08/2016'
mcap_df.Date[827] = '18/08/2015'
mcap_df.Date[1192] = '18/08/2014'
mcap_df.Date[1557] = '18/08/2013'

mcap_df.set_index("Date", inplace=True)
mcap_df.index = pd.to_datetime(mcap_df.index)
```

Not unexpectedly, the price profiles of most cryptocurrencies are quite similar...virtually nothing for a while and then explosive growth.


```python
close_df.loc[:,np.max(close_df, axis=0) < 200].plot(figsize=(10,10))
plt.legend(loc="center right", bbox_to_anchor=[1.5, 0.5],
           ncol=2,title="CryptoCurrency")
plt.ylabel("Currency Price (/USD)")
plt.show()
```


![png](/img/cryptotrade_6_0.png)



```python
close_df.loc[:,np.max(close_df, axis=0) > 100].plot(figsize=(10,10))
plt.legend(loc="center right", bbox_to_anchor=[1.2, 0.5],
           ncol=1,title="CryptoCurrency")
plt.ylabel("Currency Price (/USD)")
plt.show()
```


![png](/img/cryptotrade_7_0.png)



```python
temp_df = pd.concat([close_df['bitcoin'], close_df.loc[:, close_df.columns != 'bitcoin'].shift(1)])
```

Also not unexpected, we observe some very strong correlation across a large cross-section of the cryptocurrency markets.


```python
corr = temp_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(20,15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask)
plt.show()
```


![png](/img/cryptotrade_10_0.png)


# Basic Trading Strategy
Here we implement a pretty basic trading strategy, where we look at the difference in BTC prices over a certain time-horizon and  either buy/sell depending on the direction. 

The basic premise is:
* Calculate 10d and 40d average BTC price. 
* Choose some target delta between Price_Diff = Price(10d) - Price(40d), and whenever Price_Diff > delta we buy, and when the delta closes we sell. 


```python
btc = pd.DataFrame(close_df['bitcoin'], index=close_df.index)
btc.columns = ["bitcoin"]
btc['10d'] = np.round(btc["bitcoin"].rolling(window=10).mean(),2)
btc['40d'] = np.round(btc["bitcoin"].rolling(window=40).mean(),2)
```


```python
btc[['bitcoin','10d','40d']].plot(grid=True,figsize=(8,5))
plt.show()
```


![png](/img/cryptotrade_13_0.png)


We compare our strategy against a simple buy and hold approach. Naturally given the volatility in bitcoin and the astronomical growth, we'd expect that the buy and hold would be incredibly difficult to beat... especially if we started to take into account transaction costs (and potential FX issues if purchasing via a non-USD currency). 

Given the huge shift in prices of BTC, we also would prefer to have a dynamically adjusting delta... as using a constant one over even a 6-12 month period will likely result in poor outcomes.


```python
btc = btc[btc.index > pd.datetime(2017,1,1)]
btc['10-40'] = btc['10d'] - btc['40d']
X =  50
btc['Stance'] = np.where(btc['10-40'] > X, 1, 0)
btc['Stance'] = np.where(btc['10-40'] < X, -1, btc['Stance'])
btc['Stance'].value_counts()

btc['Stance'].plot(lw=1.5,ylim=[-1.1,1.1])
plt.show()

btc['Market Returns'] = np.log(np.float64(btc['bitcoin'] / btc['bitcoin'].shift(1)))
btc['Strategy'] = btc['Market Returns'] * btc['Stance'].shift(1)

btc[['Market Returns','Strategy']].cumsum().plot(grid=True,figsize=(8,5))

plt.show()


```


![png](/img/cryptotrade_15_0.png)



![png](/img/cryptotrade_15_1.png)


### Optimisation
As above, we see that our strategy substantially underperforms the simple buy and hold strategy. Now for fun, we can generalise our approach and do some basic optimisation/variable mining. 


```python
def annualised_sharpe(returns, N=252):
    return np.sqrt(N) * (returns[~np.isnan(returns)].mean() / returns[~np.isnan(returns)].std())

def bitcoin_sim(btc, shift_time, price_tol, forecast_length):
    btc['diff'] = btc["bitcoin"] - btc["bitcoin"].shift(shift_time)

    btc['Stance'] = np.where(btc['diff'] < -3*price_tol, -1, 0)
    btc['Stance'] = np.where(btc['diff'] > price_tol, 1, btc['Stance'])
   # btc['Stance'].value_counts()

    #btc['Stance'].plot(lw=1.5,ylim=[-1.1,1.1])
    #plt.show()

    btc['Market Returns'] = np.log(np.float64(btc['bitcoin'] / btc['bitcoin'].shift(1)))
    btc['Strategy'] = btc['Market Returns'] * btc['Stance'].shift(forecast_length)

    #btc[['Market Returns','Strategy']].cumsum().plot(grid=True,figsize=(8,5))

    
    return (btc['Strategy'].cumsum().tail(1), annualised_sharpe(btc['Strategy']))
    
```


```python
bitcoin_sim(btc, 1, 1, 1)
```




    (Date
     2017-11-22    0.220979
     Name: Strategy, dtype: float64, 0.24081979662549807)




```python
shift_time = np.linspace(1, 200, 10, dtype=int)
price_tol = np.linspace(-200, 300, 10, dtype=int)
#forecast_length = np.linspace(1, 30, 5, dtype=int)

results_pnl = np.zeros((len(shift_time), len(price_tol)))
results_sharpe = np.zeros((len(shift_time), len(price_tol)))

market_returns = btc['Market Returns'].cumsum().tail(1)
market_sharpe = annualised_sharpe(btc['Market Returns'])


for i, shift_t in enumerate(shift_time):
    for j, price_t in enumerate(price_tol):
        pandl, risk = bitcoin_sim(btc, shift_t, price_t, 1)
        results_pnl[i, j] = pandl - market_returns
        results_sharpe[i, j] = risk - market_sharpe
```


```python
max(results_pnl.flatten())
```




    0.16149115817834314




```python
sns.kdeplot(pd.DataFrame(results_pnl), shade=True)
plt.show()
```


![png](/img/cryptotrade_21_0.png)



```python
sns.kdeplot(pd.DataFrame(results_sharpe), shade=True)
plt.show()
```


![png](/img/cryptotrade_22_0.png)


We can take the results out of the above... and look at what the returns profile looks now. We see that our strategy does quite well early on but has substantially lagged the market. Nonetheless, we have a decent framework for investigating very primitive and basic trading strategies.


```python
price_diff_time = 5
price_tol = 50
forecast_horizon = 2

btc['diff'] = btc["bitcoin"] - btc["bitcoin"].shift(price_diff_time)

btc['Stance'] = np.where(btc['diff'] < -3*price_tol, -1, 0)
btc['Stance'] = np.where(btc['diff'] > price_tol, 1, btc['Stance'])
btc['Stance'].value_counts()

btc['Stance'].plot(lw=1.5,ylim=[-1.1,1.1])
plt.show()

btc['Market Returns'] = np.log(np.float64(btc['bitcoin'] / btc['bitcoin'].shift(1)))
btc['Strategy'] = btc['Market Returns'] * btc['Stance'].shift(forecast_horizon)

btc[['Market Returns','Strategy']].cumsum().plot(grid=True,figsize=(8,5))

plt.show()

```


![png](/img/cryptotrade_24_0.png)



![png](/img/cryptotrade_24_1.png)


A key feature of a trading strategy that needs to be considered is both turnover and the flowon effect into transaction costs. As expected, because we're using a small delta we our executing A LOT of buy/sell orders... and naturally we would expect our underperformance to be even worse.

So in conclusion... most people would have likely been better off just buying and holding BTC as opposed to trying to trade based off of any form of technical analysis.


```python
plt.plot(btc.index, btc['bitcoin'], label="Close Price")
plt.plot(btc.ix[btc.Stance == 1]['bitcoin'].index, btc.ix[btc.Stance == 1]['bitcoin'], '^', markersize=10, color='g')
plt.plot(btc.ix[btc.Stance == -1]['bitcoin'].index, btc.ix[btc.Stance == -1]['bitcoin'], 'v', markersize=10, color='r')
plt.plot(btc.index, btc["diff"], label="pricediff")
plt.show()
```


![png](/img/cryptotrade_26_0.png)


## Extra data to explore


```python
from ggplot import *
```


```python
df = pd.read_csv(r"D:\Downloads\crypto-markets.csv")
df.date = pd.to_datetime(df.date)
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>slug</th>
      <th>symbol</th>
      <th>name</th>
      <th>date</th>
      <th>ranknow</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>market</th>
      <th>close_ratio</th>
      <th>spread</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2013-04-28</td>
      <td>1</td>
      <td>135.30</td>
      <td>135.98</td>
      <td>132.10</td>
      <td>134.21</td>
      <td>0</td>
      <td>1500520000</td>
      <td>0.5438</td>
      <td>3.88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2013-04-29</td>
      <td>1</td>
      <td>134.44</td>
      <td>147.49</td>
      <td>134.00</td>
      <td>144.54</td>
      <td>0</td>
      <td>1491160000</td>
      <td>0.7813</td>
      <td>13.49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2013-04-30</td>
      <td>1</td>
      <td>144.00</td>
      <td>146.93</td>
      <td>134.05</td>
      <td>139.00</td>
      <td>0</td>
      <td>1597780000</td>
      <td>0.3843</td>
      <td>12.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2013-05-01</td>
      <td>1</td>
      <td>139.00</td>
      <td>139.89</td>
      <td>107.72</td>
      <td>116.99</td>
      <td>0</td>
      <td>1542820000</td>
      <td>0.2882</td>
      <td>32.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bitcoin</td>
      <td>BTC</td>
      <td>Bitcoin</td>
      <td>2013-05-02</td>
      <td>1</td>
      <td>116.38</td>
      <td>125.60</td>
      <td>92.28</td>
      <td>105.21</td>
      <td>0</td>
      <td>1292190000</td>
      <td>0.3881</td>
      <td>33.32</td>
    </tr>
  </tbody>
</table>
</div>




```python

ggplot(aes(x='date', y='close', color='name'), data=df[df.ranknow <= 10]) + geom_line() 
```


![png](/img/cryptotrade_31_0.png)





    <ggplot: (166606426937)>




```python
* Shifted correlation i.e. what happens to price of other currencies in days after bitocin has large increases
```
