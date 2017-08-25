---
layout:     post
title:      "Analysis of Key Financial Indices, Sectors and Something Extra for Renewable Energy Stocks"
subtitle:   "Various ways to pull in and analyse Index data, Equities and Renewable Energy Data"
date:       2017-06-17 12:00:00
author:     "Clint Howard"
category: Portfolio
tags: [python, data, finance]
comments: True
---
# Analysis of Key Fincial Indices, Sectors and Something Extra for Renewable Energy Stocks

I built out this as my own tool to track various things I'm interested in: indices, FX, rates, commodities and specific equities. It leverages various Quandl datasets, as well as my own database of equities data that I've accumulated from various sources (check [here]( https://clinthoward.github.io/portfolio/2017/04/29/Financial-Scraper/) for set-up of DB). 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl

quandl.ApiConfig.api_key = "insert api key here"
```

## Fundamentals Data
This gives me a snapshot of key metrics I'm interested in across various things (DJIA, FX, Bitcoin, Iron Ore etc) via some basis charts, and then some returns analysis. Most data is sourced from various free Quandl databases.



```python
curr_date = pd.datetime(2017,6,17)
start_date = curr_date - pd.DateOffset(years=1)
save_path = "C:\\Users\\Clint_PC\\Google Drive\\General Learning\\Python\\Learning Plan Code\\Finance\\Results Tables\\" 
```


```python
#[SP500, SP500 VIX, DJIA_VIX, NASDAQ, DAX_future, FTSE100, Nikkei, Shanghai ]
index_quandlcodes = ['CHRIS/CME_SP1.4',
                     'CBOE/VIX.2', 
                     'CBOE/VXD.2',
                     'NASDAQOMX/COMP.1',
                     'CHRIS/EUREX_FDAX1.2', 
                     'CHRIS/LIFFE_Z1.2',
                     'NIKKEI/INDEX.4'] 
#[AUD, EUR, GBP, JPY]
fx_quandlcodes = ['FRED/DEXUSAL',
                 'FRED/DEXUSEU',
                 'FRED/DEXUSUK',
                 'FRED/DEXJPUS']

#[Ave Price, Num Transactions, Market Cap, Total Bitcoin]
bitcoin_quandlcodes = ['BCHARTS/BITSTAMPUSD.7',
                      'BCHAIN/NTRAN',
                      'BCHAIN/MKTCP',
                      'BCHAIN/TOTBC']
#[Crude, Gold, Iron Ore (NYMEX traded 62% Fe, CFR China in $US/metric tonne)]
commodities_quandlcodes = ['CHRIS/CME_CL1.4',
                          'LBMA/GOLD.2',
                          'COM/FE_TJN']

#[USD Long Term, USD10y, ON USDLibor, 1y USD T Bill]
rates_quandlcodes = ['USTREASURY/LONGTERMRATES.1',
                    'FRED/DGS10',
                    'FRED/DTB1YR']

#[CDX NA Inv. Grade, CDX NA High Yield]
credit_quandlcodes = ['COM/CDXNAIG',
                     'COM/CDXNAHY']
#aud_quandlcodes = ['RBA/H01.1',
#                  'RBA/G01.1']



all_codes = index_quandlcodes + fx_quandlcodes + bitcoin_quandlcodes + commodities_quandlcodes + rates_quandlcodes + credit_quandlcodes 

equities = ['test', 'test2', 'test3']
renewable_energy_stocks = ['test4', 'test5', 'test6']
```


```python
all_codes
```




    ['CHRIS/CME_SP1.4',
     'CBOE/VIX.2',
     'CBOE/VXD.2',
     'NASDAQOMX/COMP.1',
     'CHRIS/EUREX_FDAX1.2',
     'CHRIS/LIFFE_Z1.2',
     'NIKKEI/INDEX.4',
     'FRED/DEXUSAL',
     'FRED/DEXUSEU',
     'FRED/DEXUSUK',
     'FRED/DEXJPUS',
     'BCHARTS/BITSTAMPUSD.7',
     'BCHAIN/NTRAN',
     'BCHAIN/MKTCP',
     'BCHAIN/TOTBC',
     'CHRIS/CME_CL1.4',
     'LBMA/GOLD.2',
     'COM/FE_TJN',
     'USTREASURY/LONGTERMRATES.1',
     'FRED/DGS10',
     'FRED/DTB1YR',
     'COM/CDXNAIG',
     'COM/CDXNAHY']




```python
vals = quandl.get(all_codes, start_date=start_date, end_date = curr_date)
```

<details>
  <summary>Click to expand</summary><p>
```python
fig, ax = plt.subplots(nrows=4, ncols=2, sharex=False, sharey=False, figsize = (15,20))
fig.suptitle('Indices', fontsize=20, fontweight='bold')

ax1 = plt.subplot(421)
line1 = plt.plot(vals.iloc[:,0].dropna(), label = 'Value')
line2 = plt.plot(vals.iloc[:,0].dropna().rolling(30).mean(), label = '30D Rolling Average')

ax2 = ax1.twinx()
line3 = ax2.plot(vals.iloc[:,0].dropna().rolling(30).std(), 'xr', label = '30D Rolling Stdev')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
lns = line1+line2+line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
plt.title('S&P500')

plt.subplot(4,2,2)
plt.plot(vals.iloc[:,1].dropna())
plt.plot(vals.iloc[:,1].dropna().rolling(30).mean())
plt.title('VIX')

plt.subplot(4,2,3)
plt.plot(vals.iloc[:,2].dropna())
plt.plot(vals.iloc[:,2].dropna().rolling(30).mean())
plt.title('VXD')

plt.subplot(4,2,4)
plt.plot(vals.iloc[:,3].dropna())
plt.plot(vals.iloc[:,3].dropna().rolling(30).mean())
plt.title('NASDAQ')

plt.subplot(4,2,5)
plt.plot(vals.iloc[:,4].dropna())
plt.plot(vals.iloc[:,4].dropna().rolling(30).mean())
plt.title('DAX')

plt.subplot(4,2,6)
plt.plot(vals.iloc[:,5].dropna())
plt.plot(vals.iloc[:,5].dropna().rolling(30).mean())
plt.title('FTSE100')

plt.subplot(4,2,7)
plt.plot(vals.iloc[:,6].dropna())
plt.plot(vals.iloc[:,6].dropna().rolling(30).mean())
plt.title('NIKKEI225')

fig.delaxes(ax.flatten()[7])

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

```
</p></details>



![png](/img/findashboard_7_0.png)


<details>
  <summary>Click to expand</summary><p>
```python
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize = (10,15))
fig.suptitle('Credit', fontsize=20, fontweight='bold')

plt.subplot(211)
plt.plot(vals.iloc[:,21].dropna())
plt.plot(vals.iloc[:,21].dropna().rolling(30).mean())
plt.title('CDX NA Inv. Grade')

plt.subplot(212)
plt.plot(vals.iloc[:,22].dropna())
plt.plot(vals.iloc[:,22].dropna().rolling(30).mean())
plt.title('CDX NA High Yield')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
```
</p></details>



![png](/img/findashboard_8_0.png)

<details>
  <summary>Click to expand</summary><p>
```python
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize = (10,15))
fig.suptitle('Rates', fontsize=20, fontweight='bold')

plt.subplot(2,2,1)
plt.plot(vals.iloc[:,18].dropna())
plt.plot(vals.iloc[:,18].dropna().rolling(30).mean())
plt.title('USD Long Term Rates')

plt.subplot(2,2,2)
plt.plot(vals.iloc[:,19].dropna())
plt.plot(vals.iloc[:,19].dropna().rolling(30).mean())
plt.title('USD 10Y Bond Yield')

plt.subplot(2,2,3)
plt.plot(vals.iloc[:,20].dropna())
plt.plot(vals.iloc[:,20].dropna().rolling(30).mean())
plt.title('USD 1Y T-Bill')

fig.delaxes(ax.flatten()[3])

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
```
</p></details>




![png](/img/findashboard_9_0.png)


<details>
  <summary>Click to expand</summary><p>
```python
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize = (10,15))
fig.suptitle('Commodities', fontsize=20, fontweight='bold')

plt.subplot(2,2,1)
plt.plot(vals.iloc[:,15].dropna())
plt.plot(vals.iloc[:,15].dropna().rolling(30).mean())
plt.title('NYMEX Oil')

plt.subplot(2,2,2)
plt.plot(vals.iloc[:,16].dropna())
plt.plot(vals.iloc[:,16].dropna().rolling(30).mean())
plt.title('Gold')

plt.subplot(2,2,3)
plt.plot(vals.iloc[:,17].dropna())
plt.plot(vals.iloc[:,17].dropna().rolling(30).mean())
plt.title('Iron Ore')

fig.delaxes(ax.flatten()[3])

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
```
</p></details>

![png](/img/findashboard_10_0.png)

<details>
  <summary>Click to expand</summary><p>
```python
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize = (10,15))
fig.suptitle('BTC', fontsize=20, fontweight='bold')

plt.subplot(2,2,1)
plt.plot(vals.iloc[:,11].dropna())
plt.plot(vals.iloc[:,11].dropna().rolling(30).mean())
plt.title('BTC Average')

plt.subplot(2,2,2)
plt.plot(vals.iloc[:,12].dropna())
plt.plot(vals.iloc[:,12].dropna().rolling(30).mean())
plt.title('BTC # Transactions')

plt.subplot(2,2,3)
plt.plot(vals.iloc[:,13].dropna())
plt.plot(vals.iloc[:,13].dropna().rolling(30).mean())
plt.title('BTC Market Cap.')

plt.subplot(2,2,4)
plt.plot(vals.iloc[:,14].dropna())
plt.plot(vals.iloc[:,14].dropna().rolling(30).mean())
plt.title('Total BTC')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
```
</p></details>




![png](/img/findashboard_11_0.png)

<details>
  <summary>Click to expand</summary><p>
```python
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize = (10,15))
fig.suptitle('FX', fontsize=20, fontweight='bold')

plt.subplot(2,2,1)
plt.plot(vals.iloc[:,7].dropna())
plt.plot(vals.iloc[:,7].dropna().rolling(30).mean())
plt.title('AUDUSD')

plt.subplot(2,2,2)
plt.plot(vals.iloc[:,8].dropna())
plt.plot(vals.iloc[:,8].dropna().rolling(30).mean())
plt.title('EURUSD')

plt.subplot(2,2,3)
plt.plot(vals.iloc[:,9].dropna())
plt.plot(vals.iloc[:,9].dropna().rolling(30).mean())
plt.title('GBPUSD')

plt.subplot(2,2,4)
plt.plot(vals.iloc[:,10].dropna())
plt.plot(vals.iloc[:,10].dropna().rolling(30).mean())
plt.title('JPYUSD')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

fig.savefig(save_path+"FX_"+str(curr_date)[0:10])
```
</p></details>




![png](/img/findashboard_12_0.png)



```python
results = pd.DataFrame(index=vals.columns, columns = ['Description','Data_Date','Spot', 'Daily', 'WTD', 'MTD', 'YTD', 'YoY'])
results['Description'] = ['S&P500 Futures',
                         'S&P500 VIX', 
                         'DJIA VXD',
                         'NASDAQ',
                         'DAX',
                         'FTSE100', 
                         'NIKKEI225',
                         'AUDUSD',
                         'EURUSD',
                         'GBPUSD',
                         'JPYUSD',
                         'Average BTC Price',
                         'BTC # Transactions',
                         'BTC Market Cap.',
                         'Total BTC',
                         'NYMEX Crude Oil',
                         'Gold',
                         'Iron Ore 62%',
                         'USD Long Term Rate',
                         'USD 10Y Bond Yield',
                         'USD 1Y T-Bill',
                         'CDX NA Investment Grade',
                         'CDX NA High Yield']

for col in vals.columns:
    prices = vals[col]
    prices = prices.dropna()
    
    last_date = prices.last_valid_index()
    prior_date = last_date - pd.tseries.offsets.Day(days=1)
    
    start = prices.index[0]
    
    week = last_date - pd.tseries.offsets.Week(weekday=0)
    week = prices.index.get_loc(week,method='nearest')
    
    month = last_date - pd.tseries.offsets.BMonthBegin()
    month = prices.index.get_loc(month,method='nearest')
    
    year = last_date - pd.tseries.offsets.BYearBegin()
    year = prices.index.get_loc(year,method='nearest')

    close = prices[last_date]
    daily = round((close - prices[prior_date])/prices[prior_date]*100, 3)
    wtd = round((close - prices[week])/prices[week]*100, 3)
    mtd = round((close - prices[month])/prices[month]*100, 3)
    ytd = round((close - prices[year])/prices[year]*100, 3)
    yoy = round((close-prices[start])/prices[start]*100, 3)
    
    results.loc[col, ['Data_Date', 'Spot', 'Daily', 'WTD', 'MTD', 'YTD', 'YoY']] = [last_date, close, daily, wtd, mtd, ytd, yoy ]
    
```


```python
results
```




<div>
<font size = "2">
<table border="1" class="dataframe" width="300">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Description</th>
      <th>Data_Date</th>
      <th>Spot</th>
      <th>Daily</th>
      <th>WTD</th>
      <th>MTD</th>
      <th>YTD</th>
      <th>YoY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CHRIS/CME_SP1 - Last</th>
      <td>S&amp;P500 Futures</td>
      <td>2017-06-15 00:00:00</td>
      <td>2434.3</td>
      <td>-0.123</td>
      <td>0.231</td>
      <td>0.177</td>
      <td>8.061</td>
      <td>17.083</td>
    </tr>
    <tr>
      <th>CBOE/VIX - VIX High</th>
      <td>S&amp;P500 VIX</td>
      <td>2017-06-16 00:00:00</td>
      <td>11.35</td>
      <td>-5.495</td>
      <td>-8.246</td>
      <td>7.685</td>
      <td>-19.332</td>
      <td>-43.335</td>
    </tr>
    <tr>
      <th>CBOE/VXD - High</th>
      <td>DJIA VXD</td>
      <td>2017-06-16 00:00:00</td>
      <td>10.69</td>
      <td>-4.383</td>
      <td>-22.48</td>
      <td>-5.23</td>
      <td>-22.983</td>
      <td>-41.264</td>
    </tr>
    <tr>
      <th>NASDAQOMX/COMP - Index Value</th>
      <td>NASDAQ</td>
      <td>2017-06-16 00:00:00</td>
      <td>6151.76</td>
      <td>-0.223</td>
      <td>-0.384</td>
      <td>-1.522</td>
      <td>13.311</td>
      <td>28.153</td>
    </tr>
    <tr>
      <th>CHRIS/EUREX_FDAX1 - High</th>
      <td>DAX</td>
      <td>2017-06-16 00:00:00</td>
      <td>12765</td>
      <td>-0.375</td>
      <td>-0.153</td>
      <td>0.54</td>
      <td>9.83</td>
      <td>31.429</td>
    </tr>
    <tr>
      <th>CHRIS/LIFFE_Z1 - High</th>
      <td>FTSE100</td>
      <td>2017-06-16 00:00:00</td>
      <td>7489</td>
      <td>0.053</td>
      <td>-0.676</td>
      <td>-1.005</td>
      <td>4.829</td>
      <td>24.858</td>
    </tr>
    <tr>
      <th>NIKKEI/INDEX - Close Price</th>
      <td>NIKKEI225</td>
      <td>2017-06-16 00:00:00</td>
      <td>19943.3</td>
      <td>0.562</td>
      <td>0.174</td>
      <td>0.419</td>
      <td>1.782</td>
      <td>27.844</td>
    </tr>
    <tr>
      <th>FRED/DEXUSAL - Value</th>
      <td>AUDUSD</td>
      <td>2017-06-09 00:00:00</td>
      <td>0.7532</td>
      <td>-0.146</td>
      <td>0.709</td>
      <td>1.963</td>
      <td>4.163</td>
      <td>1.991</td>
    </tr>
    <tr>
      <th>FRED/DEXUSEU - Value</th>
      <td>EURUSD</td>
      <td>2017-06-09 00:00:00</td>
      <td>1.119</td>
      <td>-0.241</td>
      <td>-0.533</td>
      <td>-0.214</td>
      <td>7.431</td>
      <td>-0.586</td>
    </tr>
    <tr>
      <th>FRED/DEXUSUK - Value</th>
      <td>GBPUSD</td>
      <td>2017-06-09 00:00:00</td>
      <td>1.2737</td>
      <td>-1.561</td>
      <td>-1.394</td>
      <td>-1.218</td>
      <td>3.925</td>
      <td>-10.837</td>
    </tr>
    <tr>
      <th>FRED/DEXJPUS - Value</th>
      <td>JPYUSD</td>
      <td>2017-06-09 00:00:00</td>
      <td>110.61</td>
      <td>0.463</td>
      <td>0.109</td>
      <td>-0.566</td>
      <td>-6.008</td>
      <td>6.152</td>
    </tr>
    <tr>
      <th>BCHARTS/BITSTAMPUSD - Weighted Price</th>
      <td>Average BTC Price</td>
      <td>2017-06-16 00:00:00</td>
      <td>2450.92</td>
      <td>6.028</td>
      <td>-9.918</td>
      <td>2.524</td>
      <td>141.946</td>
      <td>231.684</td>
    </tr>
    <tr>
      <th>BCHAIN/NTRAN - Value</th>
      <td>BTC # Transactions</td>
      <td>2017-06-17 00:00:00</td>
      <td>269937</td>
      <td>-7.916</td>
      <td>21.107</td>
      <td>-16.073</td>
      <td>49.548</td>
      <td>0.772</td>
    </tr>
    <tr>
      <th>BCHAIN/MKTCP - Value</th>
      <td>BTC Market Cap.</td>
      <td>2017-06-17 00:00:00</td>
      <td>4.04131e+10</td>
      <td>0.932</td>
      <td>-16.729</td>
      <td>8.036</td>
      <td>150.315</td>
      <td>235.993</td>
    </tr>
    <tr>
      <th>BCHAIN/TOTBC - Value</th>
      <td>Total BTC</td>
      <td>2017-06-17 00:00:00</td>
      <td>1.6395e+07</td>
      <td>0.011</td>
      <td>0.056</td>
      <td>0.19</td>
      <td>1.976</td>
      <td>4.656</td>
    </tr>
    <tr>
      <th>CHRIS/CME_CL1 - Last</th>
      <td>NYMEX Crude Oil</td>
      <td>2017-06-16 00:00:00</td>
      <td>44.68</td>
      <td>0.995</td>
      <td>-2.87</td>
      <td>-6.975</td>
      <td>-14.847</td>
      <td>-7.418</td>
    </tr>
    <tr>
      <th>LBMA/GOLD - USD (PM)</th>
      <td>Gold</td>
      <td>2017-06-16 00:00:00</td>
      <td>1255.4</td>
      <td>0.068</td>
      <td>-0.869</td>
      <td>-0.747</td>
      <td>9.07</td>
      <td>-2.735</td>
    </tr>
    <tr>
      <th>COM/FE_TJN - Column 1</th>
      <td>Iron Ore 62%</td>
      <td>2017-06-15 00:00:00</td>
      <td>54.51</td>
      <td>0.018</td>
      <td>-0.475</td>
      <td>-10.212</td>
      <td>-26.989</td>
      <td>7.834</td>
    </tr>
    <tr>
      <th>USTREASURY/LONGTERMRATES - LT Composite &gt; 10 Yrs</th>
      <td>USD Long Term Rate</td>
      <td>2017-06-16 00:00:00</td>
      <td>2.62</td>
      <td>0</td>
      <td>-2.602</td>
      <td>-3.321</td>
      <td>-9.343</td>
      <td>22.43</td>
    </tr>
    <tr>
      <th>FRED/DGS10 - Value</th>
      <td>USD 10Y Bond Yield</td>
      <td>2017-06-14 00:00:00</td>
      <td>2.15</td>
      <td>-2.715</td>
      <td>-2.715</td>
      <td>-2.715</td>
      <td>-12.245</td>
      <td>32.716</td>
    </tr>
    <tr>
      <th>FRED/DTB1YR - Value</th>
      <td>USD 1Y T-Bill</td>
      <td>2017-06-14 00:00:00</td>
      <td>1.17</td>
      <td>-0.847</td>
      <td>0.862</td>
      <td>3.54</td>
      <td>34.483</td>
      <td>138.776</td>
    </tr>
    <tr>
      <th>COM/CDXNAIG - value</th>
      <td>CDX NA Investment Grade</td>
      <td>2017-06-15 00:00:00</td>
      <td>60.22</td>
      <td>2.467</td>
      <td>0.333</td>
      <td>-0.446</td>
      <td>-11.141</td>
      <td>-27.253</td>
    </tr>
    <tr>
      <th>COM/CDXNAHY - value</th>
      <td>CDX NA High Yield</td>
      <td>2017-06-15 00:00:00</td>
      <td>107.48</td>
      <td>-0.223</td>
      <td>-0.204</td>
      <td>-0.232</td>
      <td>1.234</td>
      <td>5.342</td>
    </tr>
  </tbody>
</table>
</font>
</div>




```python
pathres_to_save = save_path + str(curr_date)[0:10] + "_agg.csv"
pathdata_to_save = save_path + str(curr_date)[0:10] + "_data.csv"
results.to_csv(pathres_to_save)
vals.to_csv(pathdata_to_save)
```

## Sector Trends
This section deals with some equities analysis. I leverage off my databse of equities to perform some high level sector analysis, this gives me a quick overview of what various industries have been doing relative to each other and I can then deep-dive into anything that piques my interest. 


```python
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import pandas.io.sql as psql

db_host = 'localhost'
db_user = 'username'
db_pass = 'password'
db_name = 'pricing'

con = pymysql.connect(db_host, db_user, db_pass, db_name)

sql_usd = """SELECT *
         FROM symbol AS sym
         INNER JOIN daily_price AS dp
         ON dp.symbol_id = sym.id
         where price_date > '2016-01-01' AND currency = 'USD'
         ORDER BY dp.price_date ASC
         ;"""
         
sql_aud = """SELECT *
         FROM symbol AS sym
         INNER JOIN daily_price AS dp
         ON dp.symbol_id = sym.id
         where price_date > '2016-01-01' AND currency = 'AUD'
         ORDER BY dp.price_date ASC
         ;"""

engine = create_engine('mysql+pymysql://username:password@localhost:3306/pricing')

# Create a pandas dataframe from the SQL query
with engine.connect() as conn, conn.begin():
    p_usd_data = pd.read_sql(sql_usd, con)
    p_aud_data = pd.read_sql(sql_aud, con)
```

### AUD Sectors
Here I leverage my AUD dataset to look at stock prices. I adjust based on volume... I would have preferred to have done it by Market Cap but haven't got around to incorporating that into my scraper yet.


```python
p_aud_data['volume_traded'] = p_aud_data.adj_close_price*p_aud_data.volume
total_marketcap = p_aud_data.groupby(['price_date', 'sector'])['volume_traded'].sum()
total_marketcap = total_marketcap.reset_index()  
total_marketcap.columns = ['price_date', 'sector', 'total_volume_traded']
merged_aud_data = p_aud_data.merge(total_marketcap)
merged_aud_data['weighted_price'] =  merged_aud_data['adj_close_price']*merged_aud_data['volume_traded']/merged_aud_data['total_volume_traded']
```


```python
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize = (10,8))
fig.suptitle('Unadjusted v Adjusted Volume Weighted Stock Prices', fontsize=20, fontweight='bold')

unadj_sectprices = p_aud_data.groupby(['price_date', 'sector'])['adj_close_price'].mean()
unadj_sectprices = unadj_sectprices.unstack()
unadj_sectprices.plot(ax=ax[0], legend = False)
ax[0].set_ylabel('Unadjusted Price ($)')

adj_sectorprices = merged_aud_data.groupby(['price_date', 'sector'])['weighted_price'].sum()
adj_sectorprices= adj_sectorprices.unstack()
adj_sectorprices = adj_sectorprices.dropna()

adj_sectorprices.plot(ax=ax[1])
ax[1].set_ylabel('Volume Weighted Price ($)')
plt.legend(bbox_to_anchor=(1.1, 1.5), prop={'size':12})
plt.show()
```


![png](/img/findashboard_20_0.png)



```python
results_audsectors = pd.DataFrame(index=unadj_sectprices.columns, columns = ['Data_Date','Spot', 'Daily', 'WTD', 'MTD', 'YTD', 'YoY'])


for col in unadj_sectprices.columns:
    prices = unadj_sectprices[col]
    prices = prices.dropna()
    
    last_date = prices.last_valid_index()
    prior_date = last_date - pd.tseries.offsets.Day(days=2)
    
    start = prices.index[0]
    
    week = last_date - pd.tseries.offsets.Week(weekday=0)
    week = prices.index.get_loc(week,method='nearest')
    
    month = last_date - pd.tseries.offsets.BMonthBegin()
    month = prices.index.get_loc(month,method='nearest')
    
    year = last_date - pd.tseries.offsets.BYearBegin()
    year = prices.index.get_loc(year,method='nearest')

    close = prices[last_date]
    daily = round((close - prices[prior_date])/prices[prior_date]*100, 3)
    wtd = round((close - prices[week])/prices[week]*100, 3)
    mtd = round((close - prices[month])/prices[month]*100, 3)
    ytd = round((close - prices[year])/prices[year]*100, 3)
    yoy = round((close-prices[start])/prices[start]*100, 3)
    
    results_audsectors.loc[col, ['Data_Date', 'Spot', 'Daily', 'WTD', 'MTD', 'YTD', 'YoY']] = [last_date, close, daily, wtd, mtd, ytd, yoy ]
    
```


```python
results_audsectors
```




<div>
<font size="2">
<table border="1" class="dataframe" width="300">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Data_Date</th>
      <th>Spot</th>
      <th>Daily</th>
      <th>WTD</th>
      <th>MTD</th>
      <th>YTD</th>
      <th>YoY</th>
    </tr>
    <tr>
      <th>sector</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Consumer Discretionary</th>
      <td>2017-06-09 00:00:00</td>
      <td>11.2737</td>
      <td>-0.144</td>
      <td>-5.988</td>
      <td>-1.477</td>
      <td>3.026</td>
      <td>15.742</td>
    </tr>
    <tr>
      <th>Consumer Staples</th>
      <td>2017-06-09 00:00:00</td>
      <td>13.2524</td>
      <td>-0.561</td>
      <td>18.072</td>
      <td>-5.876</td>
      <td>-2.883</td>
      <td>-32.342</td>
    </tr>
    <tr>
      <th>Energy</th>
      <td>2017-06-09 00:00:00</td>
      <td>11.8119</td>
      <td>-1.568</td>
      <td>-5.164</td>
      <td>-4.309</td>
      <td>3.019</td>
      <td>11.897</td>
    </tr>
    <tr>
      <th>Financials</th>
      <td>2017-06-09 00:00:00</td>
      <td>21.5229</td>
      <td>-0.089</td>
      <td>10.438</td>
      <td>-0.899</td>
      <td>1.907</td>
      <td>6.242</td>
    </tr>
    <tr>
      <th>Health Care</th>
      <td>2017-06-09 00:00:00</td>
      <td>26.9329</td>
      <td>-0.107</td>
      <td>-1.552</td>
      <td>0.267</td>
      <td>22.376</td>
      <td>25.649</td>
    </tr>
    <tr>
      <th>Industrials</th>
      <td>2017-06-09 00:00:00</td>
      <td>8.96132</td>
      <td>0.283</td>
      <td>-8.559</td>
      <td>-0.812</td>
      <td>13.241</td>
      <td>35.912</td>
    </tr>
    <tr>
      <th>Information Technology</th>
      <td>2017-06-09 00:00:00</td>
      <td>7.3155</td>
      <td>-0.184</td>
      <td>-2.408</td>
      <td>-0.469</td>
      <td>1.463</td>
      <td>15.954</td>
    </tr>
    <tr>
      <th>Materials</th>
      <td>2017-06-09 00:00:00</td>
      <td>9.33486</td>
      <td>0.891</td>
      <td>3.22</td>
      <td>0.913</td>
      <td>2.208</td>
      <td>45.456</td>
    </tr>
    <tr>
      <th>Real Estate</th>
      <td>2017-06-09 00:00:00</td>
      <td>6.50386</td>
      <td>-0.369</td>
      <td>-3.879</td>
      <td>-3.604</td>
      <td>2.136</td>
      <td>55.885</td>
    </tr>
    <tr>
      <th>Telecommunication Services</th>
      <td>2017-06-09 00:00:00</td>
      <td>4.3025</td>
      <td>0.702</td>
      <td>6.147</td>
      <td>3.55</td>
      <td>-9.563</td>
      <td>-29.031</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>2017-06-09 00:00:00</td>
      <td>7.982</td>
      <td>0.365</td>
      <td>-2.978</td>
      <td>-5.292</td>
      <td>24.915</td>
      <td>56.627</td>
    </tr>
  </tbody>
</table>
</font>
</div>




```python
results_adjaudsectors = pd.DataFrame(index=adj_sectorprices.columns, columns = ['Data_Date','Spot', 'Daily', 'WTD', 'MTD', 'YTD', 'YoY'])

for col in adj_sectorprices.columns:
    prices = adj_sectorprices[col]
    prices = prices.dropna()
    
    last_date = prices.last_valid_index()
    prior_date = last_date - pd.tseries.offsets.Day(days=2)
    
    start = prices.index[0]
    
    week = last_date - pd.tseries.offsets.Week(weekday=0)
    week = prices.index.get_loc(week,method='nearest')
    
    month = last_date - pd.tseries.offsets.BMonthBegin()
    month = prices.index.get_loc(month,method='nearest')
    
    year = last_date - pd.tseries.offsets.BYearBegin()
    year = prices.index.get_loc(year,method='nearest')

    close = prices[last_date]
    daily = round((close - prices[prior_date])/prices[prior_date]*100, 3)
    wtd = round((close - prices[week])/prices[week]*100, 3)
    mtd = round((close - prices[month])/prices[month]*100, 3)
    ytd = round((close - prices[year])/prices[year]*100, 3)
    yoy = round((close-prices[start])/prices[start]*100, 3)
    
    results_adjaudsectors.loc[col, ['Data_Date', 'Spot', 'Daily', 'WTD', 'MTD', 'YTD', 'YoY']] = [last_date, close, daily, wtd, mtd, ytd, yoy ]
    
```


```python
results_adjaudsectors
```




<div>
<font size = "2">
<table border="1" class="dataframe" width = "300">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Data_Date</th>
      <th>Spot</th>
      <th>Daily</th>
      <th>WTD</th>
      <th>MTD</th>
      <th>YTD</th>
      <th>YoY</th>
    </tr>
    <tr>
      <th>sector</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Consumer Discretionary</th>
      <td>2017-06-09 00:00:00</td>
      <td>17.4852</td>
      <td>-13.385</td>
      <td>-0.893</td>
      <td>-10.008</td>
      <td>13.229</td>
      <td>3.063</td>
    </tr>
    <tr>
      <th>Consumer Staples</th>
      <td>2017-06-09 00:00:00</td>
      <td>25.0601</td>
      <td>6.979</td>
      <td>92.76</td>
      <td>-18.096</td>
      <td>-7.165</td>
      <td>9.286</td>
    </tr>
    <tr>
      <th>Energy</th>
      <td>2017-06-09 00:00:00</td>
      <td>17.3026</td>
      <td>-3.792</td>
      <td>-4.748</td>
      <td>-3.941</td>
      <td>29.104</td>
      <td>-4.311</td>
    </tr>
    <tr>
      <th>Financials</th>
      <td>2017-06-09 00:00:00</td>
      <td>38.5179</td>
      <td>6.569</td>
      <td>-0.214</td>
      <td>6.104</td>
      <td>-0.031</td>
      <td>7.166</td>
    </tr>
    <tr>
      <th>Health Care</th>
      <td>2017-06-09 00:00:00</td>
      <td>84.7876</td>
      <td>-8.615</td>
      <td>1.844</td>
      <td>-1.168</td>
      <td>49.879</td>
      <td>55.14</td>
    </tr>
    <tr>
      <th>Industrials</th>
      <td>2017-06-09 00:00:00</td>
      <td>8.70551</td>
      <td>-5.145</td>
      <td>-13.527</td>
      <td>-9.584</td>
      <td>-8.031</td>
      <td>5.921</td>
    </tr>
    <tr>
      <th>Information Technology</th>
      <td>2017-06-09 00:00:00</td>
      <td>8.70933</td>
      <td>-1.757</td>
      <td>-2.608</td>
      <td>-8.475</td>
      <td>2.295</td>
      <td>3.455</td>
    </tr>
    <tr>
      <th>Materials</th>
      <td>2017-06-09 00:00:00</td>
      <td>17.466</td>
      <td>8.354</td>
      <td>-5.28</td>
      <td>1.476</td>
      <td>5.38</td>
      <td>26.412</td>
    </tr>
    <tr>
      <th>Real Estate</th>
      <td>2017-06-09 00:00:00</td>
      <td>6.55767</td>
      <td>1.143</td>
      <td>-0.893</td>
      <td>3.349</td>
      <td>-1.541</td>
      <td>21.346</td>
    </tr>
    <tr>
      <th>Telecommunication Services</th>
      <td>2017-06-09 00:00:00</td>
      <td>4.28168</td>
      <td>1.674</td>
      <td>1.294</td>
      <td>-2.579</td>
      <td>-15.936</td>
      <td>-24.512</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>2017-06-09 00:00:00</td>
      <td>16.2257</td>
      <td>-2.395</td>
      <td>-4.747</td>
      <td>11.369</td>
      <td>56.829</td>
      <td>73.865</td>
    </tr>
  </tbody>
</table>
</font>
</div>



### USD Sectors


```python
p_usd_data['volume_traded'] = p_usd_data.adj_close_price*p_usd_data.volume
total_marketcap = p_usd_data.groupby(['price_date', 'sector'])['volume_traded'].sum()
total_marketcap = total_marketcap.reset_index()  
total_marketcap.columns = ['price_date', 'sector', 'total_volume_traded']
merged_usd_data = p_usd_data.merge(total_marketcap)
merged_usd_data['weighted_price'] =  merged_usd_data['adj_close_price']*merged_usd_data['volume_traded']/merged_usd_data['total_volume_traded']
```


```python
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize = (10,8))
fig.suptitle('Unadjusted v Adjusted Volume Weighted USD Stock Prices', fontsize=20, fontweight='bold')

unadj_sectprices = p_usd_data.groupby(['price_date', 'sector'])['adj_close_price'].mean()
unadj_sectprices = unadj_sectprices.unstack()
unadj_sectprices.plot(ax=ax[0], legend = False)
ax[0].set_ylabel('Unadjusted Price ($)')

adj_sectorprices = merged_usd_data.groupby(['price_date', 'sector'])['weighted_price'].sum()
adj_sectorprices= adj_sectorprices.unstack()
adj_sectorprices = adj_sectorprices.dropna()

adj_sectorprices.plot(ax=ax[1])
ax[1].set_ylabel('Volume Weighted Price ($)')
plt.legend(bbox_to_anchor=(1.1, 1.5), prop={'size':12})
plt.show()
```


![png](/img/findashboard_27_0.png)



```python
sectors = p_usd_data.groupby(['sector', 'subsector']).count()
sectors.reset_index(inplace=True)
np.unique(sectors[sectors.sector=='information_technology']['subsector'])
np.unique(sectors[sectors.sector=='financials']['subsector'])

```




    array(['asset_management_&_custody_banks', 'consumer_finance',
           'diversified_banks', 'financial_exchanges_&_data',
           'insurance_brokers', 'investment_banking_&_brokerage',
           'life_&_health_insurance', 'multi-line_insurance',
           'multi-sector_holdings', 'property_&_casualty_insurance',
           'regional_banks', 'thrifts_&_mortgage_finance'], dtype=object)



## Renewable Energy Stocks
Here I run some analysis on a renewable energy stocks. I first parse some in from Wikipedia, and then use an adapted Yahoo-API to pull in all the relevant data. Note that the Yahoo-API has been officially discontinued as at May-2017, so this process likely won't last for long.


```python
import datetime
from bs4 import BeautifulSoup
import urllib
import io
import requests
from math import ceil

# Function to pull table from Wikipedia containing Renewable Energy Companies
def obtain_parse_wiki_renewables():
    now = datetime.datetime.utcnow()
    url = "https://en.wikipedia.org/wiki/List_of_renewable_energy_companies_by_stock_exchange"
    request = urllib.request.Request(url)
    page = urllib.request.urlopen(request)
    soup = BeautifulSoup(page)
    
    table = soup.find("table", {"class": "wikitable sortable"})  
    
    symbols = pd.DataFrame(columns=["Company", "Exchange", "Exchange_Ticker", "Stock_Ticker", "IPO_Date", "Industry"])
  
    for row in table.findAll("tr"):
        col = row.findAll("td")
        if len(col) > 0:
            try:
                company = str(col[0].find("a").string.strip()).lower().replace(" ", "_")
                exchange = str(col[1].find("a").string.strip()).lower().replace(" ", "_")
                exc_ticker = str(col[2].findAll("a")[0].string.strip())
                stock_ticker = str(col[2].findAll("a")[1].string.strip())
                if(len(col[3]) == 0):
                    ipo_date = "-"
                else:
                    ipo_date = str(col[3].string.strip())
                industry = str(col[4].string.strip().lower().replace(" ", "_"))

                symbols.loc[len(symbols)] = [company, exchange, exc_ticker, stock_ticker, ipo_date, industry]
            except:
                print("Passed.")

    return symbols

symbols1 = obtain_parse_wiki_renewables()

```

    C:\Users\Clint_PC\Anaconda3\lib\site-packages\bs4\__init__.py:181: UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("lxml"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.
    
    The code that caused this warning is on line 193 of the file C:\Users\Clint_PC\Anaconda3\lib\runpy.py. To get rid of this warning, change code that looks like this:
    
     BeautifulSoup(YOUR_MARKUP})
    
    to this:
    
     BeautifulSoup(YOUR_MARKUP, "lxml")
    
      markup_type=markup_type))
    

    Passed.
    Passed.
    


```python
# Dictionary to add Yahoo Finance Exchange codes
yahoo_conv = dict({
   'AIM' :  ".L",
'Athex' : ".AT",
'Euronext' : ".LS",
'GTSM' :"",
'KRX' : ".KR",
'Nasdaq Copenhagen' : "",
'OTC Pink' : "",
'OTCQB' : "",
'TASE' : ".TA",
'TSX' : ".TO",
'TSX-V' : ".CO",
'TWSE' :  ".TW",
'FWD' : ".DE",
'BSE' : ".BSE",
'BMAD' :  ".MC",
'LSE' : ".L",
'OTCBB' : ".OB",
'NYSE' : "",
'NASDAW' : "",
'ASX' : ".AX",
'SEHK' :  ".HK",
'BIT' : ".MI",
'SGX' :  ".SI",
'NZX' : ".NZ" ,
'FWB' : ".DE",
'NASDAQ' :""
})

# Clean up data from Wikipedia
symbols1.ix[symbols1.Company == 'suntech_power', 'Stock_Ticker'] = 'STPFQ' 
tmp_list = [yahoo_conv[x] for x in symbols1.Exchange_Ticker] # applies conversion so we can access Yahoo Finance API (going to be deprecated)
symbols1['adj_ticker'] = symbols1.Stock_Ticker + tmp_list
```


```python
symbols1
```


<div>
<font size="2">
<table border="1" class="dataframe" width="300">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Exchange</th>
      <th>Exchange_Ticker</th>
      <th>Stock_Ticker</th>
      <th>IPO_Date</th>
      <th>Industry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7c_solarparken</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>HRPK</td>
      <td>-</td>
      <td>renewables</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a2z_group</td>
      <td>mumbai</td>
      <td>BSE</td>
      <td>533292</td>
      <td>-</td>
      <td>solar_thermal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abengoa,_sa</td>
      <td>madrid</td>
      <td>BMAD</td>
      <td>ABG</td>
      <td>-</td>
      <td>solar_thermal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aleo_solar</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>AS1</td>
      <td>2006</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clean_power_investors,_ltd</td>
      <td>london</td>
      <td>LSE</td>
      <td>ALR</td>
      <td>2004</td>
      <td>renewables</td>
    </tr>
    <tr>
      <th>5</th>
      <td>alterra_power</td>
      <td>toronto</td>
      <td>TSX</td>
      <td>AXY</td>
      <td>2011</td>
      <td>geothermal,_hydro,_wind,_solar</td>
    </tr>
    <tr>
      <th>6</th>
      <td>americas_wind_energy_corporation</td>
      <td>new_york_city</td>
      <td>OTCBB</td>
      <td>AWNE</td>
      <td>2006</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>7</th>
      <td>anwell_technologies</td>
      <td>singapore</td>
      <td>SGX</td>
      <td>G5X</td>
      <td>2004</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ascent_solar_technologies,_inc</td>
      <td>new_york_city</td>
      <td>OTCQB</td>
      <td>ASTI</td>
      <td>2006</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>9</th>
      <td>aventine_renewable_energy</td>
      <td>new_york_city</td>
      <td>NYSE</td>
      <td>AVR</td>
      <td>-</td>
      <td>bio_energy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ballard_power_systems</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>BLDP</td>
      <td>1995</td>
      <td>fuel_cells</td>
    </tr>
    <tr>
      <th>11</th>
      <td>brookfield_renewable_energy_partners_lp</td>
      <td>new_york_city</td>
      <td>NYSE</td>
      <td>BEP</td>
      <td>1995</td>
      <td>hydroelectric,_solar,_wind</td>
    </tr>
    <tr>
      <th>12</th>
      <td>carnegie_wave_energy,_ltd</td>
      <td>sydney</td>
      <td>ASX</td>
      <td>CCE</td>
      <td>1993</td>
      <td>wave</td>
    </tr>
    <tr>
      <th>13</th>
      <td>canadian_solar,_inc</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>CSIQ</td>
      <td>2006</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>14</th>
      <td>centrosolar_group,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>C3O</td>
      <td>2005</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>15</th>
      <td>centrotherm_photovoltaics,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>CTN</td>
      <td>2007</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ceramic_fuel_cells,_ltd</td>
      <td>sydney</td>
      <td>ASX</td>
      <td>CFU</td>
      <td>2004</td>
      <td>fuel_cells</td>
    </tr>
    <tr>
      <th>17</th>
      <td>china_power_new_energy</td>
      <td>hong_kong</td>
      <td>SEHK</td>
      <td>735</td>
      <td>1999</td>
      <td>wind/hydro/biomass</td>
    </tr>
    <tr>
      <th>18</th>
      <td>china_sunergy_co,_ltd</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>CSUN</td>
      <td>2007</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>19</th>
      <td>comtec_solar_systems_group_limited</td>
      <td>hong_kong</td>
      <td>SEHK</td>
      <td>712</td>
      <td>2009</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>20</th>
      <td>conergy,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>CGY</td>
      <td>-</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>21</th>
      <td>clenergen_corporation</td>
      <td>new_york_city</td>
      <td>OTCBB</td>
      <td>CRGE</td>
      <td>-</td>
      <td>biomass</td>
    </tr>
    <tr>
      <th>22</th>
      <td>daystar_technologies,_inc</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>DSTI</td>
      <td>2004</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>23</th>
      <td>delsolar_co,_ltd</td>
      <td>taiwan</td>
      <td>GTSM</td>
      <td>3599</td>
      <td>-</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>24</th>
      <td>dongfang_electric</td>
      <td>hong_kong</td>
      <td>SEHK</td>
      <td>1072</td>
      <td>1994</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>25</th>
      <td>dyesol,_ltd</td>
      <td>sydney</td>
      <td>ASX</td>
      <td>DYE</td>
      <td>2005</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>26</th>
      <td>enel_green_power_s.p.a.</td>
      <td>milano</td>
      <td>BIT</td>
      <td>EGPW</td>
      <td>2010</td>
      <td>renewables</td>
    </tr>
    <tr>
      <th>27</th>
      <td>enerdynamic_hybrid_technologies_inc.</td>
      <td>tsx-v</td>
      <td>TSX-V</td>
      <td>EHT</td>
      <td>2014</td>
      <td>wind,_solar</td>
    </tr>
    <tr>
      <th>28</th>
      <td>energiekontor,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>EKT</td>
      <td>2000</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>29</th>
      <td>enlight_renewable_energy_ltd.</td>
      <td>tel_aviv</td>
      <td>TASE</td>
      <td>ENLT</td>
      <td>-</td>
      <td>wind,_solar</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>renesola,_ltd</td>
      <td>new_york_city</td>
      <td>NYSE</td>
      <td>SOL</td>
      <td>2007</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>67</th>
      <td>renewable_energy_generation,_ltd</td>
      <td>london</td>
      <td>LSE</td>
      <td>RWE</td>
      <td>2005</td>
      <td>renewables</td>
    </tr>
    <tr>
      <th>68</th>
      <td>renewable_energy_holdings,_plc</td>
      <td>london</td>
      <td>LSE</td>
      <td>REH</td>
      <td>2005</td>
      <td>renewables</td>
    </tr>
    <tr>
      <th>69</th>
      <td>renewable_energy_resources,_inc</td>
      <td>new_york_city</td>
      <td>OTCBB</td>
      <td>RWER</td>
      <td>2008</td>
      <td>hydro</td>
    </tr>
    <tr>
      <th>70</th>
      <td>run_of_river_power_inc.</td>
      <td>tsx-v</td>
      <td>TSX-V</td>
      <td>ROR</td>
      <td>2005</td>
      <td>hydro</td>
    </tr>
    <tr>
      <th>71</th>
      <td>shear_wind_inc.</td>
      <td>tsx-v</td>
      <td>TSX-V</td>
      <td>SWX</td>
      <td>2006</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>72</th>
      <td>sinovel</td>
      <td>shanghai</td>
      <td>SSE</td>
      <td>601558</td>
      <td>2011</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>73</th>
      <td>s.a.g._solarstrom,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>SAG</td>
      <td>-</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>74</th>
      <td>sma_solar_technology,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>S92</td>
      <td>2008</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>75</th>
      <td>solar-fabrik,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>SFX</td>
      <td>-</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>76</th>
      <td>solar3d_inc.</td>
      <td>new_york_city</td>
      <td>OTCBB</td>
      <td>SLTD</td>
      <td>2010</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>77</th>
      <td>solarcity_corporation</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>SCTY</td>
      <td>2012</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>78</th>
      <td>solarfun_power_holdings_co,_ltd</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>SOLF</td>
      <td>2006</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>79</th>
      <td>solarworld,_ag</td>
      <td>frankfurt</td>
      <td>FWB</td>
      <td>SWV</td>
      <td>1999</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>80</th>
      <td>solco</td>
      <td>sydney</td>
      <td>ASX</td>
      <td>GOE</td>
      <td>2000</td>
      <td>solar_thermal</td>
    </tr>
    <tr>
      <th>81</th>
      <td>sunedison,_inc.</td>
      <td>new_york_city</td>
      <td>NYSE</td>
      <td>SUNE</td>
      <td>1984</td>
      <td>photovoltaics/wind</td>
    </tr>
    <tr>
      <th>82</th>
      <td>sunpower_corporation</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>SPWR</td>
      <td>2005</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>83</th>
      <td>sunrun</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>RUN</td>
      <td>2015</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>84</th>
      <td>suntech_power</td>
      <td>new_york_city</td>
      <td>NYSE</td>
      <td>STPFQ</td>
      <td>2005</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>85</th>
      <td>suzlon_energy</td>
      <td>mumbai</td>
      <td>BSE</td>
      <td>532667</td>
      <td>2005</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>86</th>
      <td>synex_international</td>
      <td>toronto</td>
      <td>TSX</td>
      <td>SXI</td>
      <td>1999</td>
      <td>hydro</td>
    </tr>
    <tr>
      <th>87</th>
      <td>terna_energy</td>
      <td>athens</td>
      <td>Athex</td>
      <td>TENERGY</td>
      <td>2009</td>
      <td>wind,_hydro</td>
    </tr>
    <tr>
      <th>88</th>
      <td>tiger_renewable_energy,_ltd</td>
      <td>new_york_city</td>
      <td>OTCBB</td>
      <td>TGRW</td>
      <td>-</td>
      <td>ethanol</td>
    </tr>
    <tr>
      <th>89</th>
      <td>trina_solar,_ltd</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>TSL</td>
      <td>2006</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>90</th>
      <td>verenium_corporation</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>VRNM</td>
      <td>-</td>
      <td>biofuels</td>
    </tr>
    <tr>
      <th>91</th>
      <td>vestas_wind_systems</td>
      <td>copenhagen</td>
      <td>Nasdaq Copenhagen</td>
      <td>VWS</td>
      <td>1998</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>92</th>
      <td>vivint_solar</td>
      <td>new_york_city</td>
      <td>NASDAQ</td>
      <td>VSLR</td>
      <td>2014</td>
      <td>photovoltaics</td>
    </tr>
    <tr>
      <th>93</th>
      <td>waterfurnace_renewable_energy,_inc.</td>
      <td>toronto</td>
      <td>TSX</td>
      <td>WFI</td>
      <td>2001</td>
      <td>geothermal</td>
    </tr>
    <tr>
      <th>94</th>
      <td>windflow_technology,_ltd</td>
      <td>wellington</td>
      <td>NZX</td>
      <td>WTL</td>
      <td>2001</td>
      <td>wind</td>
    </tr>
    <tr>
      <th>95</th>
      <td>yingli_green_energy_holding_co,_ltd</td>
      <td>new_york_city</td>
      <td>NYSE</td>
      <td>YGE</td>
      <td>2007</td>
      <td>photovoltaics</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 6 columns</p>
</font>
</div>




```python
import yahoo_extractor

# Gathers price data from yahoo API using https://github.com/c0redumb/yahoo_quote_download and then calculates summary statistics 
def get_prices(adj_ticker, start_date):
    
    prices = pd.DataFrame(columns=["Date","Adj.Close"])
    
    # Try to read in data, exit if fail
    try:
        data = yahoo_extractor.load_yahoo_quote(adj_ticker, begindate=start_date.strftime("%Y-%m-%d"), 
                                                enddate = datetime.date.today().strftime("%Y-%m-%d"))
        for i in data:
            tmp = i.split(',')
            if(len(i) > 0 and tmp[0] != 'Date'):
                tmp = [x if x != 'null' else None for x in tmp]
                prices.loc[len(prices)] = [tmp[0],tmp[5]]
    except:
        print("Could not download Yahoo data for " + adj_ticker)
        
    # Try to calculate summary statistics, exit if fail
    try:   
        if(len(prices) > 5):
            prices.Date = pd.to_datetime(prices.Date)
            prices['Adj.Close'] = np.float64(prices['Adj.Close'])
            prices.index = prices.Date
            results = pd.DataFrame(columns=['Ticker', 'Price_Date', 'Adj.Close', 'Daily Change', 'WTD', 'MTD', 'YTD', 'YoY'])

            last_date = prices.last_valid_index()
            prior_date = last_date - pd.tseries.offsets.Day(days=1)

            start = prices.Date[0]

            week = last_date - pd.tseries.offsets.Week(weekday=0)
            week = prices.index.get_loc(week,method='nearest')

            month = last_date - pd.tseries.offsets.BMonthBegin()
            month = prices.index.get_loc(month,method='nearest')

            year = last_date - pd.tseries.offsets.BYearBegin()
            year = prices.index.get_loc(year,method='nearest')

            close = prices.loc[last_date, 'Adj.Close']
            daily = round((close - prices.ix[prior_date,'Adj.Close'])/prices.ix[prior_date, 'Adj.Close']*100, 3)
            wtd = round((close - prices.ix[week, 'Adj.Close'])/prices.ix[week, 'Adj.Close']*100, 3)
            mtd = round((close - prices.ix[month, 'Adj.Close'])/prices.ix[month, 'Adj.Close']*100, 3)
            ytd = round((close - prices.ix[year, 'Adj.Close'])/prices.ix[year, 'Adj.Close']*100, 3)
            yoy = round((close - prices.ix[start, 'Adj.Close'])/prices.ix[start, 'Adj.Close']*100, 3)

            results.loc[len(results)]=[adj_ticker,last_date.strftime("%Y-%m-%d"), close, daily, wtd, mtd, ytd, yoy ]
            return results
        else:
            return 'pass'
    except:
        return 'pass'

```


```python
# Pull in all relevant data
renewables_analysis = pd.DataFrame(columns=['Ticker', 'Price_Date', 'Adj.Close', 'Daily Change', 'WTD', 'MTD', 'YTD', 'YoY'])
for t in symbols1.index:
    print("Adding data for %s" % symbols1.loc[t].adj_ticker)
    results = get_prices(symbols1.loc[t].adj_ticker, datetime.date(2016,6,9))
    if(type(results) != str):
        renewables_analysis.loc[len(renewables_analysis)] = results.loc[0]

```

### Summary Plots
Now that we've pulled in all the data, we want to actually see if we can learn anything useful from it! As a first pass, I'm just going to look at various returns over the past year and see if anything pops up about certain sectors. Note that our dataset is quite small as the entire listed renewables sector isn't massive, definitely significant work can be done here to expand this analysis to incorporate many more companies.


```python
combined_data = renewables_analysis.merge(symbols1, left_on='Ticker', right_on = 'adj_ticker')
combined_data.drop(31)
```

```python
combined_data.groupby(['Company'])['YTD'].sum().sort_values(ascending=True).plot(kind='barh', figsize = (12,10))
plt.title("YTD Change in Stock Price for Renewables", fontsize=20, fontweight='bold')
plt.xlabel("YTD Change (%)")
plt.show()
```

![png](/img/findashboard_37_0.png)


A somewhat unsurprising result, we see that stocks in fuel cells have been doing very well over the past year.


```python
combined_data.groupby(['Industry'])['YTD'].mean().sort_values(ascending=False).plot(kind='barh', figsize = (12,10))
plt.title("Average YTD Change in Stock Price for Renewable Subsectors", fontsize=20, fontweight='bold')
plt.xlabel("Average YTD Change (%)")
plt.show()
```


![png](/img/findashboard_39_0.png)

