---
layout:     post
title:      "Exploration of FDIC Failures Data"
subtitle:   "We look at historical bank failures in the USA, and identify some key temporal trends as well as location based trends. We then kick-off a basic look at what metrics might drive these failures."
date:       2017-03-11 12:00:00
author:     "Clint Howard"
header-img: "/img/post-bg-06.jpg"
category: Portfolio
tags: [python, data, finance]
comments: True
---



# Investigation of Bank Failures
Data provided from Kaggle: https://www.kaggle.com/fdic/bank-failures


```python

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

```


```python
df = pd.read_csv("bank-failures/banks.csv")
df.head()
df.count()
```




    Financial Institution Number    2883
    Institution Name                3484
    Institution Type                3484
    Charter Type                    3484
    Headquarters                    3484
    Failure Date                    3484
    Insurance Fund                  3484
    Certificate Number              2999
    Transaction Type                3484
    Total Deposits                  3484
    Total Assets                    3333
    Estimated Loss (2015)           2509
    dtype: int64



## Data Cleaning


```python
df = df.join( df["Headquarters"].str.split(",", expand = True))
df["D/A"] = df["Total Deposits"]/df["Total Assets"]
df["Failure Date"] = pd.to_datetime(df["Failure Date"])
df = df.rename(columns = {0:"City", 1:"State", 2:"Del"})
df = df.drop("Del", 1)
df.count()
```




    Financial Institution Number    2883
    Institution Name                3484
    Institution Type                3484
    Charter Type                    3484
    Headquarters                    3484
    Failure Date                    3484
    Insurance Fund                  3484
    Certificate Number              2999
    Transaction Type                3484
    Total Deposits                  3484
    Total Assets                    3333
    Estimated Loss (2015)           2509
    City                            3484
    State                           3484
    D/A                             3333
    dtype: int64



## Time Plots
Not unexpectedly, we see the largest peaks in the lates 80's and early 90's corresponding to the Savings and Loan crisis, 2009/2010 corresponding to the GFC and the late 30's with The Great Depression. What would be more interesting would be to look at the percentage failures of banks (to do.)


```python
fails_by_year = df['Failure Date'].groupby([df["Failure Date"].dt.year]).agg('count')
plt.figure(figsize=(15,10))
sns.barplot(fails_by_year.index, fails_by_year)
plt.xticks(rotation="vertical")
plt.ylabel("Bank Failures")
plt.show()
```


![png](/img/fdic_6_0.png)


## Geographical Plots



```python
df.groupby("State").count()["Failure Date"].sort_values(ascending=False)[0:25].plot(kind="bar")
plt.ylabel("Total Failures")
plt.show()
```


![png](/img/fdic_8_0.png)


We see that TX has both the largest number of failures and the largest total estimated losses. Let's see whose contributing to this.


```python
df.groupby("State").sum()["Estimated Loss (2015)"].sort_values(ascending=False)[0:25].plot(kind="bar")
plt.ylabel("Total Estimated Losses")
plt.show()
```


![png](/img/fdic_10_0.png)



```python
df[df["State"] == " TX"].sort_values(by = "Estimated Loss (2015)", ascending = False)[0:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Financial Institution Number</th>
      <th>Institution Name</th>
      <th>Institution Type</th>
      <th>Charter Type</th>
      <th>Headquarters</th>
      <th>Failure Date</th>
      <th>Insurance Fund</th>
      <th>Certificate Number</th>
      <th>Transaction Type</th>
      <th>Total Deposits</th>
      <th>Total Assets</th>
      <th>Estimated Loss (2015)</th>
      <th>City</th>
      <th>State</th>
      <th>D/A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1520</th>
      <td>6938.0</td>
      <td>UNIVERSITY FEDERAL SAVINGS</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>HOUSTON, TX</td>
      <td>1989-02-14</td>
      <td>RTC</td>
      <td>30685.0</td>
      <td>ACQUISITION</td>
      <td>3776427</td>
      <td>4480389.0</td>
      <td>2177985.0</td>
      <td>HOUSTON</td>
      <td>TX</td>
      <td>0.842879</td>
    </tr>
    <tr>
      <th>1377</th>
      <td>2846.0</td>
      <td>FIRST REPUBLICBANK-DALLAS, N.A.</td>
      <td>COMMERCIAL BANK</td>
      <td>FEDERAL</td>
      <td>DALLAS, TX</td>
      <td>1988-07-29</td>
      <td>FDIC</td>
      <td>3165.0</td>
      <td>ACQUISITION</td>
      <td>7680063</td>
      <td>17085655.0</td>
      <td>2017459.0</td>
      <td>DALLAS</td>
      <td>TX</td>
      <td>0.449504</td>
    </tr>
    <tr>
      <th>2363</th>
      <td>2124.0</td>
      <td>SAN JACINTO SAVINGS</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>HOUSTON, TX</td>
      <td>1990-11-30</td>
      <td>RTC</td>
      <td>31058.0</td>
      <td>ACQUISITION</td>
      <td>2894745</td>
      <td>2869629.0</td>
      <td>1700654.0</td>
      <td>HOUSTON</td>
      <td>TX</td>
      <td>1.008752</td>
    </tr>
    <tr>
      <th>1506</th>
      <td>7070.0</td>
      <td>GILL SA</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>SAN ANTONIO, TX</td>
      <td>1989-02-07</td>
      <td>RTC</td>
      <td>31503.0</td>
      <td>ACQUISITION</td>
      <td>1448432</td>
      <td>1207294.0</td>
      <td>1659803.0</td>
      <td>SAN ANTONIO</td>
      <td>TX</td>
      <td>1.199734</td>
    </tr>
    <tr>
      <th>1618</th>
      <td>7335.0</td>
      <td>COMMONWEALTH SAVINGS ASSOC.</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>HOUSTON, TX</td>
      <td>1989-03-09</td>
      <td>RTC</td>
      <td>31896.0</td>
      <td>TRANSFER</td>
      <td>1608452</td>
      <td>1647893.0</td>
      <td>1613353.0</td>
      <td>HOUSTON</td>
      <td>TX</td>
      <td>0.976066</td>
    </tr>
    <tr>
      <th>1714</th>
      <td>2985.0</td>
      <td>MBANK DALLAS, NATIONAL ASSOCIATION</td>
      <td>COMMERCIAL BANK</td>
      <td>FEDERAL</td>
      <td>DALLAS, TX</td>
      <td>1989-03-28</td>
      <td>FDIC</td>
      <td>3163.0</td>
      <td>ACQUISITION</td>
      <td>4033803</td>
      <td>6556056.0</td>
      <td>1610809.0</td>
      <td>DALLAS</td>
      <td>TX</td>
      <td>0.615279</td>
    </tr>
    <tr>
      <th>1517</th>
      <td>6952.0</td>
      <td>BRIGHT BANC</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>DALLAS, TX</td>
      <td>1989-02-10</td>
      <td>RTC</td>
      <td>31095.0</td>
      <td>ACQUISITION</td>
      <td>3004443</td>
      <td>4388466.0</td>
      <td>1307798.0</td>
      <td>DALLAS</td>
      <td>TX</td>
      <td>0.684623</td>
    </tr>
    <tr>
      <th>1823</th>
      <td>2100.0</td>
      <td>VICTORIA SA</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>SAN ANTONIO, TX</td>
      <td>1989-06-29</td>
      <td>RTC</td>
      <td>29378.0</td>
      <td>PAYOUT</td>
      <td>855717</td>
      <td>882849.0</td>
      <td>968972.0</td>
      <td>SAN ANTONIO</td>
      <td>TX</td>
      <td>0.969268</td>
    </tr>
    <tr>
      <th>1627</th>
      <td>2104.0</td>
      <td>BANCPLUS SAVINGS ASSOCIATION</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>PASADENA, TX</td>
      <td>1989-03-09</td>
      <td>RTC</td>
      <td>31128.0</td>
      <td>ACQUISITION</td>
      <td>923026</td>
      <td>751461.0</td>
      <td>964160.0</td>
      <td>PASADENA</td>
      <td>TX</td>
      <td>1.228309</td>
    </tr>
    <tr>
      <th>1619</th>
      <td>7429.0</td>
      <td>BENJAMIN FRANKLIN SA</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>HOUSTON, TX</td>
      <td>1989-03-09</td>
      <td>RTC</td>
      <td>30761.0</td>
      <td>ACQUISITION</td>
      <td>2004722</td>
      <td>2641392.0</td>
      <td>882240.0</td>
      <td>HOUSTON</td>
      <td>TX</td>
      <td>0.758964</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby("City").count()["Failure Date"].sort_values(ascending=False)[0:25].plot(kind="bar")
plt.ylabel("Total Failures")
plt.show()
```


![png](/img/fdic_12_0.png)



```python
df.groupby("State").mean()["D/A"].sort_values(ascending=False)[0:50].plot(kind="bar")
plt.ylabel("Mean Deposit to Asset Ratio")
plt.show()
```


![png](/img/fdic_13_0.png)


## Deposit to Assets
Ignoring the possibility that this is not real and hence not inflation adjusted. Interesting to see whether there is any relationship between D/A and the losses suffered. It looks like there is no strong correlation at all, which would indicate that it's not necessarily a mismanagement of deposits/assets which have driven bank failures, but other factors. Typically it's liquidity which brings the banks down, after sustaining a systemic asset shock (housing/mortgage exposures in the GFC).


```python
plt.scatter(x = df["D/A"], y = df["Estimated Loss (2015)"])
plt.xlabel("Deposits to Asset Ratio")
plt.ylabel("Estimated Loss")
plt.show()
```


![png](/img/fdic_15_0.png)



```python
plt.plot_date(x = df["Failure Date"], y = df["D/A"])
plt.ylabel("Deposits to Asset Ratio")
plt.show()
```


![png](/img/fdic_16_0.png)

