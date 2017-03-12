---
layout:     post
title:      "Analysis of FDIC Failures Data"
subtitle:   ""
date:       2017-03-11 12:00:00
author:     "Clint Howard"
header-img: "/img/post-bg-06.jpg"
category: Portfolio
tags: [jekyll, python]
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
df.groupby("State").count()["Failure Date"].sort_values(ascending=False)[1:25].plot(kind="bar")
plt.ylabel("Total Failures")
plt.show()
```


![png](/img/fdic_8_0.png)


We see that CA has both the largest number of failures and the largest total estimated losses. Let's see whose contributing to this.


```python
df.groupby("State").sum()["Estimated Loss (2015)"].sort_values(ascending=False)[1:25].plot(kind="bar")
plt.ylabel("Total Estimated Losses")
plt.show()
```


![png](/img/fdic_10_0.png)



```python
df[df["State"] == " CA"].sort_values(by = "Estimated Loss (2015)", ascending = False)[1:25]
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
      <th>1773</th>
      <td>7283.0</td>
      <td>LINCOLN SAVINGS &amp; LOAN</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>IRVINE, CA</td>
      <td>1989-04-14</td>
      <td>RTC</td>
      <td>29642.0</td>
      <td>ACQUISITION</td>
      <td>4193981</td>
      <td>4857204.0</td>
      <td>3142552.0</td>
      <td>IRVINE</td>
      <td>CA</td>
      <td>0.863456</td>
    </tr>
    <tr>
      <th>2573</th>
      <td>1279.0</td>
      <td>GREAT AMERICAN FSA</td>
      <td>SAVINGS BANK</td>
      <td>STATE</td>
      <td>SAN DIEGO, CA</td>
      <td>1991-08-09</td>
      <td>RTC</td>
      <td>28041.0</td>
      <td>ACQUISITION</td>
      <td>7230789</td>
      <td>9523603.0</td>
      <td>995496.0</td>
      <td>SAN DIEGO</td>
      <td>CA</td>
      <td>0.759249</td>
    </tr>
    <tr>
      <th>2766</th>
      <td>1243.0</td>
      <td>HOMEFED BANK, FA</td>
      <td>SAVINGS BANK</td>
      <td>STATE</td>
      <td>SAN DIEGO, CA</td>
      <td>1992-07-06</td>
      <td>RTC</td>
      <td>29234.0</td>
      <td>ACQUISITION</td>
      <td>8903571</td>
      <td>12175590.0</td>
      <td>751633.0</td>
      <td>SAN DIEGO</td>
      <td>CA</td>
      <td>0.731264</td>
    </tr>
    <tr>
      <th>3110</th>
      <td>10147.0</td>
      <td>UNITED COMMERCIAL BANK</td>
      <td>COMMERCIAL BANK</td>
      <td>STATE</td>
      <td>SAN FRANCISCO, CA</td>
      <td>2009-11-06</td>
      <td>DIF</td>
      <td>32469.0</td>
      <td>ACQUISITION</td>
      <td>7653666</td>
      <td>10895336.0</td>
      <td>645369.0</td>
      <td>SAN FRANCISCO</td>
      <td>CA</td>
      <td>0.702472</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>10185.0</td>
      <td>LA JOLLA BANK, FSB</td>
      <td>SAVINGS BANK</td>
      <td>STATE</td>
      <td>LA JOLLA, CA</td>
      <td>2010-02-19</td>
      <td>DIF</td>
      <td>32423.0</td>
      <td>ACQUISITION</td>
      <td>2799362</td>
      <td>3646071.0</td>
      <td>604483.0</td>
      <td>LA JOLLA</td>
      <td>CA</td>
      <td>0.767775</td>
    </tr>
    <tr>
      <th>2408</th>
      <td>2188.0</td>
      <td>FAR WEST SAVINGS &amp; LOAN</td>
      <td>SAVINGS ASSOCIATION</td>
      <td>FEDERAL/STATE</td>
      <td>NEWPORT BEACH, CA</td>
      <td>1991-01-11</td>
      <td>RTC</td>
      <td>28292.0</td>
      <td>ACQUISITION</td>
      <td>2981632</td>
      <td>3714988.0</td>
      <td>498684.0</td>
      <td>NEWPORT BEACH</td>
      <td>CA</td>
      <td>0.802595</td>
    </tr>
    <tr>
      <th>3102</th>
      <td>10134.0</td>
      <td>CALIFORNIA NATIONAL BANK</td>
      <td>COMMERCIAL BANK</td>
      <td>FEDERAL</td>
      <td>LOS ANGELES, CA</td>
      <td>2009-10-30</td>
      <td>DIF</td>
      <td>34659.0</td>
      <td>ACQUISITION</td>
      <td>6145207</td>
      <td>7781100.0</td>
      <td>413527.0</td>
      <td>LOS ANGELES</td>
      <td>CA</td>
      <td>0.789761</td>
    </tr>
    <tr>
      <th>3124</th>
      <td>10161.0</td>
      <td>IMPERIAL CAPITAL BANK</td>
      <td>COMMERCIAL BANK</td>
      <td>STATE</td>
      <td>LA JOLLA, CA</td>
      <td>2009-12-18</td>
      <td>DIF</td>
      <td>26348.0</td>
      <td>ACQUISITION</td>
      <td>2822300</td>
      <td>4046888.0</td>
      <td>328347.0</td>
      <td>LA JOLLA</td>
      <td>CA</td>
      <td>0.697400</td>
    </tr>
    <tr>
      <th>2985</th>
      <td>10024.0</td>
      <td>PFF BANK &amp; TRUST</td>
      <td>SAVINGS BANK</td>
      <td>STATE</td>
      <td>POMONA, CA</td>
      <td>2008-11-21</td>
      <td>DIF</td>
      <td>28344.0</td>
      <td>ACQUISITION</td>
      <td>2393845</td>
      <td>3715433.0</td>
      <td>318123.0</td>
      <td>POMONA</td>
      <td>CA</td>
      <td>0.644298</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby("City").count()["Failure Date"].sort_values(ascending=False)[1:25].plot(kind="bar")
plt.ylabel("Total Failures")
plt.show()
```


![png](/img/fdic_12_0.png)



```python
df.groupby("State").mean()["D/A"].sort_values(ascending=False)[1:50].plot(kind="bar")
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




