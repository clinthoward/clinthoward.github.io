---
layout:     post
title:      "Home Credit Default Risk Kaggle Competition"
subtitle:   "Basic approach to a Kaggle competition!"
date:       2018-05-26 12:00:00
author:     "Clint Howard"
category: Portfolio
tags: [python, data, kaggle]
comments: True
---
# Home Credit Default Risk Kaggle Competition
We'll look at a pretty primitive problem, estimating credit default risk of a consumer on a loan. This is from the Kaggle competition here: https://www.kaggle.com/c/home-credit-default-risk. More interesting is some of the exploratory analysis we can do on the data to look at the relationship between income, occuptation and credit amounts.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
```


```python
#payment_df = pd.read_csv(r"D:\Downloads\home_credit_data\installments_payments.csv") # repayment history one row for each payment/missed payment
#balance_df = pd.read_csv(r"D:\Downloads\home_credit_data\bureau_balance.csv") # monthly balances of previous credits
test_df = pd.read_csv(r"D:\Downloads\home_credit_data\application_test.csv")
train_df = pd.read_csv(r"D:\Downloads\home_credit_data\application_train.csv") # one row = one loan
#bureau_df = pd.read_csv(r"D:\Downloads\home_credit_data\bureau.csv") # clients previous credits/loadns
#pos_cash_df = pd.read_csv(r"D:\Downloads\home_credit_data\POS_CASH_balance.csv") # monthly balance of previous pos/cash loans
#creditcard_df = pd.read_csv(r"D:\Downloads\home_credit_data\credit_card_balance.csv") # monthly snapshots of previous credit cards
prevapp_df = pd.read_csv(r"D:\Downloads\home_credit_data\previous_application.csv") # all previous applications
```

## Training Data
Let's start with a high level view of the training data. Our goal is to predict the TARGET variable, where TARGET is described as 1 - client has payment difficulties and 0 - all other cases. Effectively, we want to predict whether a client will have difficulties repaying their loan based on the features we've been provided with. <br>
Given the breadth of extra data provided, such as credit card histories, previous loan applications across all agencies and monthly snapshots of point-of-sale/cash loans, there's a lot of scope for expanding the analysis. LET'S START SIMPLE!


```python
train_df.info(max_cols = 200)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 307511 entries, 0 to 307510
    Data columns (total 122 columns):
    SK_ID_CURR                      307511 non-null int64
    TARGET                          307511 non-null int64
    NAME_CONTRACT_TYPE              307511 non-null object
    CODE_GENDER                     307511 non-null object
    FLAG_OWN_CAR                    307511 non-null object
    FLAG_OWN_REALTY                 307511 non-null object
    CNT_CHILDREN                    307511 non-null int64
    AMT_INCOME_TOTAL                307511 non-null float64
    AMT_CREDIT                      307511 non-null float64
    AMT_ANNUITY                     307499 non-null float64
    AMT_GOODS_PRICE                 307233 non-null float64
    NAME_TYPE_SUITE                 306219 non-null object
    NAME_INCOME_TYPE                307511 non-null object
    NAME_EDUCATION_TYPE             307511 non-null object
    NAME_FAMILY_STATUS              307511 non-null object
    NAME_HOUSING_TYPE               307511 non-null object
    REGION_POPULATION_RELATIVE      307511 non-null float64
    DAYS_BIRTH                      307511 non-null int64
    DAYS_EMPLOYED                   307511 non-null int64
    DAYS_REGISTRATION               307511 non-null float64
    DAYS_ID_PUBLISH                 307511 non-null int64
    OWN_CAR_AGE                     104582 non-null float64
    FLAG_MOBIL                      307511 non-null int64
    FLAG_EMP_PHONE                  307511 non-null int64
    FLAG_WORK_PHONE                 307511 non-null int64
    FLAG_CONT_MOBILE                307511 non-null int64
    FLAG_PHONE                      307511 non-null int64
    FLAG_EMAIL                      307511 non-null int64
    OCCUPATION_TYPE                 211120 non-null object
    CNT_FAM_MEMBERS                 307509 non-null float64
    REGION_RATING_CLIENT            307511 non-null int64
    REGION_RATING_CLIENT_W_CITY     307511 non-null int64
    WEEKDAY_APPR_PROCESS_START      307511 non-null object
    HOUR_APPR_PROCESS_START         307511 non-null int64
    REG_REGION_NOT_LIVE_REGION      307511 non-null int64
    REG_REGION_NOT_WORK_REGION      307511 non-null int64
    LIVE_REGION_NOT_WORK_REGION     307511 non-null int64
    REG_CITY_NOT_LIVE_CITY          307511 non-null int64
    REG_CITY_NOT_WORK_CITY          307511 non-null int64
    LIVE_CITY_NOT_WORK_CITY         307511 non-null int64
    ORGANIZATION_TYPE               307511 non-null object
    EXT_SOURCE_1                    134133 non-null float64
    EXT_SOURCE_2                    306851 non-null float64
    EXT_SOURCE_3                    246546 non-null float64
    APARTMENTS_AVG                  151450 non-null float64
    BASEMENTAREA_AVG                127568 non-null float64
    YEARS_BEGINEXPLUATATION_AVG     157504 non-null float64
    YEARS_BUILD_AVG                 103023 non-null float64
    COMMONAREA_AVG                  92646 non-null float64
    ELEVATORS_AVG                   143620 non-null float64
    ENTRANCES_AVG                   152683 non-null float64
    FLOORSMAX_AVG                   154491 non-null float64
    FLOORSMIN_AVG                   98869 non-null float64
    LANDAREA_AVG                    124921 non-null float64
    LIVINGAPARTMENTS_AVG            97312 non-null float64
    LIVINGAREA_AVG                  153161 non-null float64
    NONLIVINGAPARTMENTS_AVG         93997 non-null float64
    NONLIVINGAREA_AVG               137829 non-null float64
    APARTMENTS_MODE                 151450 non-null float64
    BASEMENTAREA_MODE               127568 non-null float64
    YEARS_BEGINEXPLUATATION_MODE    157504 non-null float64
    YEARS_BUILD_MODE                103023 non-null float64
    COMMONAREA_MODE                 92646 non-null float64
    ELEVATORS_MODE                  143620 non-null float64
    ENTRANCES_MODE                  152683 non-null float64
    FLOORSMAX_MODE                  154491 non-null float64
    FLOORSMIN_MODE                  98869 non-null float64
    LANDAREA_MODE                   124921 non-null float64
    LIVINGAPARTMENTS_MODE           97312 non-null float64
    LIVINGAREA_MODE                 153161 non-null float64
    NONLIVINGAPARTMENTS_MODE        93997 non-null float64
    NONLIVINGAREA_MODE              137829 non-null float64
    APARTMENTS_MEDI                 151450 non-null float64
    BASEMENTAREA_MEDI               127568 non-null float64
    YEARS_BEGINEXPLUATATION_MEDI    157504 non-null float64
    YEARS_BUILD_MEDI                103023 non-null float64
    COMMONAREA_MEDI                 92646 non-null float64
    ELEVATORS_MEDI                  143620 non-null float64
    ENTRANCES_MEDI                  152683 non-null float64
    FLOORSMAX_MEDI                  154491 non-null float64
    FLOORSMIN_MEDI                  98869 non-null float64
    LANDAREA_MEDI                   124921 non-null float64
    LIVINGAPARTMENTS_MEDI           97312 non-null float64
    LIVINGAREA_MEDI                 153161 non-null float64
    NONLIVINGAPARTMENTS_MEDI        93997 non-null float64
    NONLIVINGAREA_MEDI              137829 non-null float64
    FONDKAPREMONT_MODE              97216 non-null object
    HOUSETYPE_MODE                  153214 non-null object
    TOTALAREA_MODE                  159080 non-null float64
    WALLSMATERIAL_MODE              151170 non-null object
    EMERGENCYSTATE_MODE             161756 non-null object
    OBS_30_CNT_SOCIAL_CIRCLE        306490 non-null float64
    DEF_30_CNT_SOCIAL_CIRCLE        306490 non-null float64
    OBS_60_CNT_SOCIAL_CIRCLE        306490 non-null float64
    DEF_60_CNT_SOCIAL_CIRCLE        306490 non-null float64
    DAYS_LAST_PHONE_CHANGE          307510 non-null float64
    FLAG_DOCUMENT_2                 307511 non-null int64
    FLAG_DOCUMENT_3                 307511 non-null int64
    FLAG_DOCUMENT_4                 307511 non-null int64
    FLAG_DOCUMENT_5                 307511 non-null int64
    FLAG_DOCUMENT_6                 307511 non-null int64
    FLAG_DOCUMENT_7                 307511 non-null int64
    FLAG_DOCUMENT_8                 307511 non-null int64
    FLAG_DOCUMENT_9                 307511 non-null int64
    FLAG_DOCUMENT_10                307511 non-null int64
    FLAG_DOCUMENT_11                307511 non-null int64
    FLAG_DOCUMENT_12                307511 non-null int64
    FLAG_DOCUMENT_13                307511 non-null int64
    FLAG_DOCUMENT_14                307511 non-null int64
    FLAG_DOCUMENT_15                307511 non-null int64
    FLAG_DOCUMENT_16                307511 non-null int64
    FLAG_DOCUMENT_17                307511 non-null int64
    FLAG_DOCUMENT_18                307511 non-null int64
    FLAG_DOCUMENT_19                307511 non-null int64
    FLAG_DOCUMENT_20                307511 non-null int64
    FLAG_DOCUMENT_21                307511 non-null int64
    AMT_REQ_CREDIT_BUREAU_HOUR      265992 non-null float64
    AMT_REQ_CREDIT_BUREAU_DAY       265992 non-null float64
    AMT_REQ_CREDIT_BUREAU_WEEK      265992 non-null float64
    AMT_REQ_CREDIT_BUREAU_MON       265992 non-null float64
    AMT_REQ_CREDIT_BUREAU_QRT       265992 non-null float64
    AMT_REQ_CREDIT_BUREAU_YEAR      265992 non-null float64
    dtypes: float64(65), int64(41), object(16)
    memory usage: 286.2+ MB
    


```python
train_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>AMT_GOODS_PRICE</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>REGION_POPULATION_RELATIVE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>DAYS_REGISTRATION</th>
      <th>DAYS_ID_PUBLISH</th>
      <th>OWN_CAR_AGE</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_EMP_PHONE</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_CONT_MOBILE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
      <th>REGION_RATING_CLIENT</th>
      <th>REGION_RATING_CLIENT_W_CITY</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>HOUR_APPR_PROCESS_START</th>
      <th>REG_REGION_NOT_LIVE_REGION</th>
      <th>REG_REGION_NOT_WORK_REGION</th>
      <th>LIVE_REGION_NOT_WORK_REGION</th>
      <th>REG_CITY_NOT_LIVE_CITY</th>
      <th>REG_CITY_NOT_WORK_CITY</th>
      <th>LIVE_CITY_NOT_WORK_CITY</th>
      <th>ORGANIZATION_TYPE</th>
      <th>EXT_SOURCE_1</th>
      <th>EXT_SOURCE_2</th>
      <th>EXT_SOURCE_3</th>
      <th>APARTMENTS_AVG</th>
      <th>BASEMENTAREA_AVG</th>
      <th>YEARS_BEGINEXPLUATATION_AVG</th>
      <th>YEARS_BUILD_AVG</th>
      <th>COMMONAREA_AVG</th>
      <th>ELEVATORS_AVG</th>
      <th>ENTRANCES_AVG</th>
      <th>FLOORSMAX_AVG</th>
      <th>FLOORSMIN_AVG</th>
      <th>LANDAREA_AVG</th>
      <th>LIVINGAPARTMENTS_AVG</th>
      <th>LIVINGAREA_AVG</th>
      <th>NONLIVINGAPARTMENTS_AVG</th>
      <th>NONLIVINGAREA_AVG</th>
      <th>APARTMENTS_MODE</th>
      <th>BASEMENTAREA_MODE</th>
      <th>YEARS_BEGINEXPLUATATION_MODE</th>
      <th>YEARS_BUILD_MODE</th>
      <th>COMMONAREA_MODE</th>
      <th>ELEVATORS_MODE</th>
      <th>ENTRANCES_MODE</th>
      <th>FLOORSMAX_MODE</th>
      <th>FLOORSMIN_MODE</th>
      <th>LANDAREA_MODE</th>
      <th>LIVINGAPARTMENTS_MODE</th>
      <th>LIVINGAREA_MODE</th>
      <th>NONLIVINGAPARTMENTS_MODE</th>
      <th>NONLIVINGAREA_MODE</th>
      <th>APARTMENTS_MEDI</th>
      <th>BASEMENTAREA_MEDI</th>
      <th>YEARS_BEGINEXPLUATATION_MEDI</th>
      <th>YEARS_BUILD_MEDI</th>
      <th>COMMONAREA_MEDI</th>
      <th>ELEVATORS_MEDI</th>
      <th>ENTRANCES_MEDI</th>
      <th>FLOORSMAX_MEDI</th>
      <th>FLOORSMIN_MEDI</th>
      <th>LANDAREA_MEDI</th>
      <th>LIVINGAPARTMENTS_MEDI</th>
      <th>LIVINGAREA_MEDI</th>
      <th>NONLIVINGAPARTMENTS_MEDI</th>
      <th>NONLIVINGAREA_MEDI</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>TOTALAREA_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
      <th>EMERGENCYSTATE_MODE</th>
      <th>OBS_30_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_30_CNT_SOCIAL_CIRCLE</th>
      <th>OBS_60_CNT_SOCIAL_CIRCLE</th>
      <th>DEF_60_CNT_SOCIAL_CIRCLE</th>
      <th>DAYS_LAST_PHONE_CHANGE</th>
      <th>FLAG_DOCUMENT_2</th>
      <th>FLAG_DOCUMENT_3</th>
      <th>FLAG_DOCUMENT_4</th>
      <th>FLAG_DOCUMENT_5</th>
      <th>FLAG_DOCUMENT_6</th>
      <th>FLAG_DOCUMENT_7</th>
      <th>FLAG_DOCUMENT_8</th>
      <th>FLAG_DOCUMENT_9</th>
      <th>FLAG_DOCUMENT_10</th>
      <th>FLAG_DOCUMENT_11</th>
      <th>FLAG_DOCUMENT_12</th>
      <th>FLAG_DOCUMENT_13</th>
      <th>FLAG_DOCUMENT_14</th>
      <th>FLAG_DOCUMENT_15</th>
      <th>FLAG_DOCUMENT_16</th>
      <th>FLAG_DOCUMENT_17</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100002</td>
      <td>1</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>202500.0</td>
      <td>406597.5</td>
      <td>24700.5</td>
      <td>351000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.018801</td>
      <td>-9461</td>
      <td>-637</td>
      <td>-3648.0</td>
      <td>-2120</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Business Entity Type 3</td>
      <td>0.083037</td>
      <td>0.262949</td>
      <td>0.139376</td>
      <td>0.0247</td>
      <td>0.0369</td>
      <td>0.9722</td>
      <td>0.6192</td>
      <td>0.0143</td>
      <td>0.00</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0369</td>
      <td>0.0202</td>
      <td>0.0190</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0252</td>
      <td>0.0383</td>
      <td>0.9722</td>
      <td>0.6341</td>
      <td>0.0144</td>
      <td>0.0000</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0377</td>
      <td>0.022</td>
      <td>0.0198</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0250</td>
      <td>0.0369</td>
      <td>0.9722</td>
      <td>0.6243</td>
      <td>0.0144</td>
      <td>0.00</td>
      <td>0.0690</td>
      <td>0.0833</td>
      <td>0.1250</td>
      <td>0.0375</td>
      <td>0.0205</td>
      <td>0.0193</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0149</td>
      <td>Stone, brick</td>
      <td>No</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1134.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100003</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>270000.0</td>
      <td>1293502.5</td>
      <td>35698.5</td>
      <td>1129500.0</td>
      <td>Family</td>
      <td>State servant</td>
      <td>Higher education</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>0.003541</td>
      <td>-16765</td>
      <td>-1188</td>
      <td>-1186.0</td>
      <td>-291</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Core staff</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>MONDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>School</td>
      <td>0.311267</td>
      <td>0.622246</td>
      <td>NaN</td>
      <td>0.0959</td>
      <td>0.0529</td>
      <td>0.9851</td>
      <td>0.7960</td>
      <td>0.0605</td>
      <td>0.08</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0130</td>
      <td>0.0773</td>
      <td>0.0549</td>
      <td>0.0039</td>
      <td>0.0098</td>
      <td>0.0924</td>
      <td>0.0538</td>
      <td>0.9851</td>
      <td>0.8040</td>
      <td>0.0497</td>
      <td>0.0806</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0128</td>
      <td>0.079</td>
      <td>0.0554</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0968</td>
      <td>0.0529</td>
      <td>0.9851</td>
      <td>0.7987</td>
      <td>0.0608</td>
      <td>0.08</td>
      <td>0.0345</td>
      <td>0.2917</td>
      <td>0.3333</td>
      <td>0.0132</td>
      <td>0.0787</td>
      <td>0.0558</td>
      <td>0.0039</td>
      <td>0.01</td>
      <td>reg oper account</td>
      <td>block of flats</td>
      <td>0.0714</td>
      <td>Block</td>
      <td>No</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-828.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100004</td>
      <td>0</td>
      <td>Revolving loans</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>67500.0</td>
      <td>135000.0</td>
      <td>6750.0</td>
      <td>135000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.010032</td>
      <td>-19046</td>
      <td>-225</td>
      <td>-4260.0</td>
      <td>-2531</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>Laborers</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>MONDAY</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Government</td>
      <td>NaN</td>
      <td>0.555912</td>
      <td>0.729567</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-815.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100006</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>312682.5</td>
      <td>29686.5</td>
      <td>297000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Civil marriage</td>
      <td>House / apartment</td>
      <td>0.008019</td>
      <td>-19005</td>
      <td>-3039</td>
      <td>-9833.0</td>
      <td>-2437</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Laborers</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>WEDNESDAY</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Business Entity Type 3</td>
      <td>NaN</td>
      <td>0.650442</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-617.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100007</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>121500.0</td>
      <td>513000.0</td>
      <td>21865.5</td>
      <td>513000.0</td>
      <td>Unaccompanied</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>0.028663</td>
      <td>-19932</td>
      <td>-3038</td>
      <td>-4311.0</td>
      <td>-3458</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Core staff</td>
      <td>1.0</td>
      <td>2</td>
      <td>2</td>
      <td>THURSDAY</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Religion</td>
      <td>NaN</td>
      <td>0.322738</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1106.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Loans and Income
We can see that the AMT_INCOME_TOTAL is heavily skewed by the high income earners, where the highest income earning is \$117,000,00 and the lowest is \$25,650. Once removing the larger outliers we can get a better picture of the distribution. <br>
The AMT_CREDIT seems to be slightly bi-modal, which is likely a function of the property markets that the loans are coming from.


```python
train_df['AMT_INCOME_TOTAL'].max()
```




    117000000.0




```python
train_df['AMT_INCOME_TOTAL'].min()
```




    25650.0




```python
plt.figure(figsize=(7,5))
sns.distplot(train_df['AMT_INCOME_TOTAL'].dropna())
plt.show()

plt.figure(figsize=(7,5))
sns.distplot(train_df.loc[train_df['AMT_INCOME_TOTAL'] < 0.25e7, 'AMT_INCOME_TOTAL'].dropna())
plt.show()

plt.figure(figsize=(7,5))
sns.distplot(train_df['AMT_CREDIT'].dropna())
plt.show()

plt.figure(figsize=(7,5))
sns.distplot(train_df['AMT_ANNUITY'].dropna())
plt.show()

plt.figure(figsize=(7,5))
sns.distplot(train_df['AMT_GOODS_PRICE'].dropna())
plt.show()
```


![png](/img/homecredit_9_0.png)



![png](/img/homecredit_9_1.png)



![png](/img/homecredit_9_2.png)



![png](/img/homecredit_9_3.png)



![png](/img/homecredit_9_4.png)


### Occupation vs Income/Credit
Interestingly, we see that laborers are by far the biggest users of loans, and also have the greatest variance in incomes! Not unexpected, but the average loan amounts seem fairly consistent. <br>

More interestingly, we can look at the distribution of income across different job types! Notice that there are roughly two "types" of distributions:
1. Tight with long tails (accountants, cleaning staff, managers)
2. Wide with not much of a tail (realty agents, low skill labor)

We also note that the distribution in credit amounts are roughly consistent across each occupation grouping


```python
train_df.groupby(['OCCUPATION_TYPE'])['OCCUPATION_TYPE'].count().sort_values(ascending=False).plot(kind='barh', figsize=(7,5))
plt.show()
```


![png](/img/homecredit_11_0.png)



```python
train_df.groupby(['OCCUPATION_TYPE'])['AMT_INCOME_TOTAL'].median().plot(kind='barh', figsize=(7,5))
plt.show()

train_df.groupby(['OCCUPATION_TYPE'])['AMT_INCOME_TOTAL'].std().plot(kind='barh', figsize=(7,5))
plt.show()
```


![png](/img/homecredit_12_0.png)



![png](/img/homecredit_12_1.png)



```python
train_df.groupby(['OCCUPATION_TYPE'])['AMT_CREDIT'].median().plot(kind='barh', figsize=(7,5))
plt.show()

train_df.groupby(['OCCUPATION_TYPE'])['AMT_CREDIT'].std().plot(kind='barh', figsize=(7,5))
plt.show()
```


![png](/img/homecredit_13_0.png)



![png](/img/homecredit_13_1.png)



```python
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15,10))
plt.suptitle('Distribution of Incomes by Occupation')
j = 0
for i in np.unique(train_df['OCCUPATION_TYPE'].dropna()):
    sns.distplot(train_df.loc[train_df['OCCUPATION_TYPE']==i, 'AMT_INCOME_TOTAL'], ax=axes.flat[j])
    axes.flat[j].set_title(i)
    j += 1
    
plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()
```


![png](/img/homecredit_14_0.png)



```python
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15,10))
plt.suptitle('Distribution of Credit Amounts by Occupation')
j = 0
for i in np.unique(train_df['OCCUPATION_TYPE'].dropna()):
    sns.distplot(train_df.loc[train_df['OCCUPATION_TYPE']==i, 'AMT_CREDIT'], ax=axes.flat[j])
    axes.flat[j].set_title(i)
    j += 1
    
plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()
```


![png](/img/homecredit_15_0.png)


### Categorical Variables - Education, living situation...


```python
train_df.groupby(['NAME_EDUCATION_TYPE'])['NAME_EDUCATION_TYPE'].count().sort_values(ascending=False).plot(kind='barh')
plt.show()

train_df.groupby(['NAME_INCOME_TYPE'])['NAME_INCOME_TYPE'].count().sort_values(ascending=False).plot(kind='barh')
plt.show()

train_df.groupby(['NAME_HOUSING_TYPE'])['NAME_HOUSING_TYPE'].count().sort_values(ascending=False).plot(kind='barh')
plt.show()

train_df.groupby(['NAME_FAMILY_STATUS'])['NAME_FAMILY_STATUS'].count().sort_values(ascending=False).plot(kind='barh')
plt.show()

train_df.groupby(['NAME_TYPE_SUITE'])['NAME_TYPE_SUITE'].count().sort_values(ascending=False).plot(kind='barh')
plt.show()
```


![png](/img/homecredit_17_0.png)



![png](/img/homecredit_17_1.png)



![png](/img/homecredit_17_2.png)



![png](/img/homecredit_17_3.png)



![png](/img/homecredit_17_4.png)



```python
sns.distplot(train_df['DAYS_BIRTH']/365)
plt.show()

sns.distplot(train_df['DAYS_EMPLOYED']/365)
plt.show()

sns.distplot(train_df['DAYS_REGISTRATION']/365)
plt.show()

sns.distplot(train_df['DAYS_ID_PUBLISH']/365)
plt.show()
```


![png](/img/homecredit_18_0.png)



![png](/img/homecredit_18_1.png)



![png](/img/homecredit_18_2.png)



![png](/img/homecredit_18_3.png)


### Correlation across dataset

I don't particularly feel it advantageous to examine every aspect of this dataset. So we can cheat a bit and see if there are any interesting correlation patterns across the numeric datatypes


```python
corrs = train_df.corr()
```


```python
plt.figure(figsize=(20,20))
sns.heatmap(corrs)
plt.show()
```


![png](/img/homecredit_21_0.png)


We can see a nice cluster of highly correlated features, mostly pertaining to aspects of the property (not unexpected!).


```python
plt.figure(figsize=(20,20))
sns.clustermap(corrs.dropna())
plt.show()
```


    <matplotlib.figure.Figure at 0x25fcc729cc0>



![png](/img/homecredit_23_1.png)


## Prediction! What we care about!
First we'll want to clean up some of the non-numeric data... and then just run some incredibly basic models to demonstrate how to produce a submission for the competition!


```python

#Preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Algos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Postprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
```


```python
cat_f = [x for x in train_df.columns if train_df[x].dtype == 'object']

for name in cat_f:
    enc = preprocessing.LabelEncoder()
    enc.fit(list(train_df[name].values.astype('str')) + list(test_df[name].values.astype('str')))
    test_df[name] = enc.transform(test_df[name].values.astype('str'))
    train_df[name] = enc.transform(train_df[name].values.astype('str'))
```


```python
X_train = train_df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y_train = train_df['TARGET']

X_train.fillna(-1000, inplace=True) # hopefully ok...
# our test dataset doesn't have a target variable, so we'll have to test on the train df using holdout
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

#X_test = test_df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
#y_test = test_df['TARGET']

clf = LogisticRegression()
clf.fit(x_train, y_train)
print("Logistic Regr. Score = ", clf.score(x_test, y_test))
```

    C:\Users\Clint_PC\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

    Logistic Regr. Score =  0.9197274929678227
    


```python
clf3 = XGBClassifier()
clf3.fit(x_train, y_train)
print("XGBoost Score = ", clf3.score(x_test, y_test))
```

    XGBoost Score =  0.9199876428792091
    

    C:\Users\Clint_PC\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    


```python
clf4 = KNeighborsClassifier()
clf4.fit(x_train, y_train)
print("KNN Score = ", clf4.score(x_test, y_test))
```

    KNN Score =  0.9136952668975498
    


```python
clf5 = RandomForestClassifier()
clf5.fit(x_train, y_train)
print("Random Forest Score = ", clf5.score(x_test, y_test))
```

    Random Forest Score =  0.9184755215192755
    


```python
ax = plot_importance(clf3)
fig = ax.figure
fig.set_size_inches(15, 10)
plt.show()
```


![png](/img/homecredit_31_0.png)



```python
# select features using threshold
selection = SelectFromModel(clf3, threshold=0.05, prefit=True)
select_X_train = selection.transform(x_train)
# train model
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
# eval model
X_test = test_df.fillna(-1000)
select_X_test = selection.transform(X_test.drop(['SK_ID_CURR'], axis=1))
y_pred = selection_model.predict(select_X_test)

```

    C:\Users\Clint_PC\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    


```python
y_pred = selection_model.predict_proba(select_X_test)
y_pred = pd.DataFrame(y_pred)
submission = pd.DataFrame()
submission['SK_ID_CURR'] = test_df['SK_ID_CURR']
submission['TARGET'] = y_pred.iloc[:, 1]
submission.to_csv('submission.csv', index=False)
```


```python
submission.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100001</td>
      <td>0.041684</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100005</td>
      <td>0.081148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100013</td>
      <td>0.030166</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100028</td>
      <td>0.046527</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100038</td>
      <td>0.148846</td>
    </tr>
  </tbody>
</table>
</div>


