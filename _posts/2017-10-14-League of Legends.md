---
layout:     post
title:      "Exploration and Prediction on League of Legends Match Data"
subtitle:   "I investigate LoL match data and apply some basic machine learning techniques to predict a winner."
date:       2017-10-14 12:00:00
author:     "Clint Howard"
category: Portfolio
tags: [python, data, gaming]
comments: True
---
# League of Legends Exploration

An incredibly popular RTS game, League of Legends is an interesting combination of player mechanics, individual skill and teamwork. Here we investigate a dataset of LoL matches and see whether we can use any of it to predict a winner!

## Pre-Processing


```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json as json

import seaborn as sns
```


```python
df = pd.read_csv(r"D:\Downloads\Coding\league-of-legends\games.csv")
champions = pd.read_json(r"D:\Downloads\Coding\league-of-legends\champion_info.json")
spells = pd.read_json(r"D:\Downloads\Coding\league-of-legends\summoner_spell_info.json")
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>gameDuration</th>
      <th>seasonId</th>
      <th>winner</th>
      <th>firstBlood</th>
      <th>firstTower</th>
      <th>firstInhibitor</th>
      <th>firstBaron</th>
      <th>firstDragon</th>
      <th>firstRiftHerald</th>
      <th>...</th>
      <th>t2_towerKills</th>
      <th>t2_inhibitorKills</th>
      <th>t2_baronKills</th>
      <th>t2_dragonKills</th>
      <th>t2_riftHeraldKills</th>
      <th>t2_ban1</th>
      <th>t2_ban2</th>
      <th>t2_ban3</th>
      <th>t2_ban4</th>
      <th>t2_ban5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3326086514</td>
      <td>1949</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>114</td>
      <td>67</td>
      <td>43</td>
      <td>16</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3229566029</td>
      <td>1851</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>67</td>
      <td>238</td>
      <td>51</td>
      <td>420</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3327363504</td>
      <td>1493</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>157</td>
      <td>238</td>
      <td>121</td>
      <td>57</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3326856598</td>
      <td>1758</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>164</td>
      <td>18</td>
      <td>141</td>
      <td>40</td>
      <td>51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3330080762</td>
      <td>2094</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>86</td>
      <td>11</td>
      <td>201</td>
      <td>122</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51536 entries, 0 to 51535
    Data columns (total 60 columns):
    gameId                51536 non-null int64
    gameDuration          51536 non-null int64
    seasonId              51536 non-null int64
    winner                51536 non-null int64
    firstBlood            51536 non-null int64
    firstTower            51536 non-null int64
    firstInhibitor        51536 non-null int64
    firstBaron            51536 non-null int64
    firstDragon           51536 non-null int64
    firstRiftHerald       51536 non-null int64
    t1_champ1id           51536 non-null int64
    t1_champ1_sum1        51536 non-null int64
    t1_champ1_sum2        51536 non-null int64
    t1_champ2id           51536 non-null int64
    t1_champ2_sum1        51536 non-null int64
    t1_champ2_sum2        51536 non-null int64
    t1_champ3id           51536 non-null int64
    t1_champ3_sum1        51536 non-null int64
    t1_champ3_sum2        51536 non-null int64
    t1_champ4id           51536 non-null int64
    t1_champ4_sum1        51536 non-null int64
    t1_champ4_sum2        51536 non-null int64
    t1_champ5id           51536 non-null int64
    t1_champ5_sum1        51536 non-null int64
    t1_champ5_sum2        51536 non-null int64
    t1_towerKills         51536 non-null int64
    t1_inhibitorKills     51536 non-null int64
    t1_baronKills         51536 non-null int64
    t1_dragonKills        51536 non-null int64
    t1_riftHeraldKills    51536 non-null int64
    t1_ban1               51536 non-null int64
    t1_ban2               51536 non-null int64
    t1_ban3               51536 non-null int64
    t1_ban4               51536 non-null int64
    t1_ban5               51536 non-null int64
    t2_champ1id           51536 non-null int64
    t2_champ1_sum1        51536 non-null int64
    t2_champ1_sum2        51536 non-null int64
    t2_champ2id           51536 non-null int64
    t2_champ2_sum1        51536 non-null int64
    t2_champ2_sum2        51536 non-null int64
    t2_champ3id           51536 non-null int64
    t2_champ3_sum1        51536 non-null int64
    t2_champ3_sum2        51536 non-null int64
    t2_champ4id           51536 non-null int64
    t2_champ4_sum1        51536 non-null int64
    t2_champ4_sum2        51536 non-null int64
    t2_champ5id           51536 non-null int64
    t2_champ5_sum1        51536 non-null int64
    t2_champ5_sum2        51536 non-null int64
    t2_towerKills         51536 non-null int64
    t2_inhibitorKills     51536 non-null int64
    t2_baronKills         51536 non-null int64
    t2_dragonKills        51536 non-null int64
    t2_riftHeraldKills    51536 non-null int64
    t2_ban1               51536 non-null int64
    t2_ban2               51536 non-null int64
    t2_ban3               51536 non-null int64
    t2_ban4               51536 non-null int64
    t2_ban5               51536 non-null int64
    dtypes: int64(60)
    memory usage: 23.6 MB
    


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>gameDuration</th>
      <th>seasonId</th>
      <th>winner</th>
      <th>firstBlood</th>
      <th>firstTower</th>
      <th>firstInhibitor</th>
      <th>firstBaron</th>
      <th>firstDragon</th>
      <th>firstRiftHerald</th>
      <th>...</th>
      <th>t2_towerKills</th>
      <th>t2_inhibitorKills</th>
      <th>t2_baronKills</th>
      <th>t2_dragonKills</th>
      <th>t2_riftHeraldKills</th>
      <th>t2_ban1</th>
      <th>t2_ban2</th>
      <th>t2_ban3</th>
      <th>t2_ban4</th>
      <th>t2_ban5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.153600e+04</td>
      <td>51536.000000</td>
      <td>51536.0</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>...</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
      <td>51536.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.306218e+09</td>
      <td>1832.433658</td>
      <td>9.0</td>
      <td>0.493577</td>
      <td>0.507141</td>
      <td>0.502134</td>
      <td>0.447765</td>
      <td>0.286654</td>
      <td>0.479471</td>
      <td>0.251416</td>
      <td>...</td>
      <td>5.549849</td>
      <td>0.985078</td>
      <td>0.414565</td>
      <td>1.404397</td>
      <td>0.240220</td>
      <td>108.203605</td>
      <td>107.957991</td>
      <td>108.686666</td>
      <td>108.636196</td>
      <td>108.081031</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.946137e+07</td>
      <td>511.935772</td>
      <td>0.0</td>
      <td>0.499964</td>
      <td>0.499954</td>
      <td>0.500000</td>
      <td>0.497269</td>
      <td>0.452203</td>
      <td>0.499583</td>
      <td>0.433832</td>
      <td>...</td>
      <td>3.860701</td>
      <td>1.256318</td>
      <td>0.613800</td>
      <td>1.224289</td>
      <td>0.427221</td>
      <td>102.538299</td>
      <td>102.938916</td>
      <td>102.592143</td>
      <td>103.356702</td>
      <td>102.762418</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.214824e+09</td>
      <td>190.000000</td>
      <td>9.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.292212e+09</td>
      <td>1531.000000</td>
      <td>9.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>37.000000</td>
      <td>38.000000</td>
      <td>38.000000</td>
      <td>38.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.319969e+09</td>
      <td>1833.000000</td>
      <td>9.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>90.000000</td>
      <td>90.000000</td>
      <td>90.000000</td>
      <td>90.000000</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.327097e+09</td>
      <td>2148.000000</td>
      <td>9.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
      <td>141.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.331833e+09</td>
      <td>4728.000000</td>
      <td>9.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>10.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>516.000000</td>
      <td>516.000000</td>
      <td>516.000000</td>
      <td>516.000000</td>
      <td>516.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>



## EDA

We can perform some basic manipulations to get an insight into the frequency with which differenct champions & summoner spells are selected, as well as the differences in selection between the two teams. 

* Most common champion picks
* Most common team picks
* Most common pairs
* Most common spells
* Most common champion + spells


```python
# Convert from an ID to a name
champ_names11 = [champions.ix[y][0]['name'] for y in df.t1_champ1id]
champ_names12 = [champions.ix[y][0]['name'] for y in df.t1_champ2id]
champ_names13 = [champions.ix[y][0]['name'] for y in df.t1_champ3id]
champ_names14 = [champions.ix[y][0]['name'] for y in df.t1_champ4id]
champ_names15 = [champions.ix[y][0]['name'] for y in df.t1_champ5id]

champ_names21 = [champions.ix[y][0]['name'] for y in df.t2_champ1id]
champ_names22 = [champions.ix[y][0]['name'] for y in df.t2_champ2id]
champ_names23 = [champions.ix[y][0]['name'] for y in df.t2_champ3id]
champ_names24 = [champions.ix[y][0]['name'] for y in df.t2_champ4id]
champ_names25 = [champions.ix[y][0]['name'] for y in df.t2_champ5id]
```


```python
c11 = pd.DataFrame(np.column_stack([champ_names11, np.repeat(1, len(df)), np.repeat(1, len(df))]), 
            columns=["name", "team", "champ"])
c12 = pd.DataFrame(np.column_stack([champ_names12, np.repeat(1, len(df)), np.repeat(2, len(df))]), 
            columns=["name", "team", "champ"])
c13 = pd.DataFrame(np.column_stack([champ_names13, np.repeat(1, len(df)), np.repeat(3, len(df))]), 
            columns=["name", "team", "champ"])
c14 = pd.DataFrame(np.column_stack([champ_names14, np.repeat(1, len(df)), np.repeat(4, len(df))]), 
            columns=["name", "team", "champ"])
c15 = pd.DataFrame(np.column_stack([champ_names15, np.repeat(1, len(df)), np.repeat(5, len(df))]), 
            columns=["name", "team", "champ"])

c21 = pd.DataFrame(np.column_stack([champ_names21, np.repeat(2, len(df)), np.repeat(1, len(df))]), 
            columns=["name", "team", "champ"])
c22 = pd.DataFrame(np.column_stack([champ_names22, np.repeat(2, len(df)), np.repeat(2, len(df))]), 
            columns=["name", "team", "champ"])
c23 = pd.DataFrame(np.column_stack([champ_names23, np.repeat(2, len(df)), np.repeat(3, len(df))]), 
            columns=["name", "team", "champ"])
c24 = pd.DataFrame(np.column_stack([champ_names24, np.repeat(2, len(df)), np.repeat(4, len(df))]), 
            columns=["name", "team", "champ"])
c25 = pd.DataFrame(np.column_stack([champ_names25, np.repeat(2, len(df)), np.repeat(5, len(df))]), 
            columns=["name", "team", "champ"])

```


```python
comb_names = pd.DataFrame(np.vstack([c11, c12, c13, c14, c15, c21, c22, c23, c24, c25]),
                         columns=["name", "team", "champ"])
```


```python
# Convert from an ID to a name
spells_names111 = [spells.ix[y][0]['name'] for y in df.t1_champ1_sum1]
spells_names112 = [spells.ix[y][0]['name'] for y in df.t1_champ1_sum2]
spells_names121 = [spells.ix[y][0]['name'] for y in df.t1_champ2_sum1]
spells_names122 = [spells.ix[y][0]['name'] for y in df.t1_champ2_sum2]
spells_names131 = [spells.ix[y][0]['name'] for y in df.t1_champ3_sum1]
spells_names132 = [spells.ix[y][0]['name'] for y in df.t1_champ3_sum2]
spells_names141 = [spells.ix[y][0]['name'] for y in df.t1_champ4_sum1]
spells_names142 = [spells.ix[y][0]['name'] for y in df.t1_champ4_sum2]
spells_names151 = [spells.ix[y][0]['name'] for y in df.t1_champ5_sum1]
spells_names152 = [spells.ix[y][0]['name'] for y in df.t1_champ5_sum2]

spells_names211 = [spells.ix[y][0]['name'] for y in df.t2_champ1_sum1]
spells_names212 = [spells.ix[y][0]['name'] for y in df.t2_champ1_sum2]
spells_names221 = [spells.ix[y][0]['name'] for y in df.t2_champ2_sum1]
spells_names222 = [spells.ix[y][0]['name'] for y in df.t2_champ2_sum2]
spells_names231 = [spells.ix[y][0]['name'] for y in df.t2_champ3_sum1]
spells_names232 = [spells.ix[y][0]['name'] for y in df.t2_champ3_sum2]
spells_names241 = [spells.ix[y][0]['name'] for y in df.t2_champ4_sum1]
spells_names242 = [spells.ix[y][0]['name'] for y in df.t2_champ4_sum2]
spells_names251 = [spells.ix[y][0]['name'] for y in df.t2_champ5_sum1]
spells_names252 = [spells.ix[y][0]['name'] for y in df.t2_champ5_sum2]
```


```python
t1s1 = pd.DataFrame(np.column_stack([spells_names111, spells_names112, np.repeat(1, len(df)), np.repeat(1, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t1s2 = pd.DataFrame(np.column_stack([spells_names121, spells_names122, np.repeat(1, len(df)), np.repeat(2, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t1s3 = pd.DataFrame(np.column_stack([spells_names131, spells_names132, np.repeat(1, len(df)), np.repeat(3, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t1s4 = pd.DataFrame(np.column_stack([spells_names141, spells_names142, np.repeat(1, len(df)), np.repeat(4, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t1s5 = pd.DataFrame(np.column_stack([spells_names151, spells_names152, np.repeat(1, len(df)), np.repeat(5, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])

t2s1 = pd.DataFrame(np.column_stack([spells_names211, spells_names212, np.repeat(2, len(df)), np.repeat(1, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t2s2 = pd.DataFrame(np.column_stack([spells_names221, spells_names222, np.repeat(2, len(df)), np.repeat(2, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t2s3 = pd.DataFrame(np.column_stack([spells_names231, spells_names232, np.repeat(2, len(df)), np.repeat(3, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t2s4 = pd.DataFrame(np.column_stack([spells_names241, spells_names242, np.repeat(2, len(df)), np.repeat(4, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
t2s5 = pd.DataFrame(np.column_stack([spells_names251, spells_names252, np.repeat(2, len(df)), np.repeat(5, len(df))]), 
            columns=["sum1", "sum2", "team", "champ"])
```


```python
comb_spells = pd.DataFrame(np.vstack([t1s1, t1s2, t1s3, t1s4, t1s5,
                                     t2s1, t2s2, t2s3, t2s4, t2s5]), 
                          columns=["sum1", "sum2", "team", "champ"])
```


```python
# Aggregate all info into a single dataframe
all_names = pd.merge(comb_names, comb_spells, left_index=True, right_index=True)
all_names = all_names.drop(["team_y", "champ_y"], axis=1)
```

### Champion Selections
Here we map out both the difference in selections between the two teams, as well as the total frequency of champion selection. 

Interestingly we see that Janna and Xayah have the most dispairty between the two teams. Perhaps they are both good counters to each other and so if one team sees the other being chosen, the opposing team counters. In the middle of the pack with little to no difference we see picks such as Lucian, LeBlanc, Karma, Soraka and Yasuo.


```python
plt.figure(figsize=(15,10))
test = all_names.groupby(["name", "team_x"])["champ_x"].count()
test = test.unstack()
test["diff"] = test.ix[:, 0] - test.ix[:, 1]
test["diff"].sort_values(ascending=False).plot(kind="bar")
plt.show()
```


![png](/img/leaguelegends_16_0.png)


In terms of total frequency, we see Thresh and Tristana fighting it out for first place, with them being followed up by Vayne, Kayn and Lee Sin. No surprises here given the popularity of these champions.


```python
plt.figure(figsize=(20,10))
all_names.groupby(["name"])["name"].count().sort_values(ascending=False).plot(kind="bar")
plt.ylabel("Frequency")
plt.xlabel("Champion Name")
plt.title("Frequency of Champion Selection")
plt.show()
```


![png](/img/leaguelegends_18_0.png)


### Summoner Spells

We conduct a similar process as above, and exmaine the most frequently used summoner spells and pairings with champions. Not unsurprisingly Flash tops the list in both cases. It looks like Ghost, Barrier and Cleanse are seen as the worst performing spells and rarely picked.


```python
all_names.groupby(["sum1"])["sum1"].count().sort_values(ascending=False).plot(kind="bar")
plt.xlabel("Spell 1")
plt.ylabel("Frequency")
plt.show()

all_names.groupby(["sum2"])["sum2"].count().sort_values(ascending=False).plot(kind="bar")
plt.xlabel("Spell 2")
plt.ylabel("Frequency")
plt.show()
```


![png](/img/leaguelegends_20_0.png)



![png](/img/leaguelegends_20_1.png)



```python
sum1_diff = all_names.groupby(["sum1", "team_x"])["sum1"].count().unstack()
sum1_diff["diff"] = sum1_diff.ix[:,0] - sum1_diff.ix[:,1]
sum1_diff["diff"].plot(kind="bar")
plt.ylabel("Difference Between Team 1 and Team 2")
plt.xlabel("Spell 1")
plt.show()

sum2_diff = all_names.groupby(["sum2", "team_x"])["sum2"].count().unstack()
sum2_diff["diff"] = sum2_diff.ix[:,0] - sum2_diff.ix[:,1]
sum2_diff["diff"].plot(kind="bar")
plt.ylabel("Difference Between Team 1 and Team 2")
plt.xlabel("Spell 2")
plt.show()
```


![png](/img/leaguelegends_21_0.png)



![png](/img/leaguelegends_21_1.png)


An interesting question is what are the most common spell selections for each champion. We expect that there would be considerable variation in spell selections, depending on both individual playstyle as well as the category that the champion falls into. To do this, we can simply aggregate up all the champions & spells, and then normalise across each champion to get the ratios that each one selects.

Some points of interest:
* Average ratio of flash is ~55-60%
* Hecarim has a substantially lower ratio of flash, given his special which allows him to escape quickly
* Olaf also has a substantially lower ratio of flash
* Also appears to be a fairly consistent ratio of the second preferred spell across most champions


```python
champ_spell_comb = all_names.groupby(["name", "sum1"])["name"].count().unstack().fillna(0)
champ_spell_comb = champ_spell_comb.div(champ_spell_comb.sum(axis=1), axis=0) # normalising
```


```python
champ_spell_comb.ix[0:20, :].plot(kind="bar", figsize=(15,10))
plt.ylabel("Fraction of Selections")
plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
plt.show()

champ_spell_comb.ix[21:40, :].plot(kind="bar", figsize=(15,10))
plt.ylabel("Fraction of Selections")
plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
plt.show()

champ_spell_comb.ix[41:60, :].plot(kind="bar", figsize=(15,10))
plt.ylabel("Fraction of Selections")
plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
plt.show()

champ_spell_comb.ix[61:80, :].plot(kind="bar", figsize=(15,10))
plt.ylabel("Fraction of Selections")
plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
plt.show()

champ_spell_comb.ix[81:100, :].plot(kind="bar", figsize=(15,10))
plt.ylabel("Fraction of Selections")
plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
plt.show()

champ_spell_comb.ix[101:128, :].plot(kind="bar", figsize=(15,10))
plt.ylabel("Fraction of Selections")
plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
plt.show()
```


![png](/img/leaguelegends_24_0.png)



![png](/img/leaguelegends_24_1.png)



![png](/img/leaguelegends_24_2.png)



![png](/img/leaguelegends_24_3.png)



![png](/img/leaguelegends_24_4.png)



![png](/img/leaguelegends_24_5.png)


## Predictive Analysis

Now that we've pulled apart the data, what's more interesting is seeing whether we can use it to predict the winner of a game.

From the heatmap below, we see that there are very obvious clusters of correlated variables:
* First game objectives
* Total number of objective kills

These are not unsurprising as a team who is doing good/bad will likely of opposing levels of objectives/kills.


```python
corrmat = df.corr()
plt.figure(figsize = (15,10))
sns.heatmap(corrmat, square=True)
plt.show()
```


![png](/img/leaguelegends_26_0.png)



```python
cols = ['winner','t1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't2_towerKills', 't2_inhibitorKills', 't2_baronKills',]
```


```python
sns.pairplot(df[cols])
plt.show()
```


![png](/img/leaguelegends_28_0.png)


### Structural Analysis

We can do some basic PCA and K-Means clustering to see if there is any obvious structure which exists in our dataset. We should also be clear that we are working with the dataset that does include endgame statistics... i.e. we should expect fairly high predictive strength. We will investigate later a prediction using variables only known at the start of a match.


```python
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df.drop(['gameId', 'seasonId', 'winner'], 1)
y = df['winner']

X_std = StandardScaler().fit_transform(X)
n_components = 5
X_std = X_std[:2000]
pca = PCA(n_components=n_components).fit(X_std)
Target = y[:2000]
X_5d = pca.transform(X_std)
```


```python
trace0 = go.Scatter(
    x = X_5d[:,0],
    y = X_5d[:,1],
    name = Target,
    hoveron = Target,
    mode = 'markers',
    text = Target,
    showlegend = False,
    marker = dict(
        size = 8,
        color = Target,
        colorscale ='Jet',
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
        opacity = 0.8
    )
)
data = [trace0]

layout = go.Layout(
    title= 'Principal Component Analysis (PCA)',
    hovermode= 'closest',
    xaxis= dict(
         title= 'First Principal Component',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Second Principal Component',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')
```

![png](/img/leaguelegends_py1.png)


```python
from sklearn.cluster import KMeans # KMeans clustering 
kmeans = KMeans(n_clusters=2)
X_clustered = kmeans.fit_predict(X_5d)

trace_Kmeans = go.Scatter(x=X_5d[:, 0], y= X_5d[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(
                            size=8,
                            color = X_clustered,
                            colorscale = 'Portland',
                            showscale=False, 
                            line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        )
                   ))

layout = go.Layout(
    title= 'KMeans Clustering',
    hovermode= 'closest',
    xaxis= dict(
         title= 'First Principal Component',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Second Principal Component',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)

data = [trace_Kmeans]
fig1 = dict(data=data, layout= layout)
# fig1.append_trace(contour_list)
py.iplot(fig1, filename="svm")
```


![png](/img/leaguelegends_py1.png)



```python
from sklearn.manifold import TSNE

tsne = TSNE()
tsne_results = tsne.fit_transform(X_std) 

traceTSNE = go.Scatter(
    x = tsne_results[:,0],
    y = tsne_results[:,1],
    name = Target,
     hoveron = Target,
    mode = 'markers',
    text = Target,
    showlegend = True,
    marker = dict(
        size = 8,
        color = Target,
        colorscale ='Jet',
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
        opacity = 0.8
    )
)
data = [traceTSNE]

layout = dict(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',
              hovermode= 'closest',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False),
              showlegend= False,

             )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')
```


![png](/img/leaguelegends_py1.png)


The above three techniques demonstrate that there is a pretty clear structural differentiation between winning and losing. This bodes well for running some machine learning techniques across the dataset.

As seen below, the scores we get are all in excess of 90%. This is not unexpected given the level of data which is available and the fact that it is end game data.


```python
#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Algos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Postprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from xgboost import plot_importance

X_std = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y)
clf = LogisticRegression()
clf.fit(X_train, y_train)

clf2 = SVC(kernel="linear", C=0.025)
clf2.fit(X_train, y_train)

clf3 = XGBClassifier()
clf3.fit(X_train, y_train)

cl4 = KNeighborsClassifier()
cl4.fit(X_train, y_train)

print("Logistic Regr. Score = ", clf.score(X_test, y_test))
print("SVC Linear Kernel Score = ", clf2.score(X_test, y_test))
print("XGBoost Score = ", clf3.score(X_test, y_test))
print("KNN Score = ", cl4.score(X_test, y_test))
```

    Logistic Regr. Score =  0.961192176343
    SVC Linear Kernel Score =  0.96010555728
    XGBoost Score =  0.96763427507
    KNN Score =  0.924712822105
    

### Feature Importance

The basic question to answer is... what is driving these high scores? What features in our dataset are allowing such high predictions?

The answer is ... tower kills and inhibitor kills. Not unsurprisingly, the team which destroys the most towers and most inhibitors wins. Given that this is the prime objective of the game, this is not unexpected.


```python
plt.bar(range(len(clf3.feature_importances_)), clf3.feature_importances_)
plt.xlabel("Feature")
plt.ylabel("Relative Importance")
plt.show()

plot_importance(clf3)
plt.show()
```


![png](/img/leaguelegends_37_0.png)



![png](/img/leaguelegends_37_1.png)



```python
selection = SelectFromModel(clf3, threshold=0.05, prefit=True)
X_train_sel = selection.transform(X_train)
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
X_test_sel = selection.transform(X_test)
y_pred = selection_model.predict(X_test_sel)

predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (0.05, X_train_sel.shape[1], accuracy*100.0))
```

    Thresh=0.050, n=4, Accuracy: 96.05%
    


```python
X_new = pd.DataFrame(X.columns)
X_new['importance'] = clf3.feature_importances_
X_new.sort_values(by='importance', ascending=False).head(15)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>t2_towerKills</td>
      <td>0.195015</td>
    </tr>
    <tr>
      <th>22</th>
      <td>t1_towerKills</td>
      <td>0.190616</td>
    </tr>
    <tr>
      <th>23</th>
      <td>t1_inhibitorKills</td>
      <td>0.107038</td>
    </tr>
    <tr>
      <th>48</th>
      <td>t2_inhibitorKills</td>
      <td>0.068915</td>
    </tr>
    <tr>
      <th>3</th>
      <td>firstInhibitor</td>
      <td>0.049853</td>
    </tr>
    <tr>
      <th>24</th>
      <td>t1_baronKills</td>
      <td>0.033724</td>
    </tr>
    <tr>
      <th>49</th>
      <td>t2_baronKills</td>
      <td>0.027859</td>
    </tr>
    <tr>
      <th>13</th>
      <td>t1_champ3id</td>
      <td>0.021994</td>
    </tr>
    <tr>
      <th>16</th>
      <td>t1_champ4id</td>
      <td>0.020528</td>
    </tr>
    <tr>
      <th>19</th>
      <td>t1_champ5id</td>
      <td>0.020528</td>
    </tr>
    <tr>
      <th>50</th>
      <td>t2_dragonKills</td>
      <td>0.017595</td>
    </tr>
    <tr>
      <th>2</th>
      <td>firstTower</td>
      <td>0.017595</td>
    </tr>
    <tr>
      <th>51</th>
      <td>t2_riftHeraldKills</td>
      <td>0.016129</td>
    </tr>
    <tr>
      <th>0</th>
      <td>gameDuration</td>
      <td>0.016129</td>
    </tr>
    <tr>
      <th>54</th>
      <td>t2_ban3</td>
      <td>0.014663</td>
    </tr>
  </tbody>
</table>
</div>



### Starting Variables

As stated above, the previous analysis was done on variables which are known at the end of the game. We would expect potentially no predictability given the starting variables. As seen below, this is the case. If we only take variables that are known at the start such as selected champions, summonr spells & team bans, the predictability drops to no better than a coin flip. 

This highlights, perhaps, why these RTS games are so popular... no matter what combination of starting variables, the outcome is still highly dependent on individual player skill and how the team comes together. The actual starting selections bare only a marginal impact on the end results.


```python
X_reduced = df.iloc[:, np.r_[10:24, 30:49, 55:59]]
X_reduced = StandardScaler().fit_transform(X_reduced)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y)

clf = LogisticRegression()
clf.fit(X_train, y_train)

clf3 = XGBClassifier()
clf3.fit(X_train, y_train)

print("Logistic Regr. Score = ", clf.score(X_test, y_test))
print("XGBoost Score = ", clf3.score(X_test, y_test))
```

    Logistic Regr. Score =  0.515135051226
    XGBoost Score =  0.519403911829
    
