---
layout:     post
title:      "Scraping and Exploration of NSW Food Authority Penalty Notices"
subtitle:   "I implement a web scraper to obtain some NSW Government data and then proceed to find out which areas and which businesses are the worst serial offenders."
date:       2017-04-08 12:00:00
author:     "Clint Howard"
category: Portfolio
tags: [python, data, food]
comments: True
---


# Scraping and Exploration of NSW Food Authority Register of Penalty Notices

There's been some recent noise around the NSW Department of Primary Industries Food Authority's naming and shaming of restaurants [here](http://www.foodauthority.nsw.gov.au/penalty-notices/default.aspx?template=results). As you can see, there's a nice embedded table so let's so how we can scrape out this information and do some exploration.

[Juypter notebooks and data](https://github.com/clinthoward/Blog-Notebooks/tree/master/NSW%20Food%20Authority)

```python
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
plt.style.use("ggplot")
```

We're going to use BeautifulSoup to scrape the table data. Upon inspection we can see that this should be fairly easy, as the HTML is fairly standard. We'll set up a skeleton dataframe and populate this.


```python
url = "http://www.foodauthority.nsw.gov.au/penalty-notices/default.aspx?template=results"

r = requests.get(url)
soup = BeautifulSoup(r.text, "lxml")      
table = soup.find_all("table")

df = pd.DataFrame(columns = ['trade_name', 'suburb', 'council', 'penalty_no',
                             'date', 'party_served', 'desc'])

```

The way our scraper will work is it will iterate through all tables, then iterate through all rows and columns. It will effectively write each row into a row of our dataframe.


```python
for t in table:
    table_body = t.find('tbody')
    try:
        rows = table_body.find_all('tr')
        for tr in rows:
            temp = []
            try:
                cols = tr.find_all('td')
                href = cols[3].find_all('a')
                temp.append(href[0].get('title'))
                for col in cols:
                    try:
                        temp.append(col.string.strip())
                    except:
                        pass
                
                df = df.append(pd.Series(temp, df.columns), ignore_index = True)
            
            except:
                print("Error reading row.")
                
        
    except:
        print("Error reading table body.")
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trade_name</th>
      <th>suburb</th>
      <th>council</th>
      <th>penalty_no</th>
      <th>date</th>
      <th>party_served</th>
      <th>desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fail to transport potentially hazardous food u...</td>
      <td>(NO TRADING NAME)</td>
      <td>CHATSWOOD</td>
      <td>Willoughby</td>
      <td>3136844078</td>
      <td>2016-08-10</td>
      <td>CHOI, JUNG DAE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fail to comply with a condition of licence - h...</td>
      <td>(NO TRADING NAME)</td>
      <td>NAROOMA</td>
      <td>Eurobodalla</td>
      <td>3120777249</td>
      <td>2016-01-05</td>
      <td>DON MANUWELLGE DON, ANOMA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sale of unsafe food - sushi</td>
      <td>(NO TRADING NAME)</td>
      <td>RYDALMERE</td>
      <td>Parramatta</td>
      <td>3013503947</td>
      <td>2016-01-19</td>
      <td>PARK, JI YOUNG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fail to maintain the food premises to the requ...</td>
      <td>3 CHIMNEYS</td>
      <td>WOLLONGONG</td>
      <td>Wollongong</td>
      <td>3132275255</td>
      <td>2016-05-20</td>
      <td>DRB 56 PTY LTD</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fail to maintain all fixtures, fittings and eq...</td>
      <td>3 CHIMNEYS</td>
      <td>WOLLONGONG</td>
      <td>Wollongong</td>
      <td>3132275264</td>
      <td>2016-05-20</td>
      <td>DRB 56 PTY LTD</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1504 entries, 0 to 1503
    Data columns (total 7 columns):
    trade_name      1504 non-null object
    suburb          1504 non-null object
    council         1504 non-null object
    penalty_no      1504 non-null object
    date            1504 non-null object
    party_served    1504 non-null object
    desc            1504 non-null object
    dtypes: object(7)
    memory usage: 82.3+ KB
    


```python
df.columns = ["desc", "trade_name", "suburb", "council", "number", "date", "party_served"]
df.number = np.int64(df.number)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>desc</th>
      <th>trade_name</th>
      <th>suburb</th>
      <th>council</th>
      <th>number</th>
      <th>date</th>
      <th>party_served</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fail to transport potentially hazardous food u...</td>
      <td>(NO TRADING NAME)</td>
      <td>CHATSWOOD</td>
      <td>Willoughby</td>
      <td>3136844078</td>
      <td>2016-08-10</td>
      <td>CHOI, JUNG DAE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fail to comply with a condition of licence - h...</td>
      <td>(NO TRADING NAME)</td>
      <td>NAROOMA</td>
      <td>Eurobodalla</td>
      <td>3120777249</td>
      <td>2016-01-05</td>
      <td>DON MANUWELLGE DON, ANOMA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sale of unsafe food - sushi</td>
      <td>(NO TRADING NAME)</td>
      <td>RYDALMERE</td>
      <td>Parramatta</td>
      <td>3013503947</td>
      <td>2016-01-19</td>
      <td>PARK, JI YOUNG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fail to maintain the food premises to the requ...</td>
      <td>3 CHIMNEYS</td>
      <td>WOLLONGONG</td>
      <td>Wollongong</td>
      <td>3132275255</td>
      <td>2016-05-20</td>
      <td>DRB 56 PTY LTD</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fail to maintain all fixtures, fittings and eq...</td>
      <td>3 CHIMNEYS</td>
      <td>WOLLONGONG</td>
      <td>Wollongong</td>
      <td>3132275264</td>
      <td>2016-05-20</td>
      <td>DRB 56 PTY LTD</td>
    </tr>
  </tbody>
</table>
</div>



Looking good. We've scraped out all the data, adjusted the column names and forced the penalty ID to be a number. Interestingly though, on the table we see that each penalty ID links to a seperate page which contains even more information. This will be a little trickier to scrape.


```python
url_2 = "http://www.foodauthority.nsw.gov.au/penalty-notices/default.aspx?template=detail&itemId="

df_2 = pd.DataFrame(columns = ["number", "trade_name", "address", "council",
                                "date_of_offence", "offence_code", "nature",
                                "penalty", "party_served", "date_served",
                                "issuer"])

for id_2 in df["penalty_id"]:
    temp_url = url_2 + id_2
    r = requests.get(temp_url)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find_all("table")
    
    for t in table:
        table_body = t.find('tbody')
        
        try:
            rows = table_body.find_all('tr')
            temp = []
            for tr in rows:
                try:
                    col = tr.find_all('td')
                    column_1 = col[1].string.strip()
                    temp.append(column_1)
                    
                except:
                    print("Error in row.")
        
        
        except:
            print("Error in table body.")
            
    df_2 = df_2.append(pd.Series(temp, df_2.columns), ignore_index = True)


```


```python
df_2.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number</th>
      <th>trade_name</th>
      <th>address</th>
      <th>council</th>
      <th>date_of_offence</th>
      <th>offence_code</th>
      <th>nature</th>
      <th>penalty</th>
      <th>party_served</th>
      <th>date_served</th>
      <th>issuer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3136844078</td>
      <td>(NO TRADING NAME)</td>
      <td>CENTENNIAL AVENUE CHATSWOOD 2067</td>
      <td>Willoughby</td>
      <td>10-08-16</td>
      <td>11338 - Fail to comply with Food Standards Cod...</td>
      <td>Fail to transport potentially hazardous food u...</td>
      <td>$440</td>
      <td>CHOI, JUNG DAE</td>
      <td>11-08-16</td>
      <td>Willoughby City Council</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3120777249</td>
      <td>(NO TRADING NAME)</td>
      <td>2 RIVERSIDE DRIVE NAROOMA 2546</td>
      <td>Eurobodalla</td>
      <td>05-01-16</td>
      <td>11016 - Contravene condition of licence - Indi...</td>
      <td>Fail to comply with a condition of licence - h...</td>
      <td>$660</td>
      <td>DON MANUWELLGE DON, ANOMA</td>
      <td>19-07-16</td>
      <td>NSW Food Authority</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3013503947</td>
      <td>(NO TRADING NAME)</td>
      <td>33 MARY PARADE RYDALMERE 2116</td>
      <td>Parramatta</td>
      <td>19-01-16</td>
      <td>11318 - Sell unsafe food - Individual</td>
      <td>Sale of unsafe food - sushi</td>
      <td>$770</td>
      <td>PARK, JI YOUNG</td>
      <td>19-05-16</td>
      <td>NSW Food Authority</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3132275255</td>
      <td>3 CHIMNEYS</td>
      <td>3/63-65 CROWN STREET WOLLONGONG 2500</td>
      <td>Wollongong</td>
      <td>20-05-16</td>
      <td>11339 - Fail to comply with Food Standards Cod...</td>
      <td>Fail to maintain the food premises to the requ...</td>
      <td>$880</td>
      <td>DRB 56 PTY LTD</td>
      <td>23-05-16</td>
      <td>Wollongong City Council</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3132275264</td>
      <td>3 CHIMNEYS</td>
      <td>3/63-65 CROWN STREET WOLLONGONG 2500</td>
      <td>Wollongong</td>
      <td>20-05-16</td>
      <td>11339 - Fail to comply with Food Standards Cod...</td>
      <td>Fail to maintain all fixtures, fittings and eq...</td>
      <td>$880</td>
      <td>DRB 56 PTY LTD</td>
      <td>23-05-16</td>
      <td>Wollongong City Council</td>
    </tr>
  </tbody>
</table>
</div>



We can see that we've gotten a lot more information this time. Street address, penalty amounts as well as the specific codes which have been breached. Lets combine our two dataframes and simplify them down so we have a nice dataset to work with.


```python
df_full = pd.merge(df, df_2, on = "number", how = "left", suffixes = ("_basic", "full"))
df_full = df_full.drop(["trade_namefull", "councilfull", "date_of_offence", "party_servedfull"], axis = 1)
df_full["penalty"] = df_full["penalty"].str.replace("$", "")
df_full["penalty"] = np.int64(df_full["penalty"].str.replace(",", ""))
df_full.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>desc</th>
      <th>trade_name_basic</th>
      <th>suburb</th>
      <th>council_basic</th>
      <th>number</th>
      <th>date</th>
      <th>party_served_basic</th>
      <th>address</th>
      <th>offence_code</th>
      <th>nature</th>
      <th>penalty</th>
      <th>date_served</th>
      <th>issuer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fail to transport potentially hazardous food u...</td>
      <td>(NO TRADING NAME)</td>
      <td>CHATSWOOD</td>
      <td>Willoughby</td>
      <td>3136844078</td>
      <td>2016-08-10</td>
      <td>CHOI, JUNG DAE</td>
      <td>CENTENNIAL AVENUE CHATSWOOD 2067</td>
      <td>11338 - Fail to comply with Food Standards Cod...</td>
      <td>Fail to transport potentially hazardous food u...</td>
      <td>440</td>
      <td>11-08-16</td>
      <td>Willoughby City Council</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fail to comply with a condition of licence - h...</td>
      <td>(NO TRADING NAME)</td>
      <td>NAROOMA</td>
      <td>Eurobodalla</td>
      <td>3120777249</td>
      <td>2016-01-05</td>
      <td>DON MANUWELLGE DON, ANOMA</td>
      <td>2 RIVERSIDE DRIVE NAROOMA 2546</td>
      <td>11016 - Contravene condition of licence - Indi...</td>
      <td>Fail to comply with a condition of licence - h...</td>
      <td>660</td>
      <td>19-07-16</td>
      <td>NSW Food Authority</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sale of unsafe food - sushi</td>
      <td>(NO TRADING NAME)</td>
      <td>RYDALMERE</td>
      <td>Parramatta</td>
      <td>3013503947</td>
      <td>2016-01-19</td>
      <td>PARK, JI YOUNG</td>
      <td>33 MARY PARADE RYDALMERE 2116</td>
      <td>11318 - Sell unsafe food - Individual</td>
      <td>Sale of unsafe food - sushi</td>
      <td>770</td>
      <td>19-05-16</td>
      <td>NSW Food Authority</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fail to maintain the food premises to the requ...</td>
      <td>3 CHIMNEYS</td>
      <td>WOLLONGONG</td>
      <td>Wollongong</td>
      <td>3132275255</td>
      <td>2016-05-20</td>
      <td>DRB 56 PTY LTD</td>
      <td>3/63-65 CROWN STREET WOLLONGONG 2500</td>
      <td>11339 - Fail to comply with Food Standards Cod...</td>
      <td>Fail to maintain the food premises to the requ...</td>
      <td>880</td>
      <td>23-05-16</td>
      <td>Wollongong City Council</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fail to maintain all fixtures, fittings and eq...</td>
      <td>3 CHIMNEYS</td>
      <td>WOLLONGONG</td>
      <td>Wollongong</td>
      <td>3132275264</td>
      <td>2016-05-20</td>
      <td>DRB 56 PTY LTD</td>
      <td>3/63-65 CROWN STREET WOLLONGONG 2500</td>
      <td>11339 - Fail to comply with Food Standards Cod...</td>
      <td>Fail to maintain all fixtures, fittings and eq...</td>
      <td>880</td>
      <td>23-05-16</td>
      <td>Wollongong City Council</td>
    </tr>
  </tbody>
</table>
</div>



## Exploration!
Time to dig into the details and see what we can find out from this dataset.


```python
(df_full.groupby(["suburb"]).count()["desc"]/df.shape[0]).sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Proportion of Penalties")
plt.show()

df_full.groupby(["council_basic"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Number of Penalties")
plt.show()

df_full.groupby(["trade_name_basic"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Number of Penalties")
plt.show()
```


![png](/img/foodhealth_15_0.png)



![png](/img/foodhealth_15_1.png)



![png](/img/foodhealth_15_2.png)


Coming out strong we see that Chatswood dominates the suburb infringement, whilst Sydney leads the overall council issuance. In terms of total number of penalties, Hokka Hokka leads the way with 14 penalties in the last year!


```python
df_full.groupby(["offence_code"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind='barh')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel("Number of Penalties")
plt.show()

df_full.groupby(["nature"]).count()["date"].sort_values(ascending=False)[0:25].plot(kind='barh')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel("Number of Penalties")
plt.ylabel("Nature of Offence")
plt.show()
```


![png](/img/foodhealth_17_0.png)



![png](/img/foodhealth_17_1.png)


A bit unclear, but we can see that offence code "11339 - Fail to comply with Food Standards Code - Corporation" is the clear winner in number of penalties. Not unexpectedly, restuarant cleanliness and food storage the leaders in infrigements.


```python
df_full.groupby(["date"]).count()["penalty"].cumsum().plot()
plt.ylabel("Cumulative Number of Penalties")
plt.show()

df_full.groupby(["date"]).count()["penalty"].plot()
plt.ylabel("Number of Penalties")
plt.show()
```


![png](/img/foodhealth_19_0.png)



![png](/img/foodhealth_19_1.png)


We see a pretty large spike in penalties occuring early March 2016! Perhaps there was some sort of Government initiative which drove this, or maybe it was just an unusually bad period of the year. Ideally we could get a longer dataset to draw out some better conclusions on trends in penalty notices.


```python
df_full.groupby(["trade_name_basic"]).sum()["penalty"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.xlabel("Total Fines ($)")
plt.show()

temp = df_full.groupby(["trade_name_basic"]).sum()
temp["count"] = df_full.groupby(["trade_name_basic"]).count()["penalty"]

plt.scatter(temp["count"], temp["penalty"])
plt.xlabel("Number of Penalties")
plt.ylabel("Total Fines ($)")
plt.show()

df_full.groupby(["issuer"]).count()["penalty"].sort_values(ascending=False)[0:25].plot(kind="barh")
plt.ylabel("Number of Penalties")
plt.show()
```


![png](/img/foodhealth_21_0.png)



![png](/img/foodhealth_21_1.png)



![png](/img/foodhealth_21_2.png)


As expected, there's a strong lienar trend in the number of penalties received and the total fine amount. We also see that the NSW Food Authority is taking clear lead over the councils in terms of penalty issuance.


We'll close out for now with an attempt at a word cloud...


```python
from wordcloud import WordCloud, STOPWORDS

text = df_full["nature"].to_string()
text = str(text.split())

stopwords = set(STOPWORDS)
stopwords.add("to'")
stopwords.add("the'")

wordcloud = WordCloud(background_color = "white", max_words = 200, 
                      stopwords = stopwords, width = 2600, height = 1400).generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](/img/foodhealth_24_0.png)

