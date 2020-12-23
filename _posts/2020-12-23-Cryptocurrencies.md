---
layout:     post
title:      "Cryptocurrency Asset Pricing"
subtitle:   "We apply empirical asset pricing techniques to study the cross-section of cryptocurrency returns."
date:       2020-12-23 12:00:00
author:     "Clint Howard"
category: Portfolio
tags: [python, data, finance, cryptocurrency, investing]
comments: True
---

# Cryptocurrency - Empirical Asset Pricing

Cryptocurrency is all the rage again. Bitcoin has been on a strong, steady surge over 2020. In addition, factor investing has been under the microscope, with exceptional volatility and weekly claims of "value rebounds". In addition, there is an emerging literature which is applying standard empirical asset pricing techniques to cryptocurrency datasets. What better way to jump on the bandwagon, then combine all of them and setup an infrastructure for crypto asset pricing. 

Disclaimer: this is purely my opinion and for research interests/illustrative purposes. It is in no way investment advice, and may not reflect the views of my employer.

## Data Cleaning and Calculations

We use coinmetrics.io data (https://coinmetrics.io/community-network-data/), this isn't a full dataset but the analysis is equally valid on a larger set of data. 

In asset pricing literature, there is a "zoo" of anomalies which are used to explain the cross-section of stock returns. We can lean on this for a first pass. Naturally, this is assuming that the same theories and logic from financial markets also apply to cryptocurrency markets. This assumption is not too far of a stretch, as ultimately humans are behind most of the decisions and choices being made in crypto markets and thus any behavioural anomalies that manifest in other markets, we could reasonably expect that they may also manifest in crypto markets.

Additionally, there might be crypto specific risk factors/anomalies which could be researched/considered. The primary issue with studying cross-sectional returns for crypto is only having 6 years of data to study. This makes it very difficult to draw meaningful conclusions about statistical validity / existence of any anomalies/risk factors. Rather, this research can simply provide indicationjs of potential factors/behaviours in crypto markets.


### Factors
We broadly categorise our factors into sentiment/momentum, volume and volatility. The factors below all have substantial literatures/seminal papers behind them if interested.

In addition to these base factors, we also construct a "crypto market return". We construct a daily and monthly measure of the crypto market return using both a value-weighted (market cap) and equal-weighted approach. We use data from Kenneth French's data library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) to get risk free rates.

We filter our data from 2014/01/01 to 2020/12/22. We only take coins with a market cap > $1m. 

#### Sentiment/momentum factors
* Market capitalisation on rebalance date
* Closing Price on rebalance date
* Maximum return over previous month
* Minimum return over previous month
* 1month/3month/6month price momentum

#### Volume
* 3 Month average daily value
* Amihud's Illiquidity

#### Volatility
* Price volality
* Idiosyncratic vol
* Kurtosis
* Price lottery
* Beta measures: equity/gold/crypto market
* Idiosyncratic skewness/expected skewness/total skewness

### Papers
https://www.sciencedirect.com/science/article/abs/pii/S026499931931020X   
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3394671

## Dataset breadth

We only have a small fraction of the available coin data. This is primarily because we're using the freely available coinmetrics.io dataset. This will naturally have some biases when looking at cross-sectional return predictability and the existence of asset pricing afctors. Survivorship bias is likely the biggest bias, as well as size bias as you might expect that only the larger/more succesful coins are having their data captured with a sufficient history. 

This is a challenging problem in any empirical asset pricing study. For US equities, the CRSP dataset is the standard used in academic literature, as it has been carefully curated and studied for decades. For other equity/asset markets, globally, there is no such universally accepted dataset. This tends to mean that the academic literature has focused on studying US equities. In recent years, Refinitiv/Thomson Reuters and other vendors have increased access to global equity price and market data, but it requires substantially more cleaning and munging to get into an acceptable format.

![png](/img/cryptosyste_7_1.png)


## Risk vs. return of assets/coins

The first question to ask when looking at any asset tends to be, what have historical returns looked like? The second question, if the returns are high, tends to be, and how "risky" is this asset? The commonly accepted measure of risk is price volatility, which is simply the standard deviation of returns over some time period. This is a fairly naive measure of risk, but is easy to calculate so is a good first approximation. More complicated measures such as semi-variance, value-at-risk (VaR) can also be calculated. You don't want to use one measure of risk, so it's good practice to put together various risk measures and study each of them. Depending on your use case/application, there will be more suitable definitions of risk that you may want to use. For example, high-frequency traders vs long term investors will likely have very different views of what constitutes an acceptable risk to take in the context of the assets they wish to hold.

Our first observation is that since 2014, the cumulative returns of cryptocurrencies have absolutely trounced equity markets. On a log scale, returns to cryptocurrencies make equity markets (proxied by S&P500) look almost no risk/bond-like in their behaviour. 

One sees this and goes "wow, I missed the boat, better dive in now and reallocate my equity investments to crypto". However, you need to consider what your risk tolerance is for investing into such an asset. Whilst the crypto markets have roared at times, they have also had violent and substantial drawdowns. If you had your retirement portfolio invested in this market, and it lost 60-70% of it's value at once, you might be putting off retirement for a while. Common advice is to do your research, understand what you're investing into, don't allocate more than you're willing to lose, and take away your human bias. If you believe that cryptocurrency will replace fiat, why not simply buy a portion of it consistently each month and not try to time the market? 


![png](/img/cryptosyste_11_0.png)


To further expand our study of the nature of these assets, we can look at the Sharpe Ratio, which is effectively a measure of risk-adjusted return. We work in absolute return space (i.e. not relative to a benchmark).

What you'll notice is that on this scale, the behaviour of cryptocurrency doesn't look so outstanding anymore. On a risk-adjusted basis, cryptocurrency looks to be generating behaviour similar to standard asset classes such as gold, equities and currency. 

Perhaps a more interesting observation is how similar the sharpe ratios between crypto, equities and gold has gotten in 2020. This is potentially a feature of the market downturn in March 2020, and the various changes in market behaviour that have occured due to COVID.

![png](/img/cryptosyste_13_0.png)


Another way to view risk and return is to plot them against each other. This arises from standard portfolio theory/efficient frontiers and the like. 

What you expect/tend to see with assets is that higher risk corresponds to higher returns. We can see that this does tend to occur, and you could almost draw the efficient frontier around the coins.

![png](/img/cryptosyste_15_0.png)


![png](/img/cryptosyste_16_0.png)

## Estimates of beta against SP500, Gold & Crypto value-weighted

Another way to study the relationship between asset classes is to regress the returns against each other. This is commonly known as the CAPM regression beta, but it is simply a standard linear model where our independent variable is the return on asset A and our dependent variable is the return on asset B: 

$$ r_a = \alpha + \beta_i r_b + \epsilon $$

What we do is for each coin, we calculate a running beta at the end of each month using the previous 260 daily return values (with a minimum of 200 observations). We can use benchmarks for equities/gold/crypto to calculate the beta of each coin to each benchmark.

We then can simply take the median value at the end of each month and plot this through time. This will give us an indication on the typical beta in the sample.

What we find is not unsurprising. The median beta to our crypto market returns is positive and between 0.6 and 1.2. We see substantially larger and more volatile values in our S&P500/gold betas. More intruiging is the sharp shift in S&P500 beta to around 1-1.2 where it has stayed for almost all of 2020.


![png](/img/cryptosyste_19_0.png)


## Predicting coin returns

We calculate a series of factors, which we can use to try and predict one-month forward stock returns. In a full study, you would look at multiple frequencies but this is mostly illustrative here.

The first common measure is the information coefficient, or the correlation between the factor scores and the one-month forward returns. This gives us an indication for which factors might be useful in predicting future stock returns. However, from an implementation perspective, there's many, many considerations... transaction costs, turnover, short constraints, to name a few.


![png](/img/cryptosyste_23_1.png)


We can also look at the relationship amongst our factors. We can use a dendrogram + heatmap to see if there's any clustered groups of factors. We see fairly expected results with clustering of momentum factors, beta factors, and then our short-term trading factors.


![png](/img/cryptosyste_25_0.png)

A common tool in litearature is to look at summary statistics of your factors/variables. This gives you a sense of the scale, and also potentially highlights any outlier/distribution issues you may have to deal with. 



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>illiq</th>
      <th>skew</th>
      <th>kurt</th>
      <th>max</th>
      <th>min</th>
      <th>vol</th>
      <th>adv</th>
      <th>prc</th>
      <th>mcap</th>
      <th>equitybeta</th>
      <th>ivol</th>
      <th>iskew</th>
      <th>sskew</th>
      <th>tskew</th>
      <th>gold_beta</th>
      <th>cryptobeta</th>
      <th>mom_3</th>
      <th>mom_6</th>
      <th>mom_12</th>
      <th>reversal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>312.8514</td>
      <td>0.1151</td>
      <td>2.1881</td>
      <td>0.1360</td>
      <td>-0.1214</td>
      <td>0.0551</td>
      <td>59,324,704.7266</td>
      <td>0.7705</td>
      <td>18.7713</td>
      <td>0.4090</td>
      <td>1.0698</td>
      <td>0.5461</td>
      <td>12.3359</td>
      <td>0.3822</td>
      <td>0.4906</td>
      <td>0.8962</td>
      <td>0.0683</td>
      <td>0.1390</td>
      <td>0.2855</td>
      <td>0.0207</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1,158.4035</td>
      <td>0.8707</td>
      <td>2.6050</td>
      <td>0.1192</td>
      <td>0.0873</td>
      <td>0.0422</td>
      <td>210,404,538.6605</td>
      <td>4.2158</td>
      <td>2.2695</td>
      <td>0.6393</td>
      <td>0.3811</td>
      <td>0.8823</td>
      <td>38.8270</td>
      <td>0.9458</td>
      <td>0.6036</td>
      <td>0.2625</td>
      <td>0.5079</td>
      <td>0.7011</td>
      <td>1.0042</td>
      <td>0.2920</td>
    </tr>
    <tr>
      <th>skew</th>
      <td>3.5171</td>
      <td>0.3409</td>
      <td>1.8674</td>
      <td>1.3973</td>
      <td>-0.7271</td>
      <td>0.8190</td>
      <td>4.5114</td>
      <td>-0.0207</td>
      <td>0.6741</td>
      <td>0.1521</td>
      <td>-0.5107</td>
      <td>0.5395</td>
      <td>0.5864</td>
      <td>0.5075</td>
      <td>0.4373</td>
      <td>-0.9455</td>
      <td>0.3478</td>
      <td>0.2244</td>
      <td>0.0803</td>
      <td>0.6012</td>
    </tr>
    <tr>
      <th>kurt</th>
      <td>16.5495</td>
      <td>1.6944</td>
      <td>5.7661</td>
      <td>3.4935</td>
      <td>2.4405</td>
      <td>2.4176</td>
      <td>25.2469</td>
      <td>-0.5362</td>
      <td>0.3606</td>
      <td>0.8562</td>
      <td>1.1539</td>
      <td>1.4082</td>
      <td>2.3305</td>
      <td>0.6322</td>
      <td>0.7250</td>
      <td>2.7936</td>
      <td>2.3221</td>
      <td>1.8711</td>
      <td>1.4487</td>
      <td>3.5935</td>
    </tr>
    <tr>
      <th>count</th>
      <td>36.0000</td>
      <td>44.0602</td>
      <td>44.0241</td>
      <td>44.1566</td>
      <td>44.1566</td>
      <td>44.1205</td>
      <td>36.0843</td>
      <td>44.2169</td>
      <td>37.3012</td>
      <td>30.2289</td>
      <td>30.2289</td>
      <td>30.2289</td>
      <td>30.2289</td>
      <td>30.2289</td>
      <td>30.2289</td>
      <td>30.2289</td>
      <td>41.8795</td>
      <td>38.4337</td>
      <td>31.9277</td>
      <td>44.2169</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0001</td>
      <td>-1.7141</td>
      <td>-0.6751</td>
      <td>0.0016</td>
      <td>-0.3634</td>
      <td>0.0017</td>
      <td>170,857.4105</td>
      <td>-7.1226</td>
      <td>15.1803</td>
      <td>-0.8190</td>
      <td>0.2622</td>
      <td>-1.0097</td>
      <td>-61.0548</td>
      <td>-1.1095</td>
      <td>-0.5689</td>
      <td>0.2401</td>
      <td>-0.9632</td>
      <td>-1.3270</td>
      <td>-1.7192</td>
      <td>-0.5641</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5,086.6411</td>
      <td>2.2996</td>
      <td>11.0954</td>
      <td>0.5119</td>
      <td>0.0020</td>
      <td>0.1781</td>
      <td>1,308,240,639.7914</td>
      <td>8.2548</td>
      <td>24.1141</td>
      <td>1.7036</td>
      <td>1.7761</td>
      <td>2.6050</td>
      <td>100.6769</td>
      <td>2.4813</td>
      <td>1.8168</td>
      <td>1.2935</td>
      <td>1.3427</td>
      <td>1.8044</td>
      <td>2.5441</td>
      <td>0.8457</td>
    </tr>
    <tr>
      <th>median</th>
      <td>0.1084</td>
      <td>0.0545</td>
      <td>1.5419</td>
      <td>0.1119</td>
      <td>-0.1172</td>
      <td>0.0506</td>
      <td>2,253,162.0171</td>
      <td>0.7036</td>
      <td>18.3326</td>
      <td>0.4150</td>
      <td>1.0869</td>
      <td>0.4640</td>
      <td>8.6625</td>
      <td>0.2920</td>
      <td>0.4367</td>
      <td>0.9320</td>
      <td>0.0485</td>
      <td>0.1384</td>
      <td>0.3107</td>
      <td>0.0018</td>
    </tr>
    <tr>
      <th>quantile0.05</th>
      <td>0.0012</td>
      <td>-1.0884</td>
      <td>-0.2828</td>
      <td>0.0071</td>
      <td>-0.2458</td>
      <td>0.0031</td>
      <td>282,974.3722</td>
      <td>-5.6922</td>
      <td>15.9418</td>
      <td>-0.4724</td>
      <td>0.4579</td>
      <td>-0.5034</td>
      <td>-38.4297</td>
      <td>-0.7979</td>
      <td>-0.2634</td>
      <td>0.4381</td>
      <td>-0.5761</td>
      <td>-0.8020</td>
      <td>-1.0894</td>
      <td>-0.3292</td>
    </tr>
    <tr>
      <th>quantile0.25</th>
      <td>0.0155</td>
      <td>-0.3721</td>
      <td>0.5694</td>
      <td>0.0591</td>
      <td>-0.1592</td>
      <td>0.0303</td>
      <td>825,318.9527</td>
      <td>-2.1898</td>
      <td>17.3030</td>
      <td>0.0573</td>
      <td>0.8882</td>
      <td>-0.0055</td>
      <td>-5.7353</td>
      <td>-0.2375</td>
      <td>0.1094</td>
      <td>0.8273</td>
      <td>-0.2121</td>
      <td>-0.2545</td>
      <td>-0.3392</td>
      <td>-0.1334</td>
    </tr>
    <tr>
      <th>quantile0.75</th>
      <td>1.2623</td>
      <td>0.5732</td>
      <td>2.9655</td>
      <td>0.1763</td>
      <td>-0.0671</td>
      <td>0.0708</td>
      <td>13,218,682.3488</td>
      <td>3.8612</td>
      <td>20.0003</td>
      <td>0.7506</td>
      <td>1.2778</td>
      <td>0.9592</td>
      <td>26.3862</td>
      <td>0.8757</td>
      <td>0.7996</td>
      <td>1.0357</td>
      <td>0.2919</td>
      <td>0.4756</td>
      <td>0.8697</td>
      <td>0.1390</td>
    </tr>
    <tr>
      <th>quantile0.95</th>
      <td>1,521.1662</td>
      <td>1.4567</td>
      <td>6.6843</td>
      <td>0.3297</td>
      <td>-0.0062</td>
      <td>0.1184</td>
      <td>269,932,881.8052</td>
      <td>7.0933</td>
      <td>22.4340</td>
      <td>1.2600</td>
      <td>1.5466</td>
      <td>1.8803</td>
      <td>67.5178</td>
      <td>1.8483</td>
      <td>1.3720</td>
      <td>1.2010</td>
      <td>0.8373</td>
      <td>1.2024</td>
      <td>1.7257</td>
      <td>0.4500</td>
    </tr>
  </tbody>
</table>
</div>



## Univariate factor trends

Alongside looking at simple cross-sectional averages of our factors across time, we also care about the distribution of these factors. We can use a simple boxplot to examine this. Key features to look for are changes in the spread of values in the IQR, as well as how far away the min/max values are from the mean. If you see large volatility in the factors, or sudden changes in the distribution, this could suggest that there's potentially something wrong in your calculation, or the factor itself is very unstable and thus might not be suitable for investment.

Our first chart looks at the change in the beta for each coin against the S&P500. What's interesting is the marked shift in beta that occured in April 2020, as well as the significant decline in spread that occurred in April 2018. 

Our second charge look at return volatility, where we see the natural result of the very large spike in volatility that occurred in late 2017/early 2018. The behaviour of this spike is significantly different to the recent 2020 acceleration in BTC price. This suggests that there's potentially a more orderly price increase going on, which may be suggestive of a more stable price behaviour.


![png](/img/cryptosyste_30_0.png)

![png](/img/cryptosyste_31_0.png)


## Univariate portfolios

After studying the factors, the next step is to usually construct a portfolio. To do this, for your factor, on the portfolio rebalancing date, you sort based on your factor and then split all of your assets into $N$ number of groups. We use 3, due to the small number of assets. The typical approach is to use 5 or 10 portfolios. Within each portfolio, you can then calculate a portfolio return for each month. The standard approach is to use a value-weighted return (on market cap) or an equal-weighted return (simple average). In reality, your portfolio likely will be weighted differently (for example, optimised using mean-variance) so you would need to take this into account.

Now that you have returns for each of your portfolios, for each factor, for each month, you can study the average behaviour. The standard approach is to assume that you can go 100% long the "high" portfolio (i.e. portfolio associated with high values of your factor) and 100% short the "low" portfolio. This naturally has numerous problems, primarily the assumption that you can short everything in the "low" portfolio. However, it suffices as a theoretical representation of the possible validity of the factor in predicting returns. Dealing with implementation and feasibility is often left to industry, rather than academia.

One common method of studying portfolios is to study the various factors within each portfolio. This highlights potential similarities in factors and expected characteristics. Here, we sort our portfolio on market cap and then study the factors.

We find several results:
1. Low market cap is associated with highly illiquid assets, which also have the largest/smallest returns and tend to have higher levers of skewness. 
2. Interestingly, the cryptobeta is relatively similar across all market caps.

Typically, you would follow this approach for your "new" factor of interest, to examine how it related to existing factors.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Low</th>
      <th>Mid</th>
      <th>High</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>illiq</th>
      <td>12.4336</td>
      <td>0.2182</td>
      <td>0.0096</td>
    </tr>
    <tr>
      <th>skew</th>
      <td>0.0934</td>
      <td>0.1519</td>
      <td>0.0413</td>
    </tr>
    <tr>
      <th>kurt</th>
      <td>1.4823</td>
      <td>1.5420</td>
      <td>1.9720</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.1701</td>
      <td>0.1510</td>
      <td>0.1258</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.1557</td>
      <td>-0.1420</td>
      <td>-0.1240</td>
    </tr>
    <tr>
      <th>vol</th>
      <td>0.0745</td>
      <td>0.0631</td>
      <td>0.0523</td>
    </tr>
    <tr>
      <th>adv</th>
      <td>699,484.4725</td>
      <td>2,364,494.7222</td>
      <td>34,097,672.6168</td>
    </tr>
    <tr>
      <th>prc</th>
      <td>-1.6425</td>
      <td>-0.7282</td>
      <td>1.7107</td>
    </tr>
    <tr>
      <th>mcap</th>
      <td>16.7907</td>
      <td>18.4369</td>
      <td>21.0523</td>
    </tr>
    <tr>
      <th>equitybeta</th>
      <td>0.2769</td>
      <td>0.5381</td>
      <td>0.4723</td>
    </tr>
    <tr>
      <th>ivol</th>
      <td>1.2439</td>
      <td>1.1108</td>
      <td>0.9261</td>
    </tr>
    <tr>
      <th>iskew</th>
      <td>0.5959</td>
      <td>0.5582</td>
      <td>0.3027</td>
    </tr>
    <tr>
      <th>sskew</th>
      <td>16.4579</td>
      <td>5.3209</td>
      <td>8.8939</td>
    </tr>
    <tr>
      <th>tskew</th>
      <td>0.4138</td>
      <td>0.4431</td>
      <td>0.0825</td>
    </tr>
    <tr>
      <th>gold_beta</th>
      <td>0.6139</td>
      <td>0.5050</td>
      <td>0.3769</td>
    </tr>
    <tr>
      <th>cryptobeta</th>
      <td>0.9311</td>
      <td>0.9060</td>
      <td>0.9451</td>
    </tr>
    <tr>
      <th>mom_3</th>
      <td>-0.0890</td>
      <td>0.0720</td>
      <td>0.1091</td>
    </tr>
    <tr>
      <th>mom_6</th>
      <td>-0.1190</td>
      <td>0.1622</td>
      <td>0.2273</td>
    </tr>
    <tr>
      <th>mom_12</th>
      <td>-0.1919</td>
      <td>0.3876</td>
      <td>0.4364</td>
    </tr>
    <tr>
      <th>reversal</th>
      <td>-0.0373</td>
      <td>0.0040</td>
      <td>0.0323</td>
    </tr>
    <tr>
      <th>monthly_rf</th>
      <td>0.0146</td>
      <td>-0.0286</td>
      <td>0.0027</td>
    </tr>
  </tbody>
</table>
</div>



This chart shows the cumulative value-weighted log-returns associated with the long/short portfolios for each of our factors. In essence, it's the cumulative returns associated with the factor if you had been able to construct a long/short portfolio. 

What is particularly interesting here, is post 2018 the flatlining of almost every single factor. This is potentially suggestive of the market becoming incredibly saturated as investors piled in after the crash. There are many interpretations that could be explored, perhaps none of these are genuine pricing factors and it's simply random noise, perhaps the underlying assets are far too volatile to draw any meaningful conclusions.

This highlights one of the primary problems with studying cryptocurrencies, we only have 6 years of good data, at best. It's incredibly difficult to draw any meaningful results from such a small timeframe. A more fruitful area would be in examining tick-by-tick behaviour, where there is substantially more data to study.


![png](/img/cryptosyste_37_0.png)


In summary, we find that there is a significant size/volume factor present, which produces significantly negative returns. On the positive side, factors that are anti-correlated with market cap produce positive excess returns. Interestingly, we find that standard momentum/reversal factors don't produce any returns. This is likely because we use monthly frequency, and the behaviour of crypto might be simply too high frequency to capture any long-term momentum returns.


![png](/img/cryptosyste_39_0.png)


Another common technique in academic finance is to use Fama-Macbeth (1973) regressions to study the efficacy of factors in predicting future returns, in the presence of multiple factors. There are issues with this approach, such as assuming linearity, and having to deal with collinearity, but it serves as a commonly accepted approach to study factor behaviour.

We run 13 regressions, where we incrementally add a factor into each regression specification.

This table shows the average coefficient associated with each factor, and a corresponding t-stat. The primary conclusion is that there's no statistically significant factors which predict one-month forward returns. However, looking at the $R^2$ across each regression we run, we can see that we can indeed explain a large portion of the cross-sectional returns. This is potentially due to the very small dataset we're working with. If we used a full set of coins, we would expect this $R^2$ value to drop significantly.

![png](/img/cryptosyste_45_0.png)


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>variable</th>
      <th>Mean</th>
      <th>T-Stat</th>
    </tr>
    <tr>
      <th>factor</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>max</th>
      <td>-0.0169</td>
      <td>-0.9388</td>
    </tr>
    <tr>
      <th>iskew</th>
      <td>-0.0140</td>
      <td>-1.1293</td>
    </tr>
    <tr>
      <th>illiq</th>
      <td>-0.0139</td>
      <td>-1.0412</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.0071</td>
      <td>-0.7229</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>-0.0025</td>
      <td>-1.1173</td>
    </tr>
    <tr>
      <th>ivol</th>
      <td>-0.0025</td>
      <td>-1.1074</td>
    </tr>
    <tr>
      <th>cryptobeta</th>
      <td>-0.0011</td>
      <td>-1.1304</td>
    </tr>
    <tr>
      <th>skew</th>
      <td>-0.0009</td>
      <td>-1.0767</td>
    </tr>
    <tr>
      <th>reversal</th>
      <td>-0.0004</td>
      <td>-0.4896</td>
    </tr>
    <tr>
      <th>equitybeta</th>
      <td>-0.0001</td>
      <td>-1.1523</td>
    </tr>
    <tr>
      <th>sskew</th>
      <td>-0.0000</td>
      <td>-1.1251</td>
    </tr>
    <tr>
      <th>adv</th>
      <td>-0.0000</td>
      <td>-1.0430</td>
    </tr>
    <tr>
      <th>mcap</th>
      <td>0.0001</td>
      <td>1.1284</td>
    </tr>
    <tr>
      <th>kurt</th>
      <td>0.0002</td>
      <td>1.1011</td>
    </tr>
    <tr>
      <th>mom_12</th>
      <td>0.0002</td>
      <td>1.1306</td>
    </tr>
    <tr>
      <th>gold_beta</th>
      <td>0.0013</td>
      <td>1.1163</td>
    </tr>
    <tr>
      <th>tskew</th>
      <td>0.0141</td>
      <td>1.1294</td>
    </tr>
    <tr>
      <th>vol</th>
      <td>0.1056</td>
      <td>1.0993</td>
    </tr>
  </tbody>
</table>
</div>


