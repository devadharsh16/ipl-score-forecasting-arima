# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 14:11:28 2026

@author: Devadharshini
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("D:\\Academic IIPS\\MSD 4TH SEM\\Assignment\\Projection\\preeti mam\\Projection ipl.csv")

print(df.head())
print(df.columns)
#cleaning and agreegation
df = df[df['innings'] == 1]
def fix_season(x):
    x = str(x).strip()

    # special covid rule
    if x == "2020/21":
        return 2020

    # slash seasons
    if '/' in x:
        last_year = x.split('/')[-1]
        if len(last_year) == 2:
            return int("20" + last_year)
        else:
            return int(last_year)

    # short years like 08 or 10
    if len(x) == 2:
        return int("20" + x)

    # normal seasons
    return int(x)

df['season'] = df['season'].apply(fix_season)
print(sorted(df['season'].unique()))

match_scores = df.groupby(
    ['match_id','season']
)['runs_total'].sum().reset_index()

match_scores.rename(columns={'runs_total':'total_score'}, inplace=True)

print(match_scores.head())

season_series = match_scores.groupby(
    'season'
)['total_score'].mean().reset_index()

season_series.set_index('season', inplace=True)

print(season_series)
season_series = [int(year) for year in season_series]
df['season_series'] = df['season_series'].astype(int)


#plot the trend analysis
plt.figure(figsize=(14,6))
plt.plot(season_series.index,
         season_series['total_score'],
         marker='o',
         color='green')

plt.title("Average First Innings Score per Season")
plt.xlabel("Season")
plt.ylabel("Average Score")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))


plt.grid()
plt.show()

plot_acf(season_series['total_score'])
plt.show()

plot_pacf(season_series['total_score'])
plt.show()

#adf test
result = adfuller(season_series['total_score'])

print("ADF Statistic:", result[0])
print("p-value:", result[1])

#differencing
season_series['diff_score'] = \
    season_series['total_score'].diff()

print(season_series)
plt.figure(figsize=(10,5))
plt.plot(season_series['diff_score'])
plt.title("Differenced IPL Score Series")
plt.show()
result = adfuller(
    season_series['diff_score'].dropna()
)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
#arima
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(season_series['total_score'],
              order=(1,1,0))

model_fit = model.fit()
print(model_fit.summary().as_text())

print(model_fit.summary())
#forecast
forecast = model_fit.forecast(steps=5)

print(forecast)
#plot the forecast
plt.figure(figsize=(10,5))

plt.plot(season_series.index,
         season_series['total_score'],
         label='Actual Score')

future_years = range(
    season_series.index[-1] + 1,
    season_series.index[-1] + 6
)

plt.plot(future_years,
         forecast,
         label='Forecast Score')

plt.legend()
plt.title("IPL Average Score Forecast")
plt.show()
#overall fit
model_fit.summary()

#diagnostics
model_fit.plot_diagnostics(figsize=(10,8))
plt.show()

#extrass..........
forecast_res = model_fit.get_forecast(steps=5)

forecast = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()
plt.figure(figsize=(10,5))

plt.plot(season_series.index,
         season_series['total_score'],
         label='Actual score')

future_years = range(
    season_series.index[-1]+1,
    season_series.index[-1]+6
)

plt.plot(future_years, forecast, label='Forecast score')

plt.fill_between(
    future_years,
    conf_int.iloc[:,0],
    conf_int.iloc[:,1],
    alpha=0.2
)

plt.legend()
plt.title("IPL Average Score Forecast")
plt.show()















