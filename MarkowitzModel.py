# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 09:40:44 2022

@author: macwr
"""

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

NUM_TRADING_DAYS = 252

NUM_PORTFOLIOS = 10000

#list of stocks
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

#define timeframe
startdate = '2021-01-01'
enddate = '2022-01-01'

#funcion to collect data from yahoo finance
def getData():
    data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        data[stock] = ticker.history(start = startdate, end = enddate)['Close']
        
    return pd.DataFrame(data)

#displays stock ticker data in a chart
def showData(data):
    data.plot(figsize=(10,5))
    plt.show()

#calculates each daily return for selected stocks
def calcReturn(data):
    logReturn = np.log(data/data.shift(1))
    return logReturn[1:]

#Calculates and displays the annual mean and annual covariance
#for a given set of returns from a portfolio
def showStats(returns):
    print('Annual Mean: ')
    print(returns.mean() * NUM_TRADING_DAYS)
    print('Annual Covariance: ')
    print(returns.cov() * NUM_TRADING_DAYS)
    
#Displays the mean and variance for a given portfolio
def showMeanAndVariance(returns, weights):
    portfolioReturn = np.sum(returns.mean * weights) * NUM_TRADING_DAYS
    portfolioVolatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    print("Expected portfolio mean (retuns): ", portfolioReturn)
    print("Expected portfolio volatility (retuns): ", portfolioVolatility)
    
#generate a number of portfolios and determine the means
#and risks associated with each porflio
def generatePortfolios(returns):
    #output variables
    portfolioMeans = []
    portfolioRisks = [] 
    portfolioWeights = [] 
    
    for i in range(NUM_PORTFOLIOS):
        #create a normalizaed list of weights for the stocks
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        #calculate portfolio stats and append to output arrays
        portfolioWeights.append(w)
        portfolioMeans.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolioRisks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))
        
    return np.array(portfolioWeights), np.array(portfolioMeans), np.array(portfolioRisks)

def statistics(weights, returns):
    portfolioReturn = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolioVolatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    
    return np.array([portfolioReturn, portfolioVolatility, portfolioReturn/portfolioVolatility])

def minFunctionSharpe(weights, returns):
    #return sharpe ratios negative to optimize for max val
    return -statistics(weights, returns)[2]

def optimizePortfolio(weights, returns):
    #sum of portfolio weights = 1
    constraints = {'type' : 'eq', 'fun': lambda x: np.sum(x) -1}
    #Each stock must have a weight between 0 and 1
    bounds = tuple((0,1)for i in range(len(stocks)))
    #perform optimization
    return optimization.minimize(fun=minFunctionSharpe, x0=weights[0], args=returns, method='SLSQP'
                          ,bounds = bounds, constraints=constraints)

def displayOptimalPortfolio(optimum, returns):
    print("Optimal porfolio: ", optimum['x'].round(3))
    print("Expected return, volatility, Sharpe ratio: ",
          statistics(optimum['x'],returns).round(3))
    
#plots the expected means, returns, and sharpe ratio for each input portfolio
def showPortfolios(opt, rets, returns, volatilities):
    plt.figure(figsize = (10,6))
    plt.scatter(volatilities, returns, c = returns/volatilities, marker = 'o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label = "Sharpe Ratio")
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize = 30)
    plt.show()

if __name__ == '__main__':
    data = getData()
    showData(data)
    logDailyReturns = calcReturn(data)
    #showStats(logDailyReturns)
    
    portfolioWeights, means, risks = generatePortfolios(logDailyReturns)
    optimum = optimizePortfolio(portfolioWeights, logDailyReturns)
    displayOptimalPortfolio(optimum, logDailyReturns)
    showPortfolios(optimum, logDailyReturns, means, risks)
    
    
