import requests
import streamlit as st
import sys
import warnings
import brotli
import json
from panel.interact import interact
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import nselib
from scipy.stats import norm
from datetime import date
from nselib import capital_market
import yfinance as yf
from dateutil.relativedelta import relativedelta 
import panel as pn
pn.extension()
start_date = (date.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
end_date = (date.today() - relativedelta(days=1)).strftime('%Y-%m-%d')
st.set_page_config(page_title = "Option Price Calculator",layout="wide",page_icon=":material/bar_chart:")


####----------Stock selection--------------------------
st.title('🧮 OPTION PRICE CALCULATOR')

def stock_fetcher():
    symbol_df = capital_market.equity_list()            # library to get data of symbols
    symbol_df.columns = symbol_df.columns.str.strip()
    return list(symbol_df['SYMBOL'].unique())
input_symbol = st.multiselect("Select Stock:",options = stock_fetcher())

if input_symbol:
    #############################################--------------------------- 1 Year historical data from yahoo finance----------------------
    sym = input_symbol[0]
    req_symbol = str(sym) + '.NS'    
    @st.cache_data
    def get_data_from_yahoo(req_symbol):
        data = yf.download(req_symbol, start=start_date, end = end_date)
        price_data = data.reset_index()
        price_data['returns'] = price_data['Close'].pct_change()                         ### Calculation of returns
        price_data = price_data.dropna()
        return price_data
    price_data_from_yf = get_data_from_yahoo(req_symbol)
    
    ###-------------------Params and urls--------------------
    base_url = "https://www.nseindia.com"
    url = f'https://www.nseindia.com/api/option-chain-equities?symbol={sym}' 
    headers = {"accept-encoding":"gzip, deflate, br, zstd",
              "accept-language":"en-US,en;q=0.9",
              "user-agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0"}
    
    sigma = price_data_from_yf['returns'].std()*np.sqrt(252)                                 ### Annualised stock volatility
    risk_free = 0.0678      
    #########################################################################################################################################
    @st.cache_data
    def get_live_option_chain_from_nse(base_url, headers, url):
        with requests.Session() as s:
            r = s.get(base_url, headers=headers)
            api_url = url
            response = s.get(api_url, headers=headers).json()
        try:
            data = response['records']['data']
            list_of_options = []
            for i in data:
                for key,info in i.items():
                    if key == 'CE' or key == 'PE':
                        info_of_options = info
                        info_of_options['opt_type'] = key
                        list_of_options.append(info_of_options)
            return list_of_options
        except:
            st.error(f'Option Data is not Available for {sym} on NSE ⚠️')
            return []
    option_chain_from_nse = get_live_option_chain_from_nse(base_url, headers, url)

    @st.cache_data
    def Black_schole_Option_Pricing(option_chain_from_nse):
        if not option_chain_from_nse:
            st.error("Stopping further execution.")
            sys.exit()
        else:
            data_of_option = pd.DataFrame(option_chain_from_nse)[['strikePrice','expiryDate','underlying','lastPrice','underlyingValue','opt_type']]
            data_of_option.rename(columns={'strikePrice' : 'Strike','expiryDate':'Expiry','underlying':'Underlying','lastPrice':'Option_price',
                                       'underlyingValue':'StockPrice'},inplace=True)
            data_of_option['opt_type'] = data_of_option['opt_type'].replace("\s+", " ", regex=True).str.strip()
            data_of_option['Volatility'] = sigma
            data_of_option['risk_free_rate'] = risk_free
            data_of_option['Date'] = pd.to_datetime(date.today(), format='%d-%b-%Y')
            data_of_option['Expiry'] = pd.to_datetime(data_of_option['Expiry'], format='%d-%b-%Y')
            data_of_option['Maturity'] = (data_of_option['Expiry'] - data_of_option['Date'])
            data_of_option["Maturity"] = (data_of_option['Maturity'].dt.days) / 365             ### Maturity in term s of Year
            data_of_option['d1'] = (np.log(data_of_option['StockPrice'] / data_of_option['Strike'] ) + ((data_of_option['risk_free_rate'] + (data_of_option['Volatility']**2 / 2))*data_of_option['Maturity'])) / (data_of_option['Volatility'] * np.sqrt(data_of_option['Maturity']))
            data_of_option['d2'] = (np.log(data_of_option['StockPrice'] / data_of_option['Strike'] ) + ((data_of_option['risk_free_rate'] - (data_of_option['Volatility']**2 / 2))*data_of_option['Maturity'])) / (data_of_option['Volatility'] * np.sqrt(data_of_option['Maturity']))     
            data_of_option['N(d1)'] = norm.cdf(data_of_option['d1'])
            data_of_option['N(d2)'] = norm.cdf(data_of_option['d2'])
            data_of_option['N(-d1)'] = norm.cdf(-1 * data_of_option['d1'])
            data_of_option['N(-d2)'] = norm.cdf(-1* data_of_option['d2'])
            data_of_option['BS_option_price'] = np.where(
                data_of_option['opt_type'] == 'CE',
                data_of_option['StockPrice'] * data_of_option['N(d1)'] -
                data_of_option['Strike'] * data_of_option['N(d2)'] * np.exp(-1 * data_of_option['Maturity'] * data_of_option['risk_free_rate']),
                np.where(
                    data_of_option['opt_type'] == 'PE',
                    data_of_option['Strike'] * data_of_option['N(-d2)'] * np.exp(-1 * data_of_option['Maturity'] * data_of_option['risk_free_rate']) -
                    data_of_option['StockPrice'] * data_of_option['N(-d1)'],
                    np.nan
                )
            )
            data_of_option['Bs_option_price'] = round(data_of_option['BS_option_price'],4)
            data_of_option['OPTION'] = data_of_option['Underlying'] + data_of_option['Expiry'].astype(str) + data_of_option['opt_type'] + data_of_option['Strike'].astype(str)
            data_of_option = data_of_option[['OPTION','StockPrice','Option_price','risk_free_rate','Volatility','opt_type','Maturity','Bs_option_price','Strike','Expiry','Date']].sort_values(by=['Expiry','Strike']).reset_index(drop=True)
            return data_of_option
    Black_schole_Option_Price_df = Black_schole_Option_Pricing(option_chain_from_nse)
    option_name = st.multiselect("Select Option:",options = list(Black_schole_Option_Price_df['OPTION']))

    if option_name:
        opt = option_name[0]

        @st.cache_data
        def Monte_carlo_Option_pricing(opt):
            random_numbers = np.random.normal(size=10000)
            simulations = pd.DataFrame(random_numbers,columns=['Random'])
            simulations = pd.concat([Black_schole_Option_Price_df[Black_schole_Option_Price_df['OPTION'] == opt], simulations], axis=1).reset_index(drop=True)
            simulations.fillna(simulations.iloc[0], inplace=True)
            simulations['st'] = simulations['StockPrice']*np.exp(((simulations['risk_free_rate'] - (simulations['Volatility']**2 / 2)) * simulations['Maturity']) + simulations['Volatility']*simulations['Random']*np.sqrt(simulations['Maturity']))
            simulations['Payoff'] = np.where(simulations['opt_type'] == 'CE',np.maximum(simulations['st'] - simulations['Strike'],0),
                                    np.where(simulations['opt_type'] == 'PE',np.maximum(simulations['Strike'] - simulations['st'],0),
                                    np.nan))
            simulations['pv_of_payoff'] = simulations['Payoff'] * np.exp(-1*simulations['risk_free_rate']*simulations['Maturity'])
            simulations['Monte_carlo_option_price'] = simulations['pv_of_payoff'].mean()   
            BS_Monte_data = simulations.drop_duplicates(subset=['StockPrice'])
            BS_Monte_data.drop(columns=['Random', 'st','Payoff','pv_of_payoff'], inplace=True)
            return BS_Monte_data
            
        BS_Monte_Carlo_Option_Price_df = Monte_carlo_Option_pricing(opt)

        @st.cache_data
        def Binomial_Option_pricing(BS_Monte_Carlo_Option_Price_df):
            binomial_data = BS_Monte_Carlo_Option_Price_df.copy()
            binomial_data['timesteps'] = (BS_Monte_Carlo_Option_Price_df['Expiry'] - BS_Monte_Carlo_Option_Price_df['Date']).dt.days
            binomial_data['delta_t'] = binomial_data['Maturity'] / binomial_data['timesteps']    # calculating timesteps for binomial tree
            bn = binomial_data.copy()
        
            ###################### ---------------------------------------------------------------------------------------------------
            ##-------------'Parameters'
            
            u = np.exp(bn['Volatility'] * np.sqrt(bn['delta_t']))    # percentage increase in stock price when there is an up-movement
            d = (1/u)                                                # percentage decrease in stock price when there is an down-movement
            prob = (np.exp(bn['risk_free_rate'] * bn['delta_t']) - d) / (u -d )  # probability of an up movement
            prob_1 = 1 - prob                                                    # probability of an up movement
            r = float(bn['risk_free_rate'])
            del_t = float(bn['delta_t'])
            T = float(bn['Maturity'])
            K = float(bn['Strike'])
            
            ###################### ----------------------------------------------------------------------------------------------------
            ##--------------'Binomial tree of stock price in forward induction'
            tree = []
            s0 = [float(bn['StockPrice'])]
            for t in range(1,int(bn['timesteps']) + 1):
                # print(f'time:{t}')
                temp = []
                for td in range(t+1):
                    tu = t - td
                    st = round(float(s0*(u**tu)*(d**td)),4)          # binomial progression
                    temp.append(st)
                my_list = sorted(temp, reverse=True)
                # print(len(my_list))
                tree.append(my_list)
            payoff_list = tree[-1]                                  # last time step of binomial tree of stcok price
            
            ###################### --------------------------------------------------------------------------------------------------
            #---------------'Binomial tree of option payoff in backward induction'
            
            payoff_df = pd.DataFrame(payoff_list,columns=['ST']).sort_values(by='ST',ascending=False).reset_index(drop=True)
            payoff_df['Strike'] = float(bn['Strike'])
            payoff_df['Payoff_bn'] = np.where(bn['opt_type'] == 'CE',np.maximum(payoff_df['ST'] - payoff_df['Strike'], 0),
                                     np.where(bn['opt_type'] == 'PE',np.maximum(payoff_df['Strike'] - payoff_df['ST'], 0), np.nan))
            payoff_list = payoff_df['Payoff_bn'].to_list()
            payoff_list = sorted(payoff_list, reverse=True)
            while (len(payoff_list) > 1):
                payoff_from_bn = []
                for t in range(len(payoff_list)-1):
                    call = np.exp(-1*r*del_t)*(prob * payoff_list[t] + prob_1 * payoff_list[t+1])
                    payoff_from_bn.append(call)
                payoff_list = payoff_from_bn
            BS_Monte_Carlo_Binomial_Option_Price_df = binomial_data
            BS_Monte_Carlo_Binomial_Option_Price_df['Binomial_price'] = float(payoff_list[0])
            return BS_Monte_Carlo_Binomial_Option_Price_df
        
        BS_Monte_Carlo_Binomial_Option_Price_df = Binomial_Option_pricing(BS_Monte_Carlo_Option_Price_df)
        
        @st.cache_data
        def Euler_Maruyama_Option_Pricing(BS_Monte_Carlo_Binomial_Option_Price_df):
            ##############################################################
            #---------------Params---------------------
            r = risk_free
            K = float(BS_Monte_Carlo_Binomial_Option_Price_df['Strike'])
            n = 10000
            m = int(BS_Monte_Carlo_Binomial_Option_Price_df['timesteps'])+1
            dt = float(BS_Monte_Carlo_Binomial_Option_Price_df['delta_t'])
            T = float(BS_Monte_Carlo_Binomial_Option_Price_df['Maturity'])
        
            #############################################################
            #-------------Recursion---------------------------------
            S = np.zeros((n,m))
            S[:,0] = float(BS_Monte_Carlo_Binomial_Option_Price_df['StockPrice'])
            for t in range(1,m):
                Z = np.random.standard_normal(n) # generating 10,000 normally distirbuted random numbers with mean 0 and variance 1
                S[:,t] = S[:,t-1] + r * S[:,t-1] * dt + sigma * S[:,t-1] * np.sqrt(dt) * Z  # recursion of stock price
            if BS_Monte_Carlo_Binomial_Option_Price_df['opt_type'].values[0] == 'PE':
                # print('put')
                payoff = np.maximum(K - S[:,-1], 0)
            else:
                # print('call')
                payoff = np.maximum(S[:,-1] - K, 0)
            price_of_option = np.exp(-r*T) * np.mean(payoff)
            BS_Monte_Carlo_Binomial_Option_Price_df['Euler_price'] = price_of_option
            BS_Monte_Carlo_Binomial_Euler_Option_Price_df = BS_Monte_Carlo_Binomial_Option_Price_df[['OPTION','StockPrice','Option_price','Bs_option_price','Monte_carlo_option_price','Binomial_price','Euler_price']]
            BS_Monte_Carlo_Binomial_Euler_Option_Price_df.rename(columns={'Option_price':'NSE_Price'}, inplace=True)
            return BS_Monte_Carlo_Binomial_Euler_Option_Price_df
        
        st.dataframe(Euler_Maruyama_Option_Pricing(BS_Monte_Carlo_Binomial_Option_Price_df))

