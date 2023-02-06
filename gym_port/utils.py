from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np


def get_price_tensors(stocks: int, mode: str, input_periods: int) -> pd.DataFrame:

    out = []

    for stock in stocks:
        data = pd.read_csv('data\{0}_{1}.csv'.format(stock,mode),parse_dates=True) 
        len_data = len(data.index)
        array_ =  []

        for i in range(len_data-input_periods+1):
            array = []
            close = data['Close'].iloc[i:input_periods+i]
            high = data['High'].iloc[i:input_periods+i]
            low = data['Low'].iloc[i:input_periods+i]

            close_norm = close.div(close.iloc[-1])
            high_norm = high.div(close.iloc[-1])
            low_norm = low.div(close.iloc[-1])

            array.append(close_norm)
            array.append(high_norm)
            array.append(low_norm)
            array_.append(array)

        out.append(array_)

    return np.swapaxes(np.asarray(out),0,1)

def get_return_tensors(stocks: list, mode: str, input_periods: int) -> pd.DataFrame:

    returns = pd.DataFrame()

    data = pd.read_csv('data\{0}_{1}.csv'.format(stocks[0],mode),parse_dates=True) 
    data = data.iloc[input_periods-2 : ]
    data.set_index('Date', inplace=True)
    returns['Coin'] = data['Close'].div(data['Close'])
    
    for stock in stocks:
        data = pd.read_csv('data\{0}_{1}.csv'.format(stock,mode),parse_dates=True)
        data = data.iloc[input_periods-2 : ] 
        data.set_index('Date', inplace=True)
        returns[stock] = data['Close'].div(data['Close'].shift(1))
    return returns.dropna().to_numpy()


def get_env_args(stocks: list, mode: str, input_periods: int) -> Dict[str, pd.DataFrame]:


    # TO DO: modify to only import files once
    prices = get_price_tensors(stocks, mode, input_periods)
    returns = get_return_tensors(stocks, mode, input_periods)

    return {"stocks": stocks, "prices": prices, "returns": returns }

if __name__ == "__main__":

    # for testing
    stocks = ["III", "AAL", "ABDN", "ADM", "AHT", "ANTO", "AZN"]
    mode = "Train"
    test = get_return_tensors(stocks, "Train", 20) 
    test = get_price_tensors(stocks, "Train", 20)
