import pandas as pd

def save_train_test(ticker: str) -> None:
    full_file = pd.read_csv('data\{0}.L.csv'.format(ticker),parse_dates=True) 
    full_file.set_index('Date', inplace=True)

    # save train data
    full_file.loc['2010-02-04':'2020-12-01'].to_csv('data\{0}_Train.csv'.format(ticker))

    # save test data
    full_file.loc['2020-12-01':].to_csv('data\{0}_Test.csv'.format(ticker))
    

if __name__ == "__main__":

    # use to transform data
    tickers = ["III", "AAL", "ABDN", "ADM", "AHT", "ANTO", "AZN"]
    for ticker in tickers:
        save_train_test(ticker)
    
