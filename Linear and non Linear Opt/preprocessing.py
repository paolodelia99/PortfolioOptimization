from os import listdir
from os.path import isfile, join
import argparse

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nation', type=str,
                        help='Nation of the instrument to preprocess')
    parser.add_argument('-i', '--instument', type=str,
                        help='Type of financial instrument to parse')
    
    args = parser.parse_args()
    return args

def preprocess_dk_stock_data():
    def reindex_stocks(dfs: list, start_date: str, end_date: str): 
        return [dfs[i].loc[start_date:, :].reindex(pd.date_range(start=start_date, end=end_date, freq='B')).fillna(method='ffill') for i in range(len(dfs))]

    files_name = [f for f in listdir('..\data\DK\stocks') if isfile(join('..\data\DK\stocks', f))]
    stocks_names = [f.split(',')[0].split('_')[2] for f in files_name] 
    danish_stocks_df = [pd.read_csv('..\data\DK\stocks\\'+f, sep=',', index_col=0, parse_dates=['time']) for f in files_name]

    # Reindex the dataframe
    min_index = min(range(len(danish_stocks_df)), key=lambda i: len(danish_stocks_df[i]))
    max_index = max(range(len(danish_stocks_df)), key=lambda i: len(danish_stocks_df[i]))

    start_date = danish_stocks_df[min_index].index[0]
    end_date = danish_stocks_df[max_index].index[-1]

    danish_stocks_df = reindex_stocks(danish_stocks_df, start_date, end_date)

    # Get the close data
    close_dfs = [df['close'].values for df in danish_stocks_df]

    close_df = pd.DataFrame({stock: values for stock, values in zip(stocks_names, close_dfs)}, index=danish_stocks_df[0].index)

    return close_df

if __name__ == '__main__':
    args = parse_args()

    if args.nation == 'DK':
        if args.instument == 'stock':
            preprocessed_df = preprocess_dk_stock_data()
            preprocessed_df.to_csv('..\data\DK\preprocessed_data\danish_closed_stocks.csv')