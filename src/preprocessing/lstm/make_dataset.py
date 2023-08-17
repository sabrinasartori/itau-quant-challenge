import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing import List

def generate_tensor_lstm(
    returns_df : pd.DataFrame,
)->List[torch.Tensor]:

    x_arr = returns_df\
        .iloc[:,1:]\
        .to_numpy()
    
    
    # y_arr = returns_df\
    #     .iloc[:, 0]\
    #     .to_numpy()
    
    new_shape = (x_arr.shape[0], x_arr.shape[1], 1)
    X = torch\
        .from_numpy(x_arr.reshape(new_shape))\
        .type(torch.Tensor)

    # y = torch\
    #     .from_numpy(y_arr.reshape([y_arr.shape[0], 1]))\
    #     .type(torch.Tensor)

    return X

class StockReturnsDataset(Dataset):
    def __init__(self, 
                 prices_df : pd.DataFrame,
                 lookback : int = 10) -> None:
        super().__init__()
        self.lookback = lookback

        past_returns = prices_df\
            .pct_change()\
            .dropna()
        
        future_returns = prices_df\
            .pct_change(1)\
            .shift(-1)\
            .dropna()\
            .Close
        
        for i in range(lookback, 0, -1):
            past_returns[f"d-{i}"] = past_returns\
                ["Close"]\
                .shift(i)

        past_returns = past_returns.dropna()
        
        idx = future_returns.index\
            .intersection(past_returns.index)
        
        idx = idx.sort_values()
        
        past_returns = past_returns\
            .reindex(idx)
        
        self.future_returns = future_returns\
            .reindex(idx)
        
        self.past_returns = past_returns\
            .reindex(idx)
        
        future_returns = future_returns\
            .reindex(idx)\
            .to_numpy()
        
        self.X= generate_tensor_lstm(past_returns)
        
        self.y = torch\
            .from_numpy(future_returns.reshape([future_returns.shape[0], 1]))\
            .type(torch.Tensor)
                
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index):
        return self.X[index], \
            self.y[index]

def load_data(stock, lookback):
    data_raw = stock.values # convert to numpy array
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)

    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    
    return [x_train, y_train, x_test, y_test]

