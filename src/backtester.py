from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List

class Backtester:
    def __init__(self, 
                 universe : List[str],
                 y_pred_train : pd.Series,
                 y_pred_test : pd.Series,
                 y_train : pd.Series,
                 y_test : pd.Series
                ) -> None:
        self.universe = universe
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        self.y_train = y_train
        self.y_test = y_test

    def backtest(self, 
                 fees : float = 0,):
        num_trades = {}

        periods = ["train", "test", "val", "full_period"]

        self.pnls = {}
        self.sharpes = {}
        self.p_values = {}
        self.pnl_total = {}

        for period in periods:

            for ticker in tqdm(self.universe):
                if ticker not in ["AMZN", "GOOGL", "AAPL", "MSFT", "FB"]:
                    continue
                company_train = self.y_pred_train\
                    .index\
                    .get_level_values("ticker") == ticker


                company_test = self.y_pred_test\
                    .index\
                    .get_level_values("ticker") == ticker

                assert period in ["train", "val", "test", "full_period"], "period must be one of train, val, test, full_period"
                
                if period == "train":
                    y_company = self.y_train[company_train]\
                        .sort_index(level = "Date")
                    y_pred_company = self.y_pred_train[company_train]\
                        .sort_index(level = "Date")

                elif period == "full_period":
                    y_company = pd.concat([
                        self.y_train[company_train], 
                        self.y_test[company_test]
                    ])
                    y_pred_company = pd.concat([
                        self.y_pred_train[company_train],
                        self.y_pred_test[company_test]
                    ])
                    
                else: 
                    y_company = self.y_test[company_test]
                    y_pred_company = self.y_pred_test[company_test]

                    if period == "val":
                        idx = y_company\
                        .index\
                        .get_level_values("Date") < "2023"
                    
                        y_company = y_company[idx]
                        y_pred_company = y_pred_company[idx]

                    else: # period == "test"
                        idx = y_company\
                        .index\
                        .get_level_values("Date") > "2023"
                    
                        y_company = y_company[idx]
                        y_pred_company = y_pred_company[idx]
                    

                avg_price = y_pred_company\
                    .pct_change()\
                    .rolling(20)\
                    .mean()

                std_price = y_pred_company\
                    .pct_change()\
                    .rolling(20)\
                    .std()
                

                positions = pd.Series(
                    np.nan,
                    index = y_pred_company.index,
                    name = "position"
                )
                
                positions.loc[y_pred_company.pct_change() < 0]  = 1
                positions.loc[y_pred_company.pct_change() > 0] = -1 


                positions = positions\
                    .fillna(0)
                
                num_long = (positions == 1).sum()/positions.shape[0]
                num_short = (positions == -1).sum()/positions.shape[0]
                mask = positions.rolling(2)\
                    .sum() == 0
                trades = positions[mask]


                num_trades[ticker] = {
                    "num_long" : num_long,
                    "num_short" : num_short,
                    "n_trades": trades.shape[0],
                    "n_days" : positions.shape[0]
                }
                
                positions_df = pd.concat([y_company, positions], axis = 1)

                positions_df = positions_df\
                    .reset_index(level = "ticker")

                pnl = pd.Series(
                    0,
                    index = positions_df.index,
                    name = "pnl"
                )

                last_buy_price = np.nan
                cash = 0
                position = 0 

                if fees == 0:
                    returns = positions_df["price"]\
                        .pct_change() * positions_df["position"].shift(1)
                
                else:
                    for date, row in positions_df.iterrows():
                        first_trade = np.isnan(last_buy_price) and row["position"] != 0 
                        has_position = not np.isnan(last_buy_price) and position != 0
                        changing_position = has_position and row["position"] != position
                        holding_position = has_position and row["position"] == position

                        if (first_trade):
                            last_buy_price = row["price"] * (1 + fees * position)
                            position = row["position"]

                        if (changing_position):
                            cash += (row["price"] / last_buy_price - 1 )*position
                            # last_buy_price = np.nan
                            position = row["position"]
                            
                            if (position == 0):
                                last_buy_price = np.nan
                            else:
                                last_buy_price = row["price"] * (1 + fees * position)

                        if holding_position:
                            pnl.loc[date] = (row["price"]- last_buy_price)/last_buy_price * position + cash

                        else:
                            pnl.loc[date] = cash

                    if row["position"] != 0:
                        cash += (row["price"]- last_buy_price)/last_buy_price * position

                        pnl.loc[date] = cash

                    returns = pnl.diff()

                if self.pnls.get(period) is None:
                    self.pnls[period] = {}
                self.pnls[period][ticker] = returns
            
            pnl_total = pd.Series(
                0,
                index = self.pnls[period]["AAPL"].index
            )
            for k, pnl in self.pnls[period].items():

                if k not in ["AAPL", "GOOGL", "FB", "AMZN", "MSFT"]:
                    continue

                pnl_total += pnl

            self.pnl_total[period] = pnl_total
            T = len(pnl_total)
            sr = pnl_total.mean() / pnl_total.std()

            self.sharpes[period] = sr * np.sqrt(252)

            gamma_3 = pnl_total.skew()
            gamma_4 = pnl_total.kurtosis()

            denominator =  1 - gamma_3 * sr + (gamma_4 - 1)/4 * sr**2
            denominator = np.sqrt(denominator)
            expected_sr = 0
            psr = (sr - expected_sr) * np.sqrt(T - 1) / denominator
            psr = norm.cdf(psr)

            self.p_values[period] = 1 - psr

    def get_backtest_results(self):
        return {
            "sharpes": self.sharpes,
            "p-values": self.p_values,
        }
    
    def get_backtest_pnl_by_company(self):
        return self.pnls
    
    def get_backtest_pnl(self):
        return self.pnl_total