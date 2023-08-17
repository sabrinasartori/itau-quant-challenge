import requests
from tqdm import tqdm
import pandas as pd

def get_customized_news(stock, start_date, end_date, n_news, api_key, offset = 0):
    url = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={stock}&limit={n_news}&offset={offset}&from={start_date}&to={end_date}'
    news_json = requests.get(url).json()
    
    news = []
    news_complete = []
    
    for i in range(len(news_json)):
        title = news_json[-i]['title']
        complete = news_json[-i]
        news.append(title)
        news_complete.append(complete)
    
    return news, news_complete

def get_news(
    stock : str,
    start_year :int = 2019,
    end_year : int = 2023,
    api_key : str = '64d11b117233d0.77790833'
):
    news, news_complete = [], []

    for year in (range(start_year, end_year)):
        for month in tqdm(range(1,13)):
            if month == 12:
                this_news, this_news_complete = get_customized_news(
                    stock,
                    start_date = f'{year}-{12}-01',
                    end_date = f'{year + 1}-0{1}-01',
                    n_news = 1000,
                    api_key = api_key
                )

            else:
                this_month = month
                next_month = month + 1
                if month < 10 :
                    this_month = f'0{this_month}'

                if next_month < 10:
                    next_month = f'0{next_month}'

                this_news, this_news_complete = get_customized_news(
                    stock,
                    start_date = f'{year}-{this_month}-01',
                    end_date = f'{year}-{next_month}-01',
                    n_news = 1000,
                    api_key = api_key
                )

            news += (this_news)
            news_complete += (this_news_complete)
            
    return news, news_complete

def filter_news_with_name (
    news_complete,
    name : str
):
    news_with_name = []

    for new in news_complete:
        if name.lower() in new['title'].lower():
            news_with_name.append(new)

    return news_with_name

def save_company_news_df(
    news : pd.DataFrame,
    company : str
):
    news.to_pickle(f"data/{company}_news.pkl")

def read_company_news_df(
    company: str
):
    df = pd.read_pickle(f"data/{company}_news.pkl")
    df["date"] = pd.to_datetime(df["date"])

    return df