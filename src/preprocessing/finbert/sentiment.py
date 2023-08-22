from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from src.preprocessing.finbert.get_news import get_news, filter_news_with_name

def generate_sentiment_from_title(
    news : List,
    tokenizer, 
    finbert
):
    labels = {0:'neutral', 1:'positive',2:'negative'}

    sent_val = list()
    for new in tqdm(news):
        inputs = tokenizer(new['title'], return_tensors="pt", padding=True)
        outputs = finbert(**inputs)[0]
    
        val = labels[np.argmax(outputs.detach().numpy())]
        sent_val.append(val)

    return sent_val

def generate_news_df(
    ticker : str,
    usual_name : str,
    finbert, 
    tokenizer,
    save : bool = False
):
    
    print(f"Fetching news for {usual_name}...")
    news, news_complete = get_news(ticker)
    news_with_name = filter_news_with_name(news_complete, usual_name)
    print(f"{len(news_with_name)} news were found related to {ticker} ticker")

    print(f"Generating sentiment for {usual_name}...")
    sentiment = generate_sentiment_from_title(
        news_with_name,
        tokenizer,
        finbert
    )

    news_df = pd.DataFrame(news_with_name)[["date","title","content"]]

    news_df["sentiment"] = sentiment

    return news_df