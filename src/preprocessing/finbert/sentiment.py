from tqdm import tqdm
from typing import List
import numpy as np

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