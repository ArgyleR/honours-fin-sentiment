import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.model_selection import train_test_split
import model_helper as mh

EXPECTED_COLS = {"id", "ticker", "start_date", "text", "time_series", "label"}

ROOT_PATH = "drive/MyDrive/Honors_Fin_Sentiment_2024_Eoin/implementation/"

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_tokenizer, ts_col: str="time_series", text_col: str="text", label_col: str="label"):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.texts = df[text_col].tolist()
        self.time_series = df[ts_col].tolist()
        self.labels = df[label_col].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        time_series = torch.tensor(self.time_series[idx], dtype=torch.float32)

        text = self.texts[idx]
        text_tokenized = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return time_series, text_tokenized['input_ids'].squeeze(0), text_tokenized['attention_mask'].squeeze(0), label

def read_stock_emotion_ts(path):
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

    # List to hold dataframes
    df_list = []

    for file in all_files:
        df = pd.read_csv(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        df['ticker'] = filename
        df_list.append(df)

    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df


def _helper_wrangle_dataset(text_path, ts_path, data_set):
    assert data_set != None, "Dataset must be given an option"

    if data_set == "stock_emotion":
        text_df = pd.read_csv(text_path)
        text_df.rename(columns={"processed": "text"}, inplace=True)

        ts_df = read_stock_emotion_ts(ts_path)
    else:
        raise ValueError("The dataset target is not known or incorrectly spelled")
    

    return text_df, ts_df

def _helper_create_k_day_ts_window(df, k=5, mode='right'): #TODO check returns right format
    """
    @param Pandas.DataFrame df: The dataframe with a time series to generate windows over
    @param Int k: The number of days the window should be covering

    @return Pandas.Dataframe: The dataframe changed so that the time_series is over a k day period
    """
    assert mode in ['left', 'right', 'balanced'], "Mode must be 'left', 'right', or 'balanced'"

    # Ensure the Date column is a datetime type
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['ticker', 'Date'])

    # Create a list to store the new rows
    data = []

    if mode == 'right':
        offset = 0
    elif mode == 'left':
        offset = -k + 1
    elif mode == 'balanced':
        offset = -(k // 2)

    # Group by ticker and process each group
    for ticker, group in df.groupby('ticker'):
        dates = group['Date'].reset_index(drop=True)
        prices = group['Close'].reset_index(drop=True)

        # Iterate over the group to create the new rows
        for i in range(len(group)):
            start_idx = i + offset
            end_idx = start_idx + k

            if start_idx < 0 or end_idx > len(group):
                continue

            row = [ticker, dates.iloc[i], prices.iloc[start_idx:end_idx].tolist()]
            data.append(row)

    # Create the new dataframe
    columns = ['ticker', 'start_date', 'time_series']

    new_df = pd.DataFrame(data, columns=columns)

    

    return new_df


def _helper_text_window_handling(df, k): #TODO Check returns all columns not just in groupby agg
    """
    @param Pandas.DataFrame df: The dataframe with a time series to generate windows over
    @param Int k: The number of days the window should be covering

    @return Pandas.Dataframe: The dataframe changed so that the text concatenation is over a k day period
    """
    
    df['date'] = pd.to_datetime(df['date'])
    # Sort the dataframe by ticker and date
    df = df.sort_values(by=['ticker', 'date'])

    # Create a new column to group the dates into L-day periods
    df['group'] = df.groupby('ticker')['date'].transform(lambda x: (x - x.min()).dt.days // k)

    # Concatenate the text within the same ticker and group
    result = df.groupby(['ticker', 'group']).agg({
        'id': 'first',  # Keep the first id in the group
        'date': 'first',  # Keep the first date in the group
        'ticker': 'first',  # Keep the first ticker in the group
        'text': ' '.join  # Concatenate text
    }).reset_index(drop=True)

    return result

def create_negatives(df, days_away=31, negative_label=0):
    """
    @param Pandas.DataFrame df: The time series and text pair dataframe of positives to create negatives based on
    @param Int days_away: The number of days minimum that negative pairs should be taken from

    @return Pandas.DataFrame: The dataframe with negative pairs added
    """
    assert set(df.columns) == EXPECTED_COLS, "columns for df are incorrect"

    df['start_date'] = pd.to_datetime(df['start_date'])

    # List to store new negative pairs
    negative_pairs = []

    for idx, row in df.iterrows():
        ticker = row['ticker']
        start_date = row['start_date']

        # Get potential negative samples that are at least k days apart
        potential_negatives = df[(df['ticker'] != ticker) & (abs((df['start_date'] - start_date).dt.days) >= days_away)]

        if not potential_negatives.empty:
            # Randomly select a negative sample
            negative_sample = potential_negatives.sample(n=1).iloc[0]
            negative_pairs.append({
                'id': row['id'],
                'ticker': row['ticker'],  # Keeping the same ticker for consistency
                'start_date': row['start_date'],  # Keeping the same start date for consistency
                'text': row['text'],
                'time_series': negative_sample['time_series'],
                'label': negative_label  # Label for negative pair
            })

    # Create a DataFrame for negative pairs
    negative_df = pd.DataFrame(negative_pairs)

    # Append the negative pairs to the original DataFrame
    augmented_df = pd.concat([df, negative_df], ignore_index=True)

    return augmented_df

def collate_fn(batch):
    text_inputs = {}
    ts_inputs = []
    labels = []
    for text_input, ts_input, label in batch:#TODO something here is wrong, soemthing to do with how batch is being passed. Look at previous work
        for k, v in text_input.items():
            if k not in text_inputs:
                text_inputs[k] = []
            text_inputs[k].append(v.squeeze(0))
        ts_inputs.append(ts_input)
        labels.append(label)

    for k, v in text_inputs.items():
        text_inputs[k] = torch.stack(v)

    ts_inputs = torch.stack(ts_inputs)
    labels = torch.tensor(labels)
    return text_inputs, ts_inputs, labels

def _helper_get_tvt_splits(df: pd.DataFrame, text_tokenizer):
    dataset = CustomDataset(df, text_tokenizer=text_tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset


def _helper_generate_synthetic_benchmark(model:mh.ContrastiveLearningModel, tickers=['APPL', 'TSLA', 'AMZN', 'GOOGL', 'ABNB','MSFN'], 
                                         examples_per_ticker=500, ts_len=5, 
                                         negative_words=['loss', 'decline', 'drop', 'decrease', 'negative'], 
                                         positive_words=['gain', 'increase', 'rise', 'growth', 'positive'],
                                         negative_label=0,
                                         batch_size=32,
                                         num_workers=6):
    """
    @param List tickers: A list of string ticker markers to generate over
    @param Int k: an  integer value for the number of days to generate examples over for each ticker
    @param Int l: an integer value for 
    """
    def generate_time_series(start_value, days, direction):
        return [start_value + i * direction for i in range(days)]
    data = []
    start_date = datetime.now()
    i = 0
    for ticker in tickers:
        for day in range(examples_per_ticker):
            date = start_date - timedelta(days=day)

            # Randomly decide text sentiment and time series direction
            is_text_positive = random.choice([True, False])
            is_time_series_rising = random.choice([True, False])

            if is_text_positive:
                text_words = random.sample(positive_words, k=5)
                text_label = "positive"
            else:
                text_words = random.sample(negative_words, k=5)
                text_label = "negative"

            if is_time_series_rising:
                time_series = generate_time_series(random.randint(50, 100), ts_len, 1)
                series_label = "rising"
            else:
                time_series = generate_time_series(random.randint(50, 100), ts_len, -1)
                series_label = "decreasing"

            # Determine if the pair is positive or negative
            if (text_label == "negative" and series_label == "decreasing") or (text_label == "positive" and series_label == "rising"):
                label = 1
            else:
                label = negative_label

            text = ' '.join(text_words)
            data.append([i, ticker, date, text, time_series, label])
            i += 1

    df = pd.DataFrame(data, columns=['id', 'ticker', 'Start Date', 'text', 'time_series', 'label'])
    
    def make_2d(lst):
        return [[x] for x in lst]
    df["time_series"] = df["time_series"].apply(make_2d)

    #dataset = CustomDataset(df, text_tokenizer=model.get_text_encoder())

    
    train_dataset, val_dataset, test_dataset = _helper_get_tvt_splits(df, text_tokenizer=model.get_text_tokenizer())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader

def get_data(data_source: str = 'synthetic', model=None, text_window:int =5, ts_window:int=5, days_away: int=31, negative_label: int=0, text_concatenation: str='flat', negative_creation: str='',
              batch_size=32, num_workers = 6):
    """
    @param String data_source: The data to extract the ts and text data
    @param ContrastiveLearningModel model: The model used (encoders for the text and ts are needed)
    @param Integer text_window: The span of time that text is combined over
    @param Integer ts_windo: The span of time that time series are combined over
    @param Integer days_away: For simple negative creation, this is the number of days apart each negative pair is from the true time
    @param Integer negative_label: The label for negative examples (cosine similarity from pytorch requires -1)
    @param String text_concatenation: The method for combining text (simple concatenation etc)
    @param String negative_creation: The method for creating negatives (based on different distributions from true time series, different times / stocks etc)


    @return (DataLoader trainloader, DataLoader validloader, Dataloader testloader): The dataloader object consisting returning the following columns: text, time_series, label
    """



    if data_source == 'synthetic': return _helper_generate_synthetic_benchmark(model=model, negative_label=negative_label, batch_size=batch_size, ts_len=ts_window)


    text_df, ts_df = _helper_wrangle_dataset(text_path="data/stock_emotions/tweet/train_stockemo.csv",
                                             ts_path="data/stock_emotions/price/",
                                             data_set="stock_emotion")
    
    
    #handle data wrangling into the format of id, ticket, start_date, text, time_series(list of floats)

    
    text_df = _helper_text_window_handling(text_df, text_window)
    ts_df = _helper_create_k_day_ts_window(ts_df, ts_window)

    
    df = pd.merge(text_df, ts_df, left_on=['ticker', 'date'], right_on=['ticker', 'start_date'])
    df = df.drop(columns=['date'])
    df['label'] = 1
    

    df = create_negatives(df, days_away, negative_label=negative_label)

    #handle split, Dataset and DataLoader


    return df
