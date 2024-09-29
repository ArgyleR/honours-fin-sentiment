import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.model_selection import train_test_split
import model_helper as mh
from torch.nn.utils.rnn import pad_sequence
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
import ijson
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import utils.text_selection_methods as txtsm
from tqdm import tqdm

def read_ts_dir(path):
    """
    Helper function for reading a directory of time series csv files
    Joins all csv files into a single name. Adds the ticker column to denote which file it came from
    @param path str: the path to the directory being read
    
    @return pd.DataFrame: The combined dataframe with all stock price information from the give directory
    """
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

    # List to hold dataframes
    df_list = []

    for file in all_files:
        df = pd.read_csv(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        df['ticker'] = filename
        df_list.append(df)

    # Concatenate all dataframes in the list
    return pd.concat(df_list, ignore_index=True)

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_tokenizer, ts_col: str="time_series", text_col: str="text_series", label_col: str="label"):
        self.df = df
        self.text_tokenizer = text_tokenizer
        self.max_length = 512
        self.texts = df[text_col].tolist() #list of list of texts (each row has a list of associated texts)
        self.time_series = df[ts_col].tolist() #list of list of time series (each row has multiple time series [may be 1 for univariate])
        self.labels = df[label_col].tolist() #list of labels (single value as the whole row is pos or neg pair)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        past_time_values = torch.tensor(self.time_series[idx], dtype=torch.float32)
        past_observed_mask = torch.ones(1, len(past_time_values), dtype=torch.long)
        past_time_features = torch.tensor([self.df.iloc[idx]["ts_past_features"]], dtype=torch.float32)

        ts_data = [{
            "past_time_values": past_time_values,
            "past_observed_mask": past_observed_mask,
            "past_time_features": past_time_features
        }]

        #loop over all text and tokenize
        list_texts = self.texts[idx]
        text_data = [self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length, padding='max_length') for text in list_texts]
        
        input_ids = torch.stack([item['input_ids'].squeeze(0) for item in text_data])  # Shape: [number_of_texts, length_of_text]
        attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in text_data])  # Shape: [number_of_texts, length_of_text]
        text_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return ts_data, text_data, label

def process_EDT_json_to_dataframes(json_file):
    # Helper method used to read the Event Driven Trading data file 
    ts_data = []
    text_data = []

    # ijson used to parse incrementally
    with open(json_file, 'r') as file:
        
        objects = ijson.items(file, 'item')
        
        for row in objects:
            # Extract text data: ticker, pub_time (date), title and text
            #This is specific to EDT
            if 'labels' in row and row['labels']:
                ticker = row['labels']['ticker']
                pub_time = row['pub_time']
                title = row['title']
                text = row['text']

                # Append text-related data
                text_data.append({
                    'ticker': ticker,
                    'date': pub_time,
                    'title': title,
                    'text': text
                })

                #We want a new row for each day to be consistent with other datasets
                for day in ['1day', '2day', '3day']:
                    end_price = row['labels'].get(f'end_price_{day}')
                    end_time = row['labels'].get(f'end_time_{day}')

                    if end_price and end_time:
                        ts_data.append({
                            'ticker': ticker,
                            'Date': end_time,
                            'Close': end_price
                        })

    # Create DataFrames
    text_df = pd.DataFrame(text_data)
    text_df['id'] = pd.Series(range(1, len(text_df) + 1))
    ts_df = pd.DataFrame(ts_data)
    ts_df['Close'] = pd.to_numeric(ts_df['Close'], errors='coerce')  
    ts_df['id'] = pd.Series(range(1, len(ts_df) + 1))


    return text_df, ts_df

def wrangle_data(data_source:dict):
    """
    helper function used to take raw data files and convert them into the text and time series dfs
    @param data_source dict: the dictionary containing the data name, text path and time series path

    @return (pd.DataFrame, pd.DataFrame): a tuple of the text and time series dfs
    """
    data_set = data_source["name"]
    text_path = data_source["text_path"]
    ts_path = data_source["ts_path"]
    text_date_col = data_source['text_date_col']
    ts_date_col = data_source['ts_date_col']

    if data_set == "stock_emotion":
        text_df = pd.read_csv(text_path)
        text_df.rename(columns={"processed": "text"}, inplace=True)
        ts_df = read_ts_dir(ts_path)
    elif data_set == 'stock_net':
        text_df = pd.read_csv(text_path)
        ts_df = read_ts_dir(ts_path)
    elif data_set == 'EDT':
        #We treat the text path the same as the TS path as it is one file for EDT
        text_df, ts_df = process_EDT_json_to_dataframes(text_path)
        ts_df = read_ts_dir(ts_path)
    else:
        raise ValueError("The dataset name is not known or incorrectly spelled")
    
    text_df['date'] = pd.to_datetime(text_df[text_date_col], utc=True).dt.tz_localize(None) #remove time zone information
    ts_df['date'] = pd.to_datetime(ts_df[ts_date_col], utc=True).dt.tz_localize(None)

    return text_df, ts_df

def create_time_series_df(df, k, mode='start'):
    new_df = []
    
    # Ensure the DataFrame is sorted by ticker and Date
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].reset_index(drop=True)
        
        if len(ticker_df) < k:
            continue  # Skip if there are not enough data points
        
        for i in range(len(ticker_df) - k):
            #change target_date instead of the actual sequence 
            start_idx = i
            end_idx = start_idx + k

            if mode == 'start':
                target_date_idx = start_idx
            elif mode == 'middle':
                target_date_idx = start_idx + k // 2
            elif mode == 'end':
                target_date_idx = end_idx - 1
            else:
                raise ValueError("Invalid mode. Choose 'start', 'middle', or 'end'.")
            
            #time_series = ticker_df.iloc[i:end_idx]['Close'].tolist()
            #ts_past_features = ticker_df.iloc[i:end_idx]['Date'].apply(lambda x: [x.year, x.month, x.day]).tolist()
            
            if len(ticker_df.iloc[start_idx:end_idx]['Close'].tolist()) != k:
                print("updating length!!" * 100)
                
                end_idx += 1 #add additional day to meet the k requirement

            
            time_series = ticker_df.iloc[start_idx:end_idx]['Close'].tolist()
            ts_past_features = ticker_df.iloc[start_idx:end_idx]['Date'].apply(lambda x: [x.year, x.month, x.day]).tolist()
            target_date = ticker_df.iloc[target_date_idx]['Date']
            
            new_df.append({
                'ticker': ticker,
                'target_date': target_date,
                'time_series': time_series,
                'ts_past_features': ts_past_features
            })
    result_df = pd.DataFrame(new_df)
    return result_df

def create_text_series_df(df, k=1, mode='ALL', top_n=None, text_col='text', time_col='created_at'):
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.date
    new_df = []

    # Ensure the DataFrame is sorted by ticker and created_at
    df = df.sort_values(['ticker', time_col]).reset_index(drop=True)

    # Group by ticker
    if mode == 'embedding_diversity':
        embedding_diversity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    for ticker, group in df.groupby('ticker'):
        # Iterate through each date in the group
        for i, target_date in enumerate(group[time_col].unique()):
            end_date = target_date + pd.Timedelta(days=k-1)
            # Filter rows within the k-day window
            window = (group[time_col] >= target_date) & (group[time_col] <= end_date)
            window_df = group[window]
            
            # Collect text series, ids, and text_dates
            text_series = list(set(window_df[text_col].tolist())) #set as we only want unique text (otherwise some will be repeat)
            ids = window_df['id'].tolist()
            text_dates = window_df[time_col].tolist()

            if mode == 'vader':
                text_series, ids, text_dates = txtsm.apply_vader_ranking(text_series, ids, text_dates, top_n)

            elif mode == 'clustering':
                text_series, ids, text_dates = txtsm.apply_clustering_ranking(text_series, ids, text_dates, top_n)

            elif mode == 'embedding_diversity':
                
                text_series, ids, text_dates = txtsm.apply_embedding_diversity_ranking(text_series, ids, text_dates, embedding_diversity_model, top_n)

            if mode == 'TFIDF' and top_n is not None:
                text_series, ids, text_dates = txtsm.apply_tfidf_ranking(text_series, ids, text_dates, top_n)


            # Append to new dataframe
            new_df.append({
                'ids': ids,
                'ticker': ticker,
                'target_date': target_date,
                'end_date': end_date,
                'text_series': text_series,
                'text_dates': text_dates
            })
    
    return pd.DataFrame(new_df)

def process_windows(text_df, ts_df, ts_window:int, ts_mode:str, text_window:int, text_selection_method:tuple, text_col, text_time_col, ts_time_col:str):
    text_df = create_text_series_df(df=text_df, k=text_window, mode=text_selection_method[0], top_n=text_selection_method[1], text_col=text_col, time_col=text_time_col)
    ts_df = create_time_series_df(ts_df, k=ts_window, mode=ts_mode)
    ts_df = ts_df[ts_df['time_series'].apply(len)==ts_window].reset_index(drop=True)

    #we now have the windowed text and ts (including multiple days); not yet implemented for multiple tickers
    return text_df, ts_df
    
def create_pairs(text_df, ts_df, negatives_creation: str, negative_label: int = -1, random_state:int=42):
    # Create positive pairs where ticker matches and target_date of ts_df == end_date of text_df 
    # Add a unique identifier column
    text_df['text_id'] = 'text' + (text_df.index + 1).astype(str)
    ts_df['ts_id'] = 'ts' + (ts_df.index + 1).astype(str)

    merged_df = pd.merge(
        text_df,
        ts_df,
        left_on=['ticker', 'end_date'],
        right_on=['ticker', 'target_date'],
        suffixes=('_text_df', '_ts_df')
    )
    merged_df["label"] = 1  # All so far are positive pairs

    # Create negative pairs
    # Naive method where ticker isn't the same and date isn't the same
    if negatives_creation[0] == 'naive':
        days_away = negatives_creation[1]
        text_df['end_date'] = pd.to_datetime(text_df['end_date'], utc=True).dt.date

        # Create pairs where tickers aren't the same and end_date is x days away
        negative_pairs = []
        for idx, row in merged_df.iterrows():
            ticker = row['ticker']
            end_date = row['end_date']

            # Get potential negative samples that have different tickers and are at least x days away
            potential_negatives = text_df[
                (text_df['ticker'] != ticker) &
                ((text_df["end_date"] < end_date - pd.Timedelta(days=days_away)) |
                 (text_df["end_date"] > end_date + pd.Timedelta(days=days_away)))
            ]

            if not potential_negatives.empty:
                # Randomly select a negative sample
                negative_sample = potential_negatives.sample(n=1, random_state=random_state + idx).iloc[0]
                negative_pairs.append({
                    'text_id': negative_sample['text_id'],
                    'ids': negative_sample['ids'],
                    'ticker': negative_sample['ticker'],  # Using the different ticker
                    'target_date_text_df': negative_sample['target_date'],  # Keeping the same target_date for consistency
                    'target_date_ts_df': row['target_date_ts_df'],  # Keeping the same target_date for consistency
                    'end_date': negative_sample["end_date"],  # Use the end_date from the negative sample
                    'text_dates': negative_sample["text_dates"],  # Use the text_dates from the negative sample
                    'text_series': negative_sample['text_series'],  # Use the text_series from the negative sample
                    'ts_id': row['ts_id'],
                    'time_series': row['time_series'],  # Keeping the same time_series for consistency
                    'label': negative_label,  # Label for negative pair
                    "ts_past_features": row["ts_past_features"]  # Keeping the same ts_past_features for consistency
                })

    # Convert negative pairs to DataFrame
    negative_df = pd.DataFrame(negative_pairs)

    # Append the negative pairs to the original DataFrame
    return pd.concat([merged_df, negative_df], ignore_index=True)

def normalize_ts(df):
    c_col='Close'
    t_col='ticker'
    for ticker in df[t_col].unique():
        df.loc[df[t_col] == ticker, c_col] = (df.loc[df[t_col] == ticker, c_col] - df.loc[df[t_col] == ticker, c_col].min()) / (df.loc[df[t_col] == ticker, c_col].max() - df.loc[df[t_col] == ticker, c_col].min())
    return df

def get_data_loaders(dfs, text_tokenizer, batch_size, num_workers):
    final_dataloaders = []

    for df in dfs:
        dataset = CustomDataset(df=df, text_tokenizer=text_tokenizer)
        dataloader  = DataLoader(dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        final_dataloaders.append(dataloader)

    return final_dataloaders

def correct_negative_labels(df:pd.DataFrame, negative_label:int, label_column:str='label'):
    #helper method use to correct negative pairs (default is -1) as some criterion will expect different negative labels -1 or 0

    df[label_column] = df[label_column].replace(-1, negative_label)
    return df

def normalize_ts_features(df, column_name):
    # Flatten all dates and separate into year, month, and day lists
    years, months, days = [], [], []
    
    for date_list in df[column_name]:
        for year, month, day in date_list:
            years.append(year)
            months.append(month)
            days.append(day)
    
    # Normalize years, months, and days
    years_min, years_max = min(years), max(years)
    months_min, months_max = min(months), max(months)
    days_min, days_max = min(days), max(days)
    
    # Handle the case where there's only one unique year
    if years_min == years_max:
        normalized_years = {year: 0.0 for year in set(years)}
    else:
        normalized_years = {year: (year - years_min) / (years_max - years_min) for year in set(years)}
    
    normalized_months = {month: (month - months_min) / (months_max - months_min) for month in set(months)}
    normalized_days = {day: (day - days_min) / (days_max - days_min) for day in set(days)}
    
    # Apply the normalization back to the original structure
    normalized_dates = []
    
    for date_list in df[column_name]:
        normalized_date_list = [[
            normalized_years[year],
            normalized_months[month],
            normalized_days[day]
        ] for year, month, day in date_list]
        normalized_dates.append(normalized_date_list)
    
    return normalized_dates

def subset_data_helper(data_source, text_df, ts_df):
    #filter EDT data as it is too large to work with
    if data_source['name'] == 'EDT':
        with open('./data/tickers_selected/edt.txt', 'r') as file:
            edt_keep_tickers = [line.strip() for line in file]
        text_df = text_df[text_df['ticker'].isin(edt_keep_tickers)].reset_index(drop=True)
        ts_df = ts_df[ts_df['ticker'].isin(edt_keep_tickers)].reset_index(drop=True)

    sorted_tickers = text_df['ticker'].value_counts().index
    #check that the tickers we are selecting actual have a ts pair
    filtered_tickers = [ticker for ticker in sorted_tickers if ticker in ts_df['ticker'].unique()]
    top_tickers = filtered_tickers[:3]
    text_df = text_df[text_df['ticker'].isin(top_tickers)].reset_index(drop=True)
    ts_df = ts_df[ts_df['ticker'].isin(top_tickers)].reset_index(drop=True)

    return text_df, ts_df

def split_data(train_dates, test_dates, text_df, ts_df, text_date_col, ts_date_col, random_state):
    train_start, train_end = pd.to_datetime(train_dates.split(' - '), format='%d/%m/%Y')
    test_start, test_end = pd.to_datetime(test_dates.split(' - '), format='%d/%m/%Y')
    
    text_train = text_df[(pd.to_datetime(text_df[text_date_col]) >= train_start) & 
                         (pd.to_datetime(text_df[text_date_col]) <= train_end)]
    
    text_test_val = text_df[(pd.to_datetime(text_df[text_date_col]) >= test_start) & 
                            (pd.to_datetime(text_df[text_date_col]) <= test_end)]
    
    text_val = text_test_val.sample(frac=0.5, random_state=random_state)
    text_test = text_test_val.drop(text_val.index)
    
    # Splitting ts_df
    ts_train = ts_df[(pd.to_datetime(ts_df[ts_date_col]) >= train_start) & 
                     (pd.to_datetime(ts_df[ts_date_col]) <= train_end)]
    
    ts_val_test = ts_df[(pd.to_datetime(ts_df[ts_date_col]) >= test_start) & 
                        (pd.to_datetime(ts_df[ts_date_col]) <= test_end)]
    
    text_train = text_train.reset_index(drop=True)
    ts_train = ts_train.reset_index(drop=True)
    text_val = text_val.reset_index(drop=True)
    ts_val_test = ts_val_test.reset_index(drop=True)
    text_test = text_test.reset_index(drop=True)

    text_train['split'] = 'train'
    ts_train['split'] = 'train'
    text_val['split'] = 'val'
    ts_val_test['split'] = 'val_test'
    text_test['split'] = 'val_test'

    
    # Creating the result dictionary
    result = {
        'train': (text_train, ts_train),
        'validation': (text_val, ts_val_test),
        'test': (text_test, ts_val_test)
    }
    
    return result

def get_data(text_tokenizer, 
             data_source:dict, 
             ts_window:int=5, 
             ts_mode:str="middle", 
             text_window:int=1, 
             text_selection_method:tuple=('TFIDF', 5), 
             negatives_creation:tuple=("naive", 31), 
             batch_size:int=16, 
             num_workers:int=6, 
             loaders:bool=True,
             subset_data:bool=False, 
             random_state:int=42):
    
    text_df, ts_df = wrangle_data(data_source) #returns df with id, ticker, Date, text, close

    #filter data to subset for faster training
    if subset_data:
        text_df, ts_df = subset_data_helper(data_source=data_source, text_df=text_df, ts_df=ts_df)
        

    text_date_col = data_source['text_date_col']
    ts_date_col = data_source["ts_date_col"]
    text_col = data_source["text_col"]
    #make sure date columns are datetimes
    text_df[text_date_col] = pd.to_datetime(text_df[text_date_col], utc=True).dt.date
    ts_df[ts_date_col] = pd.to_datetime(ts_df[ts_date_col], utc=True).dt.date


    
    #normlaize ts_df
    ts_df = normalize_ts(ts_df)
    
    #TODO test train split here
    train_dates = data_source['train_dates']
    test_dates = data_source['test_dates']
    split_data_dict = split_data(train_dates=train_dates, 
                                 test_dates=test_dates, 
                                 text_df=text_df, 
                                 ts_df=ts_df, 
                                 text_date_col=text_date_col, 
                                 ts_date_col=ts_date_col, 
                                 random_state=random_state)
    
    final_dfs = []
    final_dataloaders = []
    for key in split_data_dict.keys():
        text_df, ts_df = split_data_dict[key]
        
        #convert df to id, tickers:[list], start_date, texts:list, time_series:list, past_time_features:[list]
        text_df, ts_df = process_windows(text_df=text_df, 
                                         ts_df=ts_df, 
                                         ts_window=ts_window,
                                         ts_mode=ts_mode, 
                                         text_window=text_window, 
                                         text_selection_method=text_selection_method, 
                                         text_col=text_col, 
                                         text_time_col=text_date_col, 
                                         ts_time_col=ts_date_col)
        #padd text_series to have empty strings for collate_fn


        df = create_pairs(text_df=text_df, ts_df=ts_df, negatives_creation=negatives_creation, random_state=random_state)
        max_length = df['text_series'].apply(len).max()
        df['text_series'] = df['text_series'].apply(lambda x: x + [''] * (max_length - len(x)))

        # Apply normalization to each row
        df['original_ts_past_features'] = df['ts_past_features']
        df['ts_past_features'] = normalize_ts_features(df, 'ts_past_features')

        


        if loaders:
            dataset = CustomDataset(df=df, text_tokenizer=text_tokenizer)
            dataloader  = DataLoader(dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
            final_dataloaders.append(dataloader)
        else:
            final_dfs.append(df)
    
    if loaders:
        return final_dataloaders
    else:
        return final_dfs