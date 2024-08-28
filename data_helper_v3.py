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
from sklearn.feature_extraction.text import TfidfVectorizer

def read_stock_net(path):
    pass
def read_ts_dir(path):
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
        #loop over all ts and tensorse them

        #list_past_time_values = [torch.tensor(time_series, dtype=torch.float32) for time_series in self.time_series[idx]]
        #list_past_observed_mask = [torch.ones(1, len(past_time_values)) for past_time_values in list_past_time_values]
        #list_past_time_features = [torch.tensor([past_time_features]) for past_time_features in self.df.iloc[idx]["past_time_features"]]


        #ts_data = {
        #    "list_past_time_values": list_past_time_values,
        #    "list_past_observed_mask": list_past_observed_mask,
        #    "list_past_time_features": list_past_time_features
        #}  

        past_time_values = torch.tensor(self.time_series[idx], dtype=torch.float32)
        past_observed_mask = torch.ones(1, len(past_time_values))
        past_time_features = torch.tensor([self.df.iloc[idx]["ts_past_features"]])

        ts_data = [{
            "past_time_values": past_time_values,
            "past_observed_mask": past_observed_mask,
            "past_time_features": past_time_features
        }]

        #loop over all text and tokenize
        list_texts = self.texts[idx]
        text_data = [self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length, padding='max_length') for text in list_texts]
        text_data = [{
            'input_ids': item['input_ids'].squeeze(0),  # Removes the batch dimension
            'attention_mask': item['attention_mask'].squeeze(0)  # Removes the batch dimension
        } 
        for item in text_data
        ]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return ts_data, text_data, label

def collate_fn(batch):
    ts_data, text_data, labels = zip(*batch)
    
    # Pad sequences to the same length
    #ts_data = pad_sequence(ts_data, batch_first=True, padding_value=0)
    i = 0
    while i < len(text_data):      #loop over all text datas 
        j = 0
        while j < len(text_data[j]):  

            text_data[i][j]["input_ids"] = pad_sequence(text_data[i][j]["input_ids"], batch_first=True, padding_value=0)
            text_data[i][j]["attention_mask"] = pad_sequence(text_data[i][j]["attention_mask"], batch_first=True, padding_value=0)
            j += 1
        i += 1
    
    labels = torch.stack(labels)
    
    return ts_data, text_data, labels

def read_stocknet_text(text_dir):
    dfs = []

    # Walk through the directory and its subdirectories
    for subdir, dirs, files in os.walk(text_dir):
        ticker = os.path.basename(subdir)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                # Read the JSON file into a DataFrame
                df = pd.read_json(file_path)
                dfs.append(df)
                df['ticker'] = ticker

    # Concatenate all the DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def wrangle_data(data_source:dict):
    data_set = data_source["name"]
    text_path = data_source["text_path"]
    ts_path = data_source["ts_path"]

    if data_set == "stock_emotion":
        text_df = pd.read_csv(text_path)
        text_df.rename(columns={"processed": "text"}, inplace=True)

        ts_df = read_ts_dir(ts_path)
    elif data_set == 'stock_net':

        text_df = pd.read_csv(text_path)

        ts_df = read_ts_dir(ts_path)

    else:
        raise ValueError("The dataset target is not known or incorrectly spelled")
    
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])

    return text_df, ts_df

def create_time_series_df(df, k, mode='start'):
    new_df = []
    
    # Ensure the DataFrame is sorted by ticker and Date
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].reset_index(drop=True)
        
        if len(ticker_df) < k:
            continue  # Skip if there are not enough data points
        
        for i in range(len(ticker_df) - k + 1):
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
            
            time_series = ticker_df.iloc[i:end_idx]['Close'].tolist()
            ts_past_features = ticker_df.iloc[i:end_idx]['Date'].apply(lambda x: [x.year, x.month, x.day]).tolist()
            target_date = ticker_df.iloc[target_date_idx]['Date']
            
            new_df.append({
                'ticker': ticker,
                'target_date': target_date,
                'time_series': time_series,
                'ts_past_features': ts_past_features
            })
    
    return pd.DataFrame(new_df)

def create_text_series_df(df, k=1, mode='ALL', top_n=None, text_col='text', time_col='created_at'):
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.date
    new_df = []

    # Ensure the DataFrame is sorted by ticker and created_at
    df = df.sort_values(['ticker', time_col]).reset_index(drop=True)

    # Group by ticker
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

            if mode == 'TFIDF' and top_n is not None:
                vectorizer = TfidfVectorizer()
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(text_series)
                    
                    if tfidf_matrix.shape[1] == 0:  # Check if TF-IDF matrix is empty
                        raise ValueError("Empty TF-IDF matrix.")
                    
                    # Sum the TF-IDF scores for each document and flatten it to 1D
                    tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
                    
                    # Get the indices of the top_n texts
                    top_indices = np.argsort(tfidf_scores)[-top_n:]

                    # Select the top_n texts, ids, and dates
                    text_series = [text_series[i] for i in top_indices]
                    ids = [ids[i] for i in top_indices]
                    text_dates = [text_dates[i] for i in top_indices]
                
                except ValueError as e:
                    # Handle cases where the TF-IDF matrix is empty
                    print(f"Skipping window for ticker {ticker} on {target_date}: {e}")
                    continue  # Skip this window if TF-IDF processing fails

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

    #we now have the windowed text and ts (including multiple days); not yet implemented for multiple tickers
    return text_df, ts_df
    
def naive_negative_creation():
    pass

def create_pairs(text_df, ts_df, negatives_creation:str, negative_label:int=-1):
    #Create positive pairs where ticker matches and target_date of ts_df == end_date of text_df 
    merged_df = pd.merge(
        text_df,
        ts_df,
        left_on=['ticker', 'end_date'],
        right_on=['ticker', 'target_date'],
        suffixes=('_text_df', '_ts_df')
    )
    merged_df["label"] = 1 #all so far are positive pairs

    #create negative pairs
    #naive method where ticker isn't the same and date isn't the same
    if negatives_creation[0] == 'naive':
        days_away = negatives_creation[1]
        ts_df['target_date'] = pd.to_datetime(ts_df['target_date'], utc=True).dt.date
        merged_df['target_date_text_df'] = pd.to_datetime(merged_df['target_date_text_df'], utc=True).dt.date

        #create pairs where tickers aren't the same and x days away
        negative_pairs = []
        for idx, row in merged_df.iterrows():
            #loop over all rows in merged_df
            ticker = row['ticker']
            start_date = row['target_date_text_df']
            

            ## Get potential negative samples that are at least k days apart            
            potential_negatives = ts_df[
                (ts_df['ticker'] != ticker) & 
                ((ts_df["target_date"] < start_date - pd.Timedelta(days=days_away)) |
                (ts_df["target_date"] > start_date + pd.Timedelta(days=days_away)))
            ]

            if not potential_negatives.empty:
                # Randomly select a negative sample
                negative_sample = potential_negatives.sample(n=1).iloc[0]
                negative_pairs.append({
                    'ids': row['ids'],
                    'ticker': row['ticker'],  # Keeping the same ticker for consistency
                    'target_date_text_df': row['target_date_text_df'],  # Keeping the same start date for consistency
                    'target_date_ts_df': negative_sample['target_date'],  # Keeping the same start date for consistency
                    'end_date': row["end_date"],
                    'text_dates': row["text_dates"],
                    'text_series': row['text_series'],
                    'time_series': negative_sample['time_series'],
                    'label': negative_label,  # Label for negative pair
                    "ts_past_features": negative_sample["ts_past_features"]
                })
            
    
    #where base mean embeddings of text are significantly far from positive pair?

    #where base mean embeddings of ts are signficiantly far form positive pair

    #simple distance (euclid) from positive


    negative_df = pd.DataFrame(negative_pairs)

    # Append the negative pairs to the original DataFrame
    return pd.concat([merged_df, negative_df], ignore_index=True)

def normalize_ts(df):
    c_col='Close'
    t_col='ticker'
    for ticker in df[t_col].unique():
        df.loc[df[t_col] == ticker, c_col] = (df.loc[df[t_col] == ticker, c_col] - df.loc[df[t_col] == ticker, c_col].min()) / (df.loc[df[t_col] == ticker, c_col].max() - df.loc[df[t_col] == ticker, c_col].min())
    return df

def get_data_loaders(df, model, batch_size, num_workers):
    dataset = CustomDataset(df=df, text_tokenizer=model.get_text_tokenizer())

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader

def correct_negative_labels(df:pd.DataFrame, negative_label:int, label_column:str='label'):
    #helper method use to correct negative pairs (default is -1) as some criterion will expect different negative labels -1 or 0

    df[label_column] = df[label_column].replace(-1, negative_label)
    return df

def get_data(model, 
             data_source:dict, 
             ts_window:int=5, 
             ts_mode:str="middle", 
             text_window:int=1, 
             text_selection_method:tuple=('TFIDF', 5), 
             negatives_creation:tuple=("naive", 31), 
             batch_size:int=16, 
             num_workers:int=6, 
             loaders:bool=True):
    
    text_df, ts_df = wrangle_data(data_source) #returns df with id, ticker, Date, text, close
    text_date_col = data_source['text_date_col']
    ts_date_col = data_source["ts_date_col"]
    text_col = data_source["text_col"]
    #make sure date columns are datetimes
    text_df[text_date_col] = pd.to_datetime(text_df[text_date_col], utc=True).dt.date
    ts_df[ts_date_col] = pd.to_datetime(ts_df[ts_date_col], utc=True).dt.date
    
    #normlaize ts_df
    ts_df = normalize_ts(ts_df)

    #convert df to id, tickers:[list], start_date, texts:list, time_series:list, past_time_features:[list]
    text_df, ts_df = process_windows(text_df=text_df, ts_df=ts_df, ts_window=ts_window, ts_mode=ts_mode, text_window=text_window, text_selection_method=text_selection_method, text_col=text_col, text_time_col=text_date_col, ts_time_col=ts_date_col)
    #padd text_series to have empty strings for collate_fn

    
    df = create_pairs(text_df=text_df, ts_df=ts_df, negatives_creation=negatives_creation)
    max_length = df['text_series'].apply(len).max()
    df['text_series'] = df['text_series'].apply(lambda x: x + [''] * (max_length - len(x)))

    if loaders:
        dataset = CustomDataset(df=df, text_tokenizer=model.get_text_tokenizer())

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size + test_size])
        val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader
    else:
        return df