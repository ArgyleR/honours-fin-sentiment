import model_helper as mh
import data_helper_v3 as dh3
import torch
import itertools
from tqdm import tqdm
import datetime
import os
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch.optim as optim
import pdb
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':        
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(optimizer_name: str, model, lr):
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer

def get_criterion(criterion_name:str):#TODO
    return torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean'), -1

def get_data_base(search_index, epochs, dataset_params, model_params, df_len, pair_count):
    #doc is shorted/longer?
    data = {
        "end_time": None,
        "search_index": search_index,
        "epochs": epochs,
        "dataset_params": dataset_params,
        "model_params": model_params,
        "df_len": df_len,
        "pair_count": pair_count,
        "train_metrics": {
            "loss": [],
            "accuracy": [],
            "f1": [],
            "conf_matrix": []
        },
        "val_metrics": {
            "loss": [],
            "accuracy": [],
            "f1": [],
            "conf_matrix": []
        },
        "test_metrics": {
            "loss": None,
            "accuracy": None,
            "f1": None,
            "conf_matrix": None
        },
        "timing": {
            "start_loop": None,
            "end_loop": None,
            "end_test": None,
            "loop_time": None,
            "test_time": None
        }
    }
    return data

def update_data_train_metrics(data, train_loss, train_acc, train_f1, train_conf_matrix,
                                    val_loss, val_acc, val_f1, val_conf_matrix):
    try:
        train_conf_matrix = train_conf_matrix.tolist()
    except Exception:
        pass
    try:
        val_conf_matrix = val_conf_matrix.tolist()
    except Exception:
        pass
    
    data["train_metrics"]["loss"].append(train_loss)
    data["train_metrics"]["accuracy"].append(train_acc)
    data["train_metrics"]["f1"].append(train_f1)
    data["train_metrics"]["conf_matrix"].append(train_conf_matrix)
    data["val_metrics"]["loss"].append(val_loss)
    data["val_metrics"]["accuracy"].append(val_acc)
    data["val_metrics"]["f1"].append(val_f1)
    data["val_metrics"]["conf_matrix"].append(val_conf_matrix)
    
    return data

def update_data_test_metrics(data, test_loss, test_acc, test_f1, test_conf_matrix):
    try:
        test_conf_matrix = test_conf_matrix.tolist()
    except Exception:
        pass
    data["test_metrics"]["loss"] = test_loss
    data["test_metrics"]["accuracy"] = test_acc
    data["test_metrics"]["f1"] = test_f1
    data["test_metrics"]["conf_matrix"] = test_conf_matrix

    return data


def update_data_timing(data, start_loop, end_loop, end_test, loop_time, test_time):
    data['timing']['start_loop'] = start_loop.isoformat()
    data['timing']['end_loop'] = end_loop.isoformat()
    data['timing']['end_test'] = end_test.isoformat()
    data['timing']['loop_time'] = loop_time
    data['timing']['test_time'] = test_time

    return data

def compare_dicts(dict1, dict2):
    for key in dict1.keys():
        if key == 'num_workers': #num workers doesn't matter
            continue
        else:
            if key != 'text_selection_method' and key != 'negatives_creation':
                #These have particular serialise/deseerialise issues where it's originally a tuple and saved as a list 
                if dict1[key] != dict2[key]:
                    return False
            else:
                if dict1[key][0] != dict2[key][0] or dict1[key][1] != dict2[key][1]:
                    return False
    return True
    
def check_args_not_used(data_parameters, model_parameters, output_file):
    with open(output_file, 'r') as file:
        data = json.load(file)
    for i in data:
        seen_dataset_params = i['dataset_params']
        seen_model_params = i['model_params']
        if compare_dicts(seen_dataset_params, data_parameters) and compare_dicts(seen_model_params, model_parameters):
            return False
    return True

def save_df_list(df_list, save_name):
    labels = ["train", "val", "test"]
    for df, label in zip(df_list, labels):
        df.to_csv(f"./data/constructed_datasets/overlap/{save_name}_{label}.csv", index = False)


def make_non_primitive_a_safe_string(non_primitive):
    non_primitive = str(non_primitive)
    return re.sub(r'[^a-zA-Z0-9]', '', non_primitive)

class LSTMClassifier(nn.Module):
    def __init__(self, text_feature_dim, lstm_input_dim, lstm_hidden_dim, dense_units, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(text_feature_dim + lstm_hidden_dim, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
    
    def forward(self, text_features, time_series_features):
        lstm_out, _ = self.lstm(time_series_features)
        lstm_out = lstm_out[:, -1, :]  # Get the output of the last LSTM cell
        combined_features = torch.cat((text_features, lstm_out), dim=1)
        x = self.dropout(torch.relu(self.fc1(combined_features)))
        output = torch.sigmoid(self.fc2(x))
        return output

def run_lstm(train_df, validation_df, test_df, max_features, lstm_input_dim, lstm_hidden_dim=64, dropout_rate=0.5, dense_units=64, epochs=10, batch_size=32):
    # Flatten text data into single strings
    train_texts = train_df['text_series'].apply(lambda x: ' '.join(x))
    valid_texts = validation_df['text_series'].apply(lambda x: ' '.join(x))
    test_texts = test_df['text_series'].apply(lambda x: ' '.join(x))

    # Apply TF-IDF vectorization to the concatenated text
    vectorizer = TfidfVectorizer(max_features=max_features)
    train_text_features = vectorizer.fit_transform(train_texts).toarray()
    validation_text_features = vectorizer.transform(valid_texts).toarray()
    test_text_features = vectorizer.transform(test_texts).toarray()

    # Standardize time series features
    scaler = StandardScaler()
    train_time_series_features = np.array(list(train_df['time_series']))
    validation_time_series_features = np.array(list(validation_df['time_series']))
    test_time_series_features = np.array(list(test_df['time_series']))

    train_time_series_features = scaler.fit_transform(train_time_series_features)
    validation_time_series_features = scaler.transform(validation_time_series_features)
    test_time_series_features = scaler.transform(test_time_series_features)

    # Convert data to PyTorch tensors
    train_text_features = torch.tensor(train_text_features, dtype=torch.float32)
    validation_text_features = torch.tensor(validation_text_features, dtype=torch.float32)
    test_text_features = torch.tensor(test_text_features, dtype=torch.float32)

    train_time_series_features = torch.tensor(train_time_series_features, dtype=torch.float32).unsqueeze(2)
    validation_time_series_features = torch.tensor(validation_time_series_features, dtype=torch.float32).unsqueeze(2)
    test_time_series_features = torch.tensor(test_time_series_features, dtype=torch.float32).unsqueeze(2)

    train_labels = torch.tensor(train_df['label'].values, dtype=torch.float32).unsqueeze(1)
    validation_labels = torch.tensor(validation_df['label'].values, dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test_df['label'].values, dtype=torch.float32).unsqueeze(1)

    # Initialize the model
    model = LSTMClassifier(
        text_feature_dim=train_text_features.shape[1],
        lstm_input_dim=train_time_series_features.shape[2],
        lstm_hidden_dim=lstm_hidden_dim,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_text_features, train_time_series_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Validation and testing
    model.eval()
    with torch.no_grad():
        # Validation
        valid_outputs = model(validation_text_features, validation_time_series_features)
        valid_predictions = (valid_outputs > 0.5).float()
        valid_accuracy = accuracy_score(validation_labels.numpy(), valid_predictions.numpy())
        valid_f1 = f1_score(validation_labels.numpy(), valid_predictions.numpy(), average='weighted')
        
        # Test
        test_outputs = model(test_text_features, test_time_series_features)
        test_predictions = (test_outputs > 0.5).float()
        test_accuracy = accuracy_score(test_labels.numpy(), test_predictions.numpy())
        test_f1 = f1_score(test_labels.numpy(), test_predictions.numpy(), average='weighted')

    # Print classification reports
    print("Validation Classification Report:")
    print(classification_report(validation_labels.numpy(), valid_predictions.numpy()))
    print("Test Classification Report:")
    print(classification_report(test_labels.numpy(), test_predictions.numpy()))

    # Store results in a JSON format
    results = {
        "validation": {
            "accuracy": valid_accuracy,
            "f1_score": valid_f1
        },
        "test": {
            "accuracy": test_accuracy,
            "f1_score": test_f1
        }
    }
    
    return results

def run_svm(train_df, validation_df, test_df, random_state, kernel, max_features, C):
    #flatten text data into single strings
    train_texts = train_df['text_series'].apply(lambda x: ' '.join(x))
    valid_texts = validation_df['text_series'].apply(lambda x: ' '.join(x))
    test_texts = test_df['text_series'].apply(lambda x: ' '.join(x))

    # Step 2: Apply TF-IDF vectorization to the concatenated text
    vectorizer = TfidfVectorizer(max_features=max_features)
    train_text_features = vectorizer.fit_transform(train_texts).toarray()
    validation_text_features = vectorizer.transform(valid_texts).toarray()
    test_text_features = vectorizer.transform(test_texts).toarray()

    # Step 3: Extract time series features and labels
    train_time_series_features = np.array(list(train_df['time_series']))
    validation_time_series_features = np.array(list(validation_df['time_series']))
    test_time_series_features = np.array(list(test_df['time_series']))

    train_labels = train_df['label'].values
    valid_labels = validation_df['label'].values
    test_labels = test_df['label'].values

    # Step 4: Combine text and time series features
    train_features = np.hstack([train_text_features, train_time_series_features])
    valid_features = np.hstack([validation_text_features, validation_time_series_features])
    test_features = np.hstack([test_text_features, test_time_series_features])

    # Step 5: Standardize the combined features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    valid_features = scaler.transform(valid_features)
    test_features = scaler.transform(test_features)

    # Step 6: Train the SVM classifier
    svm_model = SVC(kernel=kernel, C=C, random_state=random_state)
    svm_model.fit(train_features, train_labels)

    # Step 7: Make predictions and evaluate
    valid_y_pred = svm_model.predict(valid_features)
    print("Classification Report:")
    print(classification_report(valid_labels, valid_y_pred))

    # Step 8: Calculate evaluation metrics
    valid_accuracy = accuracy_score(valid_labels, valid_y_pred)
    valid_f1 = f1_score(valid_labels, valid_y_pred, average='weighted')  # 'weighted' for handling class imbalance

    # Step 7: Make predictions and evaluate
    test_y_pred = svm_model.predict(test_features)
    print("Classification Report:")
    print(classification_report(test_labels, test_y_pred))

    # Step 8: Calculate evaluation metrics
    test_accuracy = accuracy_score(test_labels, test_y_pred)
    test_f1 = f1_score(test_labels, test_y_pred, average='weighted')  # 'weighted' for handling class imbalance

    # Step 9: Store results in a JSON format
    results = {
        "validation": {
            "accuracy": valid_accuracy,
            "f1_score": valid_f1
        },
        "test": {
            "accuracy": test_accuracy,
            "fq_score": test_f1
        }
    }

    return results

def grid_search(model_param_grid: dict, dataset_param_grid: dict, out_file: str, checkpoint_dir: str, df=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Datasets take a while to load, load each variation once and perform all model experiments after that
    dataset_permutations = list(itertools.product(*dataset_param_grid.values()))
    dataset_combinations = [dict(zip(dataset_param_grid.keys(), perm)) for perm in dataset_permutations]

    best_dataset_params = None
    best_model_params = None
    best_val_loss = float('inf')

    i = 1000
    for dataset_params in tqdm(dataset_combinations, desc='Dataset Params', position=0):  
        print(dataset_params)      
        #====================================================
        #dataset params
        #====================================================
        ts_window                   = dataset_params['ts_window']
        ts_overlap                  = dataset_params['ts_overlap']
        text_window                 = dataset_params["text_window"]
        text_selection_method       = dataset_params['text_selection_method']
        data_source                 = dataset_params["data_source"]
        negatives_creation          = dataset_params['negatives_creation']
        random_state                = dataset_params["random_state"] #effects everything but first call is the ds creation

        set_seed(random_state, device=device)

        if df is None:
            #get the df before anything else: 
            df_list = dh3.get_data(text_tokenizer=None, 
                data_source=data_source, 
                ts_window=ts_window, 
                ts_mode=ts_overlap, 
                text_window=text_window, 
                text_selection_method=text_selection_method, 
                negatives_creation=negatives_creation, 
                batch_size=None, 
                num_workers=None, 
                loaders=False,
                subset_data=True, 
                random_state=random_state,
                long_run=False)
            
            #save_name = f"name_{data_source['name']}_tswin_{ts_window}_textwin_{text_window}_overlap_{ts_overlap}_textselection_{make_non_primitive_a_safe_string(text_selection_method)}_negativecreation_{make_non_primitive_a_safe_string(negatives_creation)}_randomstate_{random_state}"
            #save_df_list(df_list=df_list, save_name=save_name)
            

        df_len = len(df_list[0])#length of train df

        pair_count = list(df_list[0]['label'].value_counts().items())
        
        model_permutations = list(itertools.product(*model_param_grid.values()))
        model_combinations = [dict(zip(model_param_grid.keys(), perm)) for perm in model_permutations]
        
        df_list = [dh3.correct_negative_labels(single_df, negative_label=0) for single_df in df_list]

        for model_params in tqdm(model_combinations, desc='Model Params', leave=True, position=1):
            
            print(model_params)
            #====================================================
            #model params
            #====================================================
            if model_params['model'] == 'svm':
                kernel = model_params['kernel']
                max_features = model_params['max_features']
                C = model_params['C']
                results = run_svm(train_df=df_list[0], validation_df=df_list[1], test_df=df_list[2], random_state=random_state, kernel=kernel, max_features=max_features, C=C)

                
            
            elif model_params['model'] == 'lstm':
                lstm_hidden_dim = model_params['lstm_hidden_dim']
                lstm_input_dim = model_params['lstm_input_dim']
                dropout_rate = model_params['dropout_rate']
                dense_units = model_params['dense_units']
                epochs = model_params['epochs']
                batch_size = model_params['batch_size']
                learning_rate = model_params['learning_rate']
                max_features = model_params['max_features']

                # Run the LSTM model
                results = run_lstm(
                    train_df=df_list[0],
                    validation_df=df_list[1],
                    test_df=df_list[2],
                    max_features=max_features,
                    lstm_input_dim=lstm_input_dim,
                    lstm_hidden_dim=lstm_hidden_dim,
                    dropout_rate=dropout_rate,
                    dense_units=dense_units,
                    epochs=epochs,
                    batch_size=batch_size
                )
            dump_to_file = {
                    "datset_params": dataset_params,
                    "model_params": model_params,
                    "results": results
                }
            # Save results to a JSON file
            with open(out_file, "a") as json_file:
                json.dump(dump_to_file, json_file, indent=4)
                json_file.write(",\n")

    print(f"Best Model Params: \n{best_model_params}")
    print(f"Best Dataset Params: \n{best_dataset_params}")

def run(df=None):
    #IDEAL PARAM GRID:
    model_param_grid = {
            'model': ["svm"],
            'kernel':['linear', 'rbf'], 
            'max_features': [10, 100, 500], 
            'C': [0.01, 0.1, 1, 10, 100]
        }

    dataset_param_grid = {                                                                            
        "ts_window": [4],                                                                  
        "ts_overlap": ['start'],                                                                
        "text_window": [3],                                                 
        'text_selection_method': [('vader_polarized', 5)],#('TFIDF', 5)],# ('vader_polarized', 5), ('vader_neutral', 5), ('TFIDF', 2), ('embedding_diversity', 5), ('embedding_diversity', 2), ('vader_neural', 2), ('vader_polarized', 2)],
        "data_source": [{
            "name": "EDT",
            "text_path": "./data/EDT/evaluate_news.json",
            "ts_path": "./data/stock_emotions/price/",
            "ts_date_col": 'Date',
            'text_date_col': 'date',
            'text_col': 'text',
            'train_dates': '01/01/2020 - 03/09/2020',
            'test_dates': '04/09/2020 - 31/12/2020'
        }
        ],                                                           
        "negatives_creation": [("sentence_transformer_dissimilarity", "max")],# ("sentence_transformer_dissimilarity", "mean"), ("sentence_transformer_dissimilarity", "min"), ("naive", 30), ("naive", 45), ("naive", 60)],                      
        "random_state": [42, 43, 44],
    }
    return grid_search(model_param_grid=model_param_grid, dataset_param_grid=dataset_param_grid, out_file='./results/baselines/svm.json', checkpoint_dir='checkpoint_final/', df=df)

def run_stock_emotion_best(df=None):
    model_param_grid = {
            'model': ["svm"],
            'kernel':['linear', 'rbf'], 
            'max_features': [10, 100, 500], 
            'C': [0.01, 0.1, 1, 10, 100]
        }


    dataset_param_grid = {                                                                            
        "ts_window": [4],#4, 6 & 7 had a random error out     3, 4, 5, 6, 7, 10                                                                    
        "ts_overlap": ['start'], #'middle'                                                                   
        "text_window": [3],          #3, 4, 5, 6, 7                                              
        'text_selection_method': [('vader_polarized', 5)],#('TFIDF', 5)],# ('vader_polarized', 5), ('vader_neutral', 5), ('TFIDF', 2), ('embedding_diversity', 5), ('embedding_diversity', 2), ('vader_neural', 2), ('vader_polarized', 2)],
        "data_source": [
        {
            "name": "stock_emotion",
            "text_path": "./data/stock_emotions/tweet/processed_stockemo.csv",
            "ts_path": "./data/stock_emotions/price/",
            "ts_date_col": 'Date',
            'text_date_col': 'date',
            'text_col': 'text',
            'train_dates': '01/01/2020 - 03/09/2020',
            'test_dates': '04/09/2020 - 31/12/2020'
        }
        ],                                                        
        "negatives_creation": [("sentence_transformer_dissimilarity", "max"), ("naive", 30)],# ("sentence_transformer_dissimilarity", "mean"), ("sentence_transformer_dissimilarity", "min"), ("naive", 30), ("naive", 45), ("naive", 60)],                      
        "random_state": [42, 43, 44],
    }
    return grid_search(model_param_grid=model_param_grid, dataset_param_grid=dataset_param_grid, out_file='./results/baselines/svm.json', checkpoint_dir='checkpoint_final/', df=df)

def run_stock_net_best(df=None):
    model_param_grid = {
            'model': ["svm"],
            'kernel':['linear', 'rbf'], 
            'max_features': [10, 100, 500], 
            'C': [0.01, 0.1, 1, 10, 100]
        }
    
    
    dataset_param_grid = {                                                                            
        "ts_window": [4],#4, 6 & 7 had a random error out     3, 4, 5, 6, 7, 10                                                                    
        "ts_overlap": ['start'], #'middle'                                                                   
        "text_window": [3],          #3, 4, 5, 6, 7                                              
        'text_selection_method': [('vader_neutral', 5)],#('TFIDF', 5)],# ('vader_polarized', 5), ('vader_neutral', 5), ('TFIDF', 2), ('embedding_diversity', 5), ('embedding_diversity', 2), ('vader_neutral', 2), ('vader_polarized', 2)],
        "data_source": [{
            "name": "stock_net",
            "text_path": "./data/stocknet/tweet/organised_tweet.csv",
            "ts_path": "./data/stocknet/price/raw/",
            "ts_date_col": 'Date',
            'text_date_col': 'created_at',
            'text_col': 'text',
            'train_dates': '01/01/2014 - 01/08/2015',
            'test_dates': '01/08/2015 - 01/01/2016'
        }
        ],                                                      
        "negatives_creation": [("sentence_transformer_dissimilarity", "max"), ("naive", 30), ("sentence_transformer_dissimilarity", "min")],# ("sentence_transformer_dissimilarity", "mean"), ("sentence_transformer_dissimilarity", "min"), ("naive", 30), ("naive", 45), ("naive", 60)],                      
        "random_state": [42, 43, 44],
    }
    return grid_search(model_param_grid=model_param_grid, dataset_param_grid=dataset_param_grid, out_file='./results/baselines/svm.json', checkpoint_dir='checkpoint_final/', df=df)

if __name__ == '__main__':
    
    run()
    run_stock_emotion_best()
    run_stock_net_best()