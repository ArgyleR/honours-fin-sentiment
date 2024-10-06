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
    return True
    with open(output_file, 'r') as file:
        data = json.load(file)
    for i in data:
        seen_dataset_params = i['dataset_params']
        seen_model_params = i['model_params']
        if compare_dicts(seen_dataset_params, data_parameters) and compare_dicts(seen_model_params, model_parameters):
            return False
    return True
       
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
                subset_data=True)
        
        df_len = len(df_list[0])#length of train df
        
        pair_count = list(df_list[0]['label'].value_counts().items())
        
        model_permutations = list(itertools.product(*model_param_grid.values()))
        model_combinations = [dict(zip(model_param_grid.keys(), perm)) for perm in model_permutations]

        for model_params in tqdm(model_combinations, desc='Model Params', leave=True, position=1):
            print(model_params)
            #====================================================
            #model params
            #====================================================
            ts_encoder                  = model_params["ts_encoder"]
            ts_encoder['ts_window']     = ts_window
            
            ts_encoder['context_length'] = 1
            ts_encoder['prediction_length']=0#always 0 as we aren't predicting anything
            ts_encoder['lags_sequence'] = [i + 1 for i in range(ts_window - 1)]
            ts_encoder['num_features']  = 3#always 3 as we pass the whole time feature set year, month, day
            
            text_encoder                = model_params["text_encoder"]
            text_encoder_pretrained     = model_params['text_encoder_pretrained']
            text_aggregation_method     = model_params['text_aggregation_method']
            projection_dim              = model_params["projection_dim"]
            #ts_window already defined in dataset loop
            batch_size                  = model_params["batch_size"]
            num_workers                 = model_params["num_workers"]
            
            #====================================================
            #training params
            #====================================================
            learning_rate               = model_params["learning_rate"]
            optimizer_name              = model_params["optimizer"]
            criterion_name              = model_params["criterion"]
            num_epochs                  = model_params["num_epochs"]

            if check_args_not_used(data_parameters=dataset_params, model_parameters=model_params, output_file='./results/output_frand_normalized_plotting.json'):
                
                model = mh.get_model(ts_encoder_config=ts_encoder, text_encoder_config=text_encoder, projection_dim=projection_dim, ts_window=ts_window, text_aggregation=text_aggregation_method)
                model.to(device)

                
                optimizer                   = get_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)
                criterion, negative_label   = get_criterion(criterion_name=criterion_name)

                df_list = [dh3.correct_negative_labels(single_df, negative_label=negative_label) for single_df in df_list]     
                train_loader, valid_loader, test_loader = dh3.get_data_loaders(dfs=df_list, text_tokenizer=model.get_text_tokenizer(), batch_size=batch_size, num_workers=num_workers)

                data = get_data_base(search_index=i, epochs=num_epochs, dataset_params=dataset_params, model_params=model_params, df_len=df_len, pair_count=pair_count)


                test_loss, test_accuracy, test_f1, test_conf_matrix = None, None, None, None
                start_loop = datetime.datetime.now()

                for epoch in range(num_epochs):
                    train_loss, train_accuracy, train_f1, train_conf_matrix = mh.train(model=model, train_loader=train_loader, optimizer=optimizer, device=device, criterion=criterion, epoch=epoch)
                    
                    val_loss, val_accuracy, val_f1, val_conf_matrix = mh.validate(model=model, val_loader=valid_loader, optimizer=optimizer, device=device, criterion=criterion, epoch=epoch)
                
                    data = update_data_train_metrics(data, train_loss, train_accuracy, train_f1, train_conf_matrix,
                                        val_loss, val_accuracy, val_f1, val_conf_matrix)
                
                end_loop = datetime.datetime.now()
                
                test_loss, test_accuracy, test_f1, test_conf_matrix = mh.validate(model=model, val_loader=test_loader, optimizer=optimizer, device=device, criterion=criterion, epoch=-1)  
                data = update_data_test_metrics(data, test_loss, test_accuracy, test_f1, test_conf_matrix)

                end_test = datetime.datetime.now()
                loop_time = (start_loop - end_loop).total_seconds()
                test_time= (end_loop - end_test).total_seconds()
                data = update_data_timing(data, start_loop, end_loop, end_test, loop_time, test_time)
                
                # Write to JSON file
                with open(out_file, 'a') as file:
                    json.dump(data, file)
                    file.write('\n')

                if val_loss < best_val_loss:
                    print(json.dumps(data, indent=4))
                    best_val_loss = val_loss
                    best_dataset_params = dataset_params
                    best_model_params = model_params
                    #checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_search_id_{i}.pth")
                    #torch.save(model.state_dict(), checkpoint_path)

                torch.cuda.empty_cache()
                gc.collect()
            i += 1
            print(f"Just finihsed search: {i}")
            

    print(f"Best Model Params: \n{best_model_params}")
    print(f"Best Dataset Params: \n{best_dataset_params}")

def run(df=None):
    #IDEAL PARAM GRID:
    model_param_grid = {
            "ts_encoder": [{"name": 'TimeSeriesTransformerModel'}],#{"name": "InformerModel"}, {"name": 'AutoFormerModel'}],
            "text_encoder": [{"name": 'bert-base-uncased'}],#, {"name": 'bert-base-cased'}],
            "text_encoder_pretrained": [True],                                                                       
            "text_aggregation_method": ['mean', "max"],                                                    
            "projection_dim": [500],                                                                        
            "learning_rate": [0.00001],                                                                             
            "optimizer": ['adam'],                                                                                          
            "criterion": ['CosineEmbeddingLoss'],
            "num_epochs": [10],                                                                                             
            "batch_size": [6],                                                                                             
            "num_workers": [4],  
        }

    dataset_param_grid = {                                                                            
        "ts_window": [6],#4, 6 & 7 had a random error out     3, 4, 5, 6, 7, 10                                                                    
        "ts_overlap": ['start'],                                                                    
        "text_window": [3],                                                        
        'text_selection_method': [('TFIDF', 5)],# ('vader_polarized', 5), ('vader_neutral', 5), ('TFIDF', 2), ('embedding_diversity', 5), ('embedding_diversity', 2), ('vader_neural', 2), ('vader_polarized', 2)],
        "data_source": [{
            "name": "EDT",
            "text_path": "./data/EDT/evaluate_news.json",
            "ts_path": "./data/stock_emotions/price/",
            "ts_date_col": 'Date',
            'text_date_col': 'date',
            'text_col': 'text',
            'train_dates': '01/01/2020 - 03/09/2020',
            'test_dates': '04/09/2020 - 31/12/2020'
        },{
            "name": "stock_emotion",
            "text_path": "./data/stock_emotions/tweet/processed_stockemo.csv",
            "ts_path": "./data/stock_emotions/price/",
            "ts_date_col": 'Date',
            'text_date_col': 'date',
            'text_col': 'text',
            'train_dates': '01/01/2020 - 03/09/2020',
            'test_dates': '04/09/2020 - 31/12/2020'
        },  {
            "name": "stock_net",
            "text_path": "./data/stocknet/tweet/organised_tweet.csv",
            "ts_path": "./data/stocknet/price/raw/",
            "ts_date_col": 'Date',
            'text_date_col': 'created_at',
            'text_col': 'text',
            'train_dates': '01/01/2014 - 01/08/2015',
            'test_dates': '01/08/2015 - 01/01/2016'
        }],                                                            
        "negatives_creation": [("sentence_transformer_dissimilarity", "mean")],# ("sentence_transformer_dissimilarity", "max"), ("sentence_transformer_dissimilarity", "min"), ("naive", 30), ("naive", 45), ("naive", 60)],                          
        "random_state": [42, 43, 44],
    }
    grid_search(model_param_grid=model_param_grid, dataset_param_grid=dataset_param_grid, out_file='./results/output_frand_normalization.json', checkpoint_dir='checkpoint_final/', df=df)
if __name__ == '__main__':
    run()