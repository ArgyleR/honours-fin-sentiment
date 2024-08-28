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
import torch.optim as optim
import pdb
import random
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
    data

def update_data_train_metrics(data, train_loss, train_acc, train_f1, train_conf_matrix,
                                    val_loss, val_acc, val_f1, val_conf_matrix,
                                    test_loss, test_acc, test_f1, test_conf_matrix):
    data["train_metrics"]["loss"] = data["train_metrics"]["loss"].append(train_loss)
    data["train_metrics"]["accuracy"] = data["train_metrics"]["accuracy"].append(train_acc)
    data["train_metrics"]["f1"] = data["train_metrics"]["f1"].append(train_f1)
    data["train_metrics"]["conf_matrix"] = data["train_metrics"]["conf_matrix"].append(train_conf_matrix)
    data["val_metrics"]["loss"] = data["val_metrics"]["loss"].append(val_loss)
    data["val_metrics"]["accuracy"] = data["val_metrics"]["accuracy"].append(val_acc)
    data["val_metrics"]["f1"] = data["val_metrics"]["f1"].append(val_f1)
    data["val_metrics"]["conf_matrix"] = data["val_metrics"]["conf_matrix"].append(val_conf_matrix)
    data["test_metrics"]["loss"] = test_loss
    data["test_metrics"]["accuracy"] = test_acc
    data["test_metrics"]["f1"] = test_f1
    data["test_metrics"]["conf_matrix"] = test_conf_matrix

    return data

def update_data_timing(data, start_loop, end_loop, end_test, loop_time, test_time):
    data['timing']['start_loop'] = start_loop
    data['timing']['end_loop'] = end_loop
    data['timing']['end_test'] = end_test
    data['timing']['loop_time'] = loop_time
    data['timing']['test_time'] = test_time

    return data
            
def grid_search(model_param_grid: dict, dataset_param_grid: dict, out_file: str, checkpoint_dir: str, df=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Datasets take a while to load, load each variation once and perform all model experiments after that
    dataset_permutations = list(itertools.product(*dataset_param_grid.values()))
    dataset_combinations = [dict(zip(dataset_param_grid.keys(), perm)) for perm in dataset_permutations]

    best_dataset_params = None
    best_model_params = None
    best_val_loss = float('inf')

    i = 1000
    for dataset_params in dataset_combinations:        
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
            df = dh3.get_data(model=None, 
                data_source=data_source, 
                ts_window=ts_window, 
                ts_mode=ts_overlap, 
                text_window=text_window, 
                text_selection_method=text_selection_method, 
                negatives_creation=negatives_creation, 
                batch_size=None, 
                num_workers=None, 
                loaders=False)
        
        df_len = len(df)
        
        pair_count = df['label'].value_counts()
        
        model_permutations = list(itertools.product(*model_param_grid.values()))
        model_combinations = [dict(zip(model_param_grid.keys(), perm)) for perm in model_permutations]

        for model_params in model_combinations:
            #====================================================
            #model params
            #====================================================
            ts_encoder                  = model_params["ts_encoder"]
            ts_encoder['ts_window']     = ts_window
            ts_encoder['context_length']= 2
            ts_encoder['prediction_length']=0
            ts_encoder['lags_sequence'] = [1, 2, 3]
            ts_encoder['num_features']  = 3

            text_encoder                = model_params["text_encoder"]
            text_encoder_pretrained     = model_params['text_encoder_pretrained']
            text_aggregation_method     = model_params['text_aggregation_method']
            projection_dim              = model_params["projection_dim"]
            ts_window                   = ts_window
            batch_size                  = model_params["batch_size"]
            num_workers                 = model_params["num_workers"]
            
            #====================================================
            #training params
            #====================================================
            learning_rate               = model_params["learning_rate"]
            optimizer_name              = model_params["optimizer"]
            criterion_name              = model_params["criterion"]
            num_epochs                  = model_params["num_epochs"]
            
            model = mh.get_model(ts_encoder_config=ts_encoder, text_encoder_config=text_encoder, projection_dim=projection_dim, ts_window=ts_window, text_aggregation=text_aggregation_method)
            model.to(device)

            optimizer                   = get_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)
            criterion, negative_label   = get_criterion(criterion_name=criterion_name)

            df = dh3.correct_negative_labels(df, negative_label=negative_label)
            
            train_loader, valid_loader, test_loader = dh3.get_data_loaders(df=df, model=model, batch_size=batch_size, num_workers=num_workers)

            data = get_data_base(search_index=i, epochs=num_epochs, dataset_params=dataset_params, model_params=model_params, df_len=df_len, pair_count=pair_count)

            test_loss, test_accuracy, test_f1, test_conf_matrix = None, None, None, None
            start_loop = datetime.datetime.now()
            for epoch in range(num_epochs):
                train_loss, train_accuracy, train_f1, train_conf_matrix = mh.train(model=model, train_loader=train_loader, optimizer=optimizer, device=device, criterion=criterion)
                val_loss, val_accuracy, val_f1, val_conf_matrix = mh.validate(model=model, val_loader=valid_loader, optimizer=optimizer, device=device, criterion=criterion)
            
                data = update_data_train_metrics(data, train_loss, train_accuracy, train_f1, train_conf_matrix,
                                    val_loss, val_accuracy, val_f1, val_conf_matrix,
                                    test_loss, test_accuracy, test_f1, test_conf_matrix)
            
            end_loop = datetime.datetime.now()
            
            test_loss, test_accuracy, test_f1, test_conf_matrix = mh.validate(model=model, val_loader=test_loader, optimizer=optimizer, device=device, criterion=criterion)      
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
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_search_id_{i}.pth")
                torch.save(model.state_dict(), checkpoint_path)
        

            i += 1
            print(f"Just finihsed search: {i}")

    print(f"Best Model Params: \n{best_model_params}")
    print(f"Best Dataset Params: \n{best_dataset_params}")

def run(df=None):
    #IDEAL PARAM GRID:
    model_param_grid = {
            "ts_encoder": [{"name": 'TimeSeriesTransformerModel'}, {"name": 'AutoFormer'}, {"name": "InformerModel"}],        #MODELhelper
            "text_encoder": [{"name": 'bert-base-uncased'}, {"name": 'bert-base'}],                                         #MODELhelper
            "text_encoder_pretrained": [True, False],                                                                       #MODELhelper
            "text_aggregation_method": ["mean", "max", "string_append"],                                                    #MODELhelper
            "projection_dim": [400, 500, 600, 700],                                                                         #MODELhelper
            "learning_rate": [0.00001, 0.0001],                                                                             #GRIDSEARCH     #DONE
            "optimizer": ['adam'],                                                                                          #GRIDSEARCH     #DONE
            "criterion": ['CosineEmbeddingLoss'],                                                                           #GRIDSEARCH     #DONEISH                                                   
            "num_epochs": [10],                                                                                             #GRIDSEARCH     #DONE
            "batch_size": [16],                                                                                             #DATAhelper     #DONE
            "num_workers": [6],  
        }

    dataset_param_grid = {
        "text_selection_method": ["TFIDF"],                                                                             #DATAhelper
        "ts_window": [5],                                                                         #DATAhelper
        "ts_overlap": ['start', 'middle', 'end'],                                                                    #DATAhelper
        "text_window": [3],                                                                 #DATAhelper
        'text_selection_method': [('TFIDF', 5)],
        "data_source": [
        {
            "name": "stock_emotion",
            "text_path": "./data/stock_emotions/tweet/processed_stockemo.csv",
            "ts_path": "./data/stock_emotions/price/",
            "ts_date_col": 'Date',
            'text_date_col': 'date'
        }],                                                            #DATAhelper
        "negatives_creation": [("naive", 31), ("naive", 60), ("diff_distribution", )],                          #DATAhelper
        "random_state": [42, 43, 44],
    }

    grid_search(model_param_grid=model_param_grid, dataset_param_grid=dataset_param_grid, out_file='output_temp.json', checkpoint_dir='checkpoint_temp/', df=df)

run()