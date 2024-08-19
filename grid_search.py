import model_helper as mh
import data_helper_v2 as dh
import torch
import itertools
from tqdm.notebook import tqdm
import datetime
import os
import torch.nn as nn
import json
import torch.optim as optim


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

def grid_search(param_grid: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permutations = list(itertools.product(*param_grid.values()))
    combinations = [dict(zip(param_grid.keys(), perm)) for perm in permutations]

    best_params = None
    best_val_loss = float('inf')

    i = 0


    for params in tqdm(combinations, leave=True, position=1, desc="Grid Search"):
        #====================================================
        #general params
        #====================================================
        out_file                    = params["out_file"]
        checkpoint_dir              = params["checkpoint_dir"]
        #====================================================
        #model params
        #====================================================
        ts_encoder                  = params["ts_encoder"]
        text_encoder                = params["text_encoder"]
        projection_dim              = params["projection_dim"]
        ts_window                   = params["ts_window"]

        #====================================================
        #dataset params
        #====================================================
        text_window                 = params["text_window"]
        data_source                 = params["data_source"]
        days_away                   = params["days_away"]
        batch_size                  = params["batch_size"]
        num_workers                 = params["num_workers"]

        #====================================================
        #training params
        #====================================================
        learning_rate               = params["learning_rate"]
        optimizer_name              = params["optimizer"]
        criterion_name              = params["criterion"]
        random_state                = params["random_state"]
        num_epochs                  = params["num_epochs"]

        model = mh.get_model(ts_encoder_config=ts_encoder, text_encoder_config=text_encoder, projection_dim=projection_dim, ts_window=ts_window)
        model.to(device)

        optimizer                   = get_optimizer(optimizer_name=optimizer_name, model=model, lr=learning_rate)
        criterion, negative_label   = get_criterion(criterion_name=criterion_name)

        train_loader, valid_loader, test_loader = dh.get_data(data_source=data_source, model=model, text_window=text_window, ts_window=ts_window, days_away=days_away, negative_label=negative_label, batch_size=batch_size, num_workers=num_workers, device=device)
        
        data = {
                "end_time": [],
                "search_index": i,
                "epoch": [],
                "model_params": params,
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
                    "loss": [],
                    "accuracy": [],
                    "f1": [],
                    "conf_matrix": []
                },
                "timing": {
                    "start_epoch": [],
                    "end_train": [],
                    "end_validate": [],
                    "end_epoch_and_test": [],
                    "train_time": [],
                    "validate_time": [],
                    "test_time": [],
                    "epoch_time": []
                }
            }

        test_loss, test_accuracy, test_f1, test_conf_matrix = None, None, None, None
        for epoch in range(num_epochs):
            start_epoch = datetime.datetime.now()
            train_loss, train_accuracy, train_f1, train_conf_matrix = mh.train(model=model, train_loader=train_loader, optimizer=optimizer, device=device, criterion=criterion)
            end_train = datetime.datetime.now()
            val_loss, val_accuracy, val_f1, val_conf_matrix = mh.validate(model=model, val_loader=valid_loader, optimizer=optimizer, device=device, criterion=criterion)
            end_validate = datetime.datetime.now()
            end_epoch = datetime.datetime.now()

            if epoch == num_epochs:
                test_loss, test_accuracy, test_f1, test_conf_matrix = mh.validate(model=model, val_loader=test_loader, optimizer=optimizer, device=device, criterion=criterion)


    
            data["end_time"].append(end_epoch.isoformat())
            data["epoch"].append(epoch)
            data["train_metrics"]["loss"].append(train_loss)
            data["train_metrics"]["accuracy"].append(train_accuracy)
            data["train_metrics"]["f1"].append(train_f1)
            data["val_metrics"]["loss"].append(val_loss)
            data["val_metrics"]["accuracy"].append(val_accuracy)
            data["val_metrics"]["f1"].append(val_f1)
            data["test_metrics"]["loss"].append(test_loss)
            data["test_metrics"]["accuracy"].append(test_accuracy)
            data["test_metrics"]["f1"].append(test_f1)
            data["timing"]["start_epoch"].append(start_epoch.isoformat())
            data["timing"]["end_train"].append(end_train.isoformat())
            data["timing"]["end_validate"].append(end_validate.isoformat())
            data["timing"]["end_epoch_and_test"].append(end_epoch.isoformat())
            data["timing"]["train_time"].append((end_train - start_epoch).total_seconds())
            data["timing"]["validate_time"].append((end_validate - end_train).total_seconds())
            data["timing"]["test_time"].append((end_epoch - end_validate).total_seconds())
            data["timing"]["epoch_time"].append((end_epoch - start_epoch).total_seconds())

            try:
                data["train_metrics"]["conf_matrix"].append(train_conf_matrix.tolist())
            except AttributeError:
                continue
            try:
                data["val_metrics"]["conf_matrix"].append(val_conf_matrix.tolist())
            except AttributeError:
                continue
            try:
                data["test_metrics"]["conf_matrix"].append(test_conf_matrix.tolist())
            except AttributeError:
                continue

            if val_loss < best_val_loss:
                print(data)
                best_val_loss = val_loss
                best_params = params
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_search_index_{i}_time_{end_epoch}_epoch_{epoch}.pth")
                torch.save(model.state_dict(), checkpoint_path)

        i += 1
        # Write to JSON file
        with open(out_file, 'a') as file:
            json.dump(data, file)
            file.write('\n')


def run():
    param_grid = param_grid = {
        "out_file": ['output.json'],
        "checkpoint_dir": ["checkpoint/"],
        "ts_encoder": [{"name": 'TSTransformerBaseEncoder'}],
        "text_encoder": [{"name": 'bert-base-uncased', "auto-pre-trained": True}],
        "projection_dim": [400, 500, 600, 700],
        "ts_window": [5],
        "text_window": [1],
        "data_source": ['stock_emotion'],
        "days_away": [31, 60],
        "batch_size": [16],
        "num_workers": [6],
        "learning_rate": [0.00001, 0.0001],
        "optimizer": ['adam'],
        "criterion": ['CosineEmbeddingLoss'],
        "random_state": [42, 43, 44],
        "num_epochs": [15]
    }


    grid_search(param_grid=param_grid)

run()