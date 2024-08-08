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

        train_loader, valid_loader, test_loader = dh.get_data(data_source=data_source, model=model, text_window=text_window, ts_window=ts_window, days_away=days_away, negative_label=negative_label, batch_size=batch_size, num_workers=num_workers)

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


            data = {
                "end_time": end_epoch.isoformat(),
                "search_index": i,
                "epoch": epoch,
                "model_params": params,
                "train_metrics": {
                    "loss": train_loss,
                    "accuracy": train_accuracy,
                    "f1": train_f1,
                    "conf_matrix": train_conf_matrix.tolist()
                },
                "val_metrics": {
                    "loss": val_loss,
                    "accuracy": val_accuracy,
                    "f1": val_f1,
                    "conf_matrix": val_conf_matrix.tolist()
                },
                "test_metrics": {
                    "loss": test_loss,
                    "accuracy": test_accuracy,
                    "f1": test_f1,
                    "conf_matrix": test_conf_matrix.tolist() if test_conf_matrix is not None else None
                },
                "timing": {
                    "start_epoch": start_epoch.isoformat(),
                    "end_train": end_train.isoformat(),
                    "end_validate": end_validate.isoformat(),
                    "end_epoch_and_test": end_epoch.isoformat(),
                    "train_time": (end_train - start_epoch).total_seconds(),
                    "validate_time": (end_validate - end_train).total_seconds(),
                    "test_time": (end_epoch - end_validate).total_seconds(),
                    "epoch_time": (end_epoch - start_epoch).total_seconds()
                }
            }

            # Write to JSON file
            with open(out_file, 'a') as file:
                json.dump(data, file)
                file.write('\n')

            if val_loss < best_val_loss:
                print(data)
                best_val_loss = val_loss
                best_params = params
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_search_index_{i}_time_{end_epoch}_epoch_{epoch}.pth")
                torch.save(model.state_dict(), checkpoint_path)


        i += 1


def run():
    param_grid = param_grid = {
        "out_file": ['output.json'],
        "checkpoint_dir": ["checkpoint/"],
        "ts_encoder": [{"name": 'LSTM'}],
        "text_encoder": [{"name": 'bert-base-uncased', "auto-pre-trained": True}],
        "projection_dim": [64, 32, 16, 8, 5],
        "ts_window": [5, 7, 10],
        "text_window": [1, 5],
        "data_source": ['stock_emotion'],
        "days_away": [31, 60],
        "batch_size": [16],
        "num_workers": [6],
        "learning_rate": [0.001, 0.01, 0.1, 0.0001],
        "optimizer": ['adam'],
        "criterion": ['CosineEmbeddingLoss'],
        "random_state": [42, 43, 44],
        "num_epochs": [15]
    }


    grid_search(param_grid=param_grid)

run()