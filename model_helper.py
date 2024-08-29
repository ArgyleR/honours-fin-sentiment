import pandas as pd
import numpy as np
import transformers
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
import torch
import torch.nn as nn
import models as m

def train(model: m.ContrastiveLearningModel, train_loader: DataLoader, optimizer, device: str, criterion):
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []
    i = 0

    for ts_data, text_data, labels in tqdm(train_loader, leave=True, position=1):
        ts_data = {
            "past_time_values": torch.stack([d['past_time_values'].squeeze(1) for d in ts_data], dim=0).to(device),
            "past_observed_mask": torch.stack([d['past_observed_mask'].squeeze(0) for d in ts_data], dim=0).to(device),
            "past_time_features": torch.stack([d['past_time_features'].squeeze(0) for d in ts_data], dim=0).to(device)
        }
        text_data['input_ids'] = text_data['input_ids'].to(device)
        text_data['attention_mask'] = text_data['attention_mask'].to(device)
        
        labels = labels.to(device)

        optimizer.zero_grad()
        ts_embeddings, text_embeddings = model(ts_data, text_data)

        loss = criterion(ts_embeddings, text_embeddings, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        #get the model to predict based on the data it has just seen
        preds = model.predict(ts_data=ts_data, text_data=text_data)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        i += 1
    
    all_preds = np.array(all_preds) #convert to numpy array
    all_preds = (all_preds >= 0.5).astype(int).tolist()
    all_labels = [0 if x == -1 else x for x in all_labels]
    train_loss /= len(train_loader)
    

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return train_loss, accuracy, f1, conf_matrix
    #return random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), [[1, 2], [2, 3]]

def validate(model: m.ContrastiveLearningModel, val_loader: DataLoader, optimizer, device: str, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ts_data, text_data, labels in tqdm(val_loader, leave=True, position=1):
            ts_data = {
                "past_time_values": torch.stack([d['past_time_values'].squeeze(1) for d in ts_data], dim=0).to(device),
                "past_observed_mask": torch.stack([d['past_observed_mask'].squeeze(0) for d in ts_data], dim=0).to(device),
                "past_time_features": torch.stack([d['past_time_features'].squeeze(0) for d in ts_data], dim=0).to(device)
            }
            text_data['input_ids'] = text_data['input_ids'].to(device)
            text_data['attention_mask'] = text_data['attention_mask'].to(device)
            
            labels = labels.to(device)
            print(ts_data)
            ts_embeddings, text_embeddings = model(ts_data, text_data)


            loss = criterion(ts_embeddings, text_embeddings, labels)

            val_loss += loss.item()

            preds = model.predict(ts_data=ts_data, text_data=text_data)#TODO same as train loop, check it works with the batch of embeddings
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds) #convert to numpy array
    all_preds = (all_preds >= 0.5).astype(int).tolist()
    all_labels = [0 if x == -1 else x for x in all_labels]

    val_loss /= len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return val_loss, accuracy, f1, conf_matrix

    #return random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), [[1, 2], [2, 3]]

def get_ts_encoder(ts_encoder_config: dict={"name": "LSTM"}, ts_window: int=5, projection_dim: int=128):
    if ts_encoder_config["name"] == "LSTM":
        input_dim = 1#ts_window
        num_layers = 1
        output_dim = projection_dim
        return m.LSTMEncoder(input_dim=input_dim, num_layers=num_layers)
    elif ts_encoder_config["name"] in ["TimeSeriesTransformerModel", "AutoFormerModel", "InformerModel"]:
        return m.TSTransformerBaseEncoder(ts_encoder_config)
    else:
        raise ValueError(f"Unknown time series encoder: {ts_encoder_config}")

def get_text_encoder(text_encoder_config: dict={"name": 'bert-base-uncased',
                                         "auto-pre-trained": True}):
    text_encoder_model = m.TextEncoder(text_encoder_config["name"])
    return text_encoder_model

def get_model(ts_encoder_config: dict, text_encoder_config: dict, projection_dim: int, ts_window: int, text_aggregation: str):
    """
    
    """

    ts_encoder_model = get_ts_encoder(ts_encoder_config, ts_window=ts_window, projection_dim=projection_dim)
    text_encoder_model = get_text_encoder(text_encoder_config)

    return m.ContrastiveLearningModel(ts_encoder=ts_encoder_model, text_encoder=text_encoder_model, projection_dim=projection_dim, text_aggregation=text_aggregation)