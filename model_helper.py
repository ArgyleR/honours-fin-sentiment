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

import torch
import torch.nn as nn

def cosine_similarity_custom(batch1, batch2):
    # Calculate dot products
    dot_products = np.einsum('ij,ij->i', batch1, batch2)
    # Calculate norms
    norms1 = np.linalg.norm(batch1, axis=1)
    norms2 = np.linalg.norm(batch2, axis=1)
    # Calculate cosine similarity
    similarities = dot_products / (norms1 * norms2)
    return similarities

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_dim)
        output, (hn, cn) = self.lstm(x)
        
        # Return the last hidden state
        return hn[-1]


class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_tokenizer(self):
        return self.tokenizer    

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
class ContrastiveLearningModel(nn.Module):
    def __init__(self, ts_encoder, text_encoder, projection_dim):
        super(ContrastiveLearningModel, self).__init__()
        self.ts_encoder = ts_encoder
        self.text_encoder = text_encoder
        self.ts_projection_head = nn.Sequential(
            nn.Linear(ts_encoder.hidden_dim, projection_dim)
        )
        self.text_projection_head = nn.Sequential(
            nn.Linear(text_encoder.hidden_dim, projection_dim)
        )
    
    def get_ts_encoder(self):
        return self.ts_encoder
    
    def get_text_tokenizer(self):
        return self.text_encoder.get_tokenizer()
    
    def get_text_encoder(self):
        return self.text_encoder

    def forward(self, ts_data, input_ids, attention_mask):
        ts_embeddings = self.ts_encoder(ts_data)
        text_embeddings = self.text_encoder(input_ids, attention_mask)

        projected_ts_embeddings = self.ts_projection_head(ts_embeddings)
        projected_text_embeddings = self.text_projection_head(text_embeddings)

        return projected_ts_embeddings, projected_text_embeddings

    def predict(self, ts_data, input_ids, attention_mask):
        ts_embeddings, text_embeddings = self.forward(ts_data, input_ids, attention_mask)
        ts_embeddings = ts_embeddings.cpu().detach().numpy()
        text_embeddings = text_embeddings.cpu().detach().numpy()
        return cosine_similarity_custom(ts_embeddings, text_embeddings)

def train(model: ContrastiveLearningModel, train_loader: DataLoader, optimizer, device: str, criterion):
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []
    i = 0

    for ts_data, text_data, attention_mask, labels in tqdm(train_loader, leave=True, position=1):
        ts_data = ts_data.to(device)
        text_data = text_data.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        ts_embeddings, text_embeddings = model(ts_data, text_data, attention_mask)

        loss = criterion(ts_embeddings, text_embeddings, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        #get the model to predict based on the data it has just seen
        preds = model.predict(ts_data=ts_data, input_ids=text_data, attention_mask=attention_mask) #TODO check this works as expected and predicts all not just 1 or something
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

def validate(model: ContrastiveLearningModel, val_loader: DataLoader, optimizer, device: str, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ts_data, text_data, attention_mask, labels in tqdm(val_loader, leave=True, position=1):
            ts_data = ts_data.to(device)
            text_data = text_data.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            ts_embeddings, text_embeddings = model(ts_data, text_data, attention_mask)


            loss = criterion(ts_embeddings, text_embeddings, labels)

            val_loss += loss.item()

            preds = model.predict(ts_data=ts_data, input_ids=text_data, attention_mask=attention_mask)#TODO same as train loop, check it works with the batch of embeddings
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
        return LSTMEncoder(input_dim=input_dim, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown time series encoder: {ts_encoder_config}")

def get_text_encoder(text_encoder_config: dict={"name": 'bert-base-uncased',
                                         "auto-pre-trained": True}):
    if text_encoder_config["auto-pre-trained"]:
        text_encoder_model = TextEncoder(text_encoder_config["name"])
        return text_encoder_model
    elif text_encoder_config["name"] == 'nn.WordBag':
        # Implement a simple word bag encoder if needed
        pass
    else:
        raise ValueError(f"Unknown text encoder: {text_encoder_config}")

def get_model(ts_encoder_config: dict, text_encoder_config: dict, projection_dim: int, ts_window: int):
    """
    
    """

    ts_encoder_model = get_ts_encoder(ts_encoder_config, ts_window=ts_window, projection_dim=projection_dim)
    text_encoder_model = get_text_encoder(text_encoder_config)

    return ContrastiveLearningModel(ts_encoder=ts_encoder_model, text_encoder=text_encoder_model, projection_dim=projection_dim)