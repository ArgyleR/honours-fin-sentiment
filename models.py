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
from transformers import AutoformerConfig, AutoformerModel
from transformers import InformerConfig, InformerModel
import torch
import torch.nn as nn
import pdb

def cosine_similarity_custom(batch1, batch2):
    # Calculate dot products
    dot_products = np.einsum('ij,ij->i', batch1, batch2)
    # Calculate norms
    norms1 = np.linalg.norm(batch1, axis=1)
    norms2 = np.linalg.norm(batch2, axis=1)
    # Calculate cosine similarity
    similarities = dot_products / (norms1 * norms2)
    return similarities

#============================================================================================================================================
# TIME SERIES ENCODERS
#============================================================================================================================================

class TSTransformerBaseEncoder(nn.Module):
    def __init__(self, config:dict={"name":"TimeSeriesTransformerModel", "ts_window":5, "context_length": 2, "prediction_length": 0, "lags_sequence":[1, 2, 3], "num_features":1}):
        super(TSTransformerBaseEncoder, self).__init__()
        
        assert config["ts_window"] == max(config["lags_sequence"]) + config["context_length"], "k isn't the same size as max(lags_sequence) + context_length"

        if config["name"] == "TimeSeriesTransformerModel":
            config = TimeSeriesTransformerConfig(
                prediction_length=config["prediction_length"],  
                context_length=config["context_length"],
                lags_sequence=config["lags_sequence"],
                feature_size=len(config["lags_sequence"]) + config["num_features"] + 2
            )
            self.model = TimeSeriesTransformerModel(config)
        elif config["name"] == "AutoformerModel":
            config = AutoformerConfig(
                prediction_length=config["prediction_length"],  
                context_length=config["context_length"],
                lags_sequence=config["lags_sequence"],
                feature_size=len(config["lags_sequence"]) + config["num_features"] + 2
            )
            self.model = AutoformerModel(config)
        elif config["name"] == "InformerModel":
            config = InformerConfig(
                prediction_length=config["prediction_length"],  
                context_length=config["context_length"],
                lags_sequence=config["lags_sequence"],
                feature_size=len(config["lags_sequence"]) + config["num_features"] + 2
            )
            self.model = InformerModel(config)
        else:
            raise NotImplementedError

        
        self.config = self.model.config
        self.hidden_dim = self.config.d_model
    
    def forward(self, ts_data):
        past_time_values = ts_data["past_time_values"].squeeze()
        past_observed_mask = ts_data["past_observed_mask"].squeeze()
        past_time_features = ts_data["past_time_features"].squeeze()

        model_output = self.model(past_values=past_time_values, past_observed_mask=past_observed_mask,past_time_features=past_time_features)
        encoder_last_hidden_state = model_output.encoder_last_hidden_state
        #We want to pool to get the mean of the hidden states
        pooled_output = torch.mean(encoder_last_hidden_state, dim=1)
        #return the mean of the final state? Not sure if this or just the final state
        return pooled_output#torch.mean(encoder_last_hidden_state, dim=1)
         
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, input_dim)
        output, (hn, cn) = self.lstm(x)
        
        # Return the last hidden state for mapping to projection_dim
        return hn[-1]

#============================================================================================================================================
# TEXT ENCODERS
#============================================================================================================================================

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
    
#============================================================================================================================================
# CONTRASTIVE LEARNING MODEL
#============================================================================================================================================

class ContrastiveLearningModel(nn.Module):
    def __init__(self, ts_encoder, text_encoder, projection_dim, text_aggregation):
        super(ContrastiveLearningModel, self).__init__()
        self.text_aggregation=text_aggregation
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

    def forward(self, ts_data, text_data):
        ts_embeddings = self.ts_encoder(ts_data)
        print("text embeddings before processing: ======================================================")
        print(len(text_data))
        print(text_data['input_ids'].shape)
        print(text_data)
        text_embeddings = []
        for text_row in text_data:
            text_embeddings.append(self.text_encoder(text_row['input_ids'], text_row['attention_mask']))

        print("text embeddings after tokenizing: =======================================================")
        print(len(text_embeddings))#
        print(text_embeddings.shape)#

        text_embeddings = torch.stack(text_embeddings)  # Convert list to a tensor

        print("text embeddings after stacking: ======================================================")
        print(text_embeddings.shape) 
        
        if self.text_aggregation == 'mean':
            final_text_embeddings = torch.mean(text_embeddings, dim=1)
        elif self.text_aggregation == 'max':
            final_text_embeddings = torch.max(text_embeddings, dim=1)
        else:
            raise NotImplementedError("text embedding aggregation is only max or mean currently")
    
        print("text embeddings after aggregation: ======================================================") 
        print(final_text_embeddings.shape)
        

        projected_ts_embeddings = self.ts_projection_head(ts_embeddings)
        projected_text_embeddings = self.text_projection_head(final_text_embeddings)

        print("Projected dims: ========================================================================")
        print(projected_text_embeddings)
        print(projected_text_embeddings.shape)

        return projected_ts_embeddings, projected_text_embeddings

    def predict(self, ts_data, input_ids, attention_mask):
        ts_embeddings, text_embeddings = self.forward(ts_data, input_ids, attention_mask)
        ts_embeddings = ts_embeddings.cpu().detach().numpy()
        text_embeddings = text_embeddings.cpu().detach().numpy()
        return cosine_similarity_custom(ts_embeddings, text_embeddings)