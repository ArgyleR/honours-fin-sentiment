# honours-fin-sentiment
This is the repository for Eoin O'Sullivan's Bachelor Honours Thesis on Integrating Contrastive Learning with Time Series Forecasting and Financial Sentiment Analysis.

## Structure
- checkpoint_final - .pth files for saved models of the final run
- data - data files to be formatted for experimentation
- images - result graphs and data analysis images
- results - json and csv format of results. Json contain full results, csv contains the best synthesised results for each experiment (best validation scores etc)
- utils - contains code for conducting baseline experiments, text selection and testing of data set creation
- data_helper_v3.py - the python file used to process data on the fly
- general_analytics.ipybn - used for creating visuals of data
- grid_search.py - high level code for conducting grid searches
- model_helper.py - helper python file with model training code
- models.py - model selection helper, contains the model architectures
- pivot_analytics.ipynb - used for creating visuals of model performance

### data_helper_v3.py

This is the data formatting helper file that will dynamically organise the data including how to concatenate text, how many days of text/time series windows and how to create negative pairs.

All datasets constructed and used for training can be found here: https://drive.google.com/drive/folders/1IEvaw-h1SGS7w4hcpmuNSOfzkdTds5Db?usp=sharing

### model_helper.py

This is the model helper that contains both the train and validate code as well as the model construction methods so that different underlying encoders can be created on the fly.

### grid_search.py

This is the highest level grid searching file that will take a series of base encoders, dataset, hyper-parameters etc and run a search over all of it.