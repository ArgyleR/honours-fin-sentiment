# honours-fin-sentiment

## Structure
- data - data files to be formatted for each search
- checkpoint - .pth files for saved models
- data_helper_v2.py
- model_helper.py
- grid_search.py

### data_helper_v2.py

This is the data formatting helper file that will dynamically organise the data including how to concatenate text, how many days of text/time series windows and how to create negative pairs.

### model_helper.py

This is the model helper that contains both the train and validate code as well as the model construction methods so that different underlying encoders can be created on the fly.

### grid_search.py

This is the highest level grid searching file that will take a series of base encoders, dataset, hyper-parameters etc and run a search over all of it.