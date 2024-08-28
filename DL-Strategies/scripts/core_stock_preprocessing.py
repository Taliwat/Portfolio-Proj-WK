# Reading in the appropriate logic and libraries for the code we will use and data we will bring in for this script.
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# Reading in our core_stock_data for preprocessing.
def load_core_stock_data(project_root, file_name='core_stock_data.csv'):
    # Now let's access the main core_stock_data.csv file
    csv_path = os.path.join(project_root, 'data', file_name)
    core_stock_data = pd.read_csv(csv_path, parse_dates=['Date'], index_col= 'Date')
    return core_stock_data

# Define our main variable and read in the data to use in this script.
core_stock_data = load_core_stock_data(project_root)

# Now that we have our data let's address outliers, being strategic about how we do so.
# We will use z_scores for this, adjusting the threshold as necessary.
def outlier_removal(core_stock_data):
    # We want to only select the numeric columns for the calculation of the z_score here.
    numeric_cols = core_stock_data.select_dtypes(include = [np.number])
    
    z_scores = np.abs(stats.zscore(numeric_cols))
    
    # We will set the threshold to 3 std (standard deviation) away, this is common.
    threshold = 3
    
    outliers = (z_scores > threshold)
    
    # Filter out the outliers
    core_stock_data_no_outliers = core_stock_data[(~outliers).all(axis = 1)]
    
    return core_stock_data_no_outliers

# Now we will officially remove outliers from our data
core_stock_data = outlier_removal(core_stock_data)



