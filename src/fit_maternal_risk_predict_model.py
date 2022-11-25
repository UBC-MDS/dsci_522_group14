# author: Shirley Zhang 
# date: 2022-11-25

'''...

Usage: fit_maternal_risk_predict_model.py --train_df_path=<train_df_path> --test_df_path=<test_df_path> 


Options:  
--train_df_path=<train_df_path>     Path to the train_df.csv file 
--test_df_path=<test_df_path>       Path to the test_df.csv file 
'''

# Import statements
import pandas as pd
import numpy as np
import re
import graphviz
from docopt import docopt
import altair as alt
from altair_saver import save
alt.renderers.enable('mimetype')

from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


opt = docopt(__doc__)

# Main function 
def main(train_df_path, test_df_path):
    
    print('Main function')
    
    # 1) Load train and test files into dataframes 
    train_df, test_df = load_train_test_df(train_df_path, test_df_path)
    
    # 2) Split train and test into X and y
    X_train, y_train, X_test, y_test = split_X_y(train_df, test_df)
    
    # 3) Run basic comparison of multiple classification models, save table as .csv 
    
    
def load_train_test_df(train_df_path, test_df_path):
    
    # Read data into dataframes 
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)
    
    return train_df, test_df 

def split_X_y(train_df, test_df): 
    
    X_train = train_df.drop(columns=['RiskLevel'])
    y_train = train_df['RiskLevel']
    X_test = test_df.drop(columns=['RiskLevel'])
    y_test = test_df['RiskLevel']
    
    return X_train, y_train, X_test, y_test

def compare_models(X_train, y_train):
    
    # a) Dummy Classifier 
    return 0

    
if __name__ == "__main__":
    main(opt['--train_df_path'], opt['--test_df_path'])