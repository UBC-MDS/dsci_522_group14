# author: Shirley Zhang 
# date: 2022-11-25

'''...

Usage: fit_maternal_risk_predict_model.py --train_df_path=<train_df_path> --test_df_path=<test_df_path> --output_dir_path=<output_dir_path>


Options:  
--train_df_path=<train_df_path>     Path to the train_df.csv file 
--test_df_path=<test_df_path>       Path to the test_df.csv file 
--output_dir_path=<output_dir_path> Path to directory where outputs will be stored
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
def main(train_df_path, test_df_path, output_dir_path):
    
    print('Main function')
    
    # 1) Load train and test files into dataframes 
    train_df, test_df = load_train_test_df(train_df_path, test_df_path)
    
    # 2) Split train and test into X and y
    X_train, y_train, X_test, y_test = split_X_y(train_df, test_df)
    
    # 3) Run basic comparison of multiple classification models, save table as .csv 
    compare_models(X_train, y_train, output_dir_path)
    
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

def compare_models(X_train, y_train, output_dir_path):
    
    model_comparison_dict = {}
    
    # a) Dummy Classifier 
    dc = DummyClassifier()
    dc.fit(X_train, y_train) 
    train_score = round(dc.score(X_train, y_train), 3)
    mean_cv_score = round(cross_val_score(dc, X_train, y_train).mean(), 3)
    model_comparison_dict['Dummy'] = {'Training Score': train_score,
                                      'Mean Cross Validation Score': mean_cv_score}
    
    # b) Decision Tree
    decisiontree_clf = DecisionTreeClassifier(random_state=123)
    decisiontree_clf_pipe = make_pipeline(
        StandardScaler(), decisiontree_clf
    )
    decisiontree_clf_pipe.fit(X_train, y_train)
    train_score = round(decisiontree_clf_pipe.score(X_train, y_train), 3)
    mean_cv_score = round(cross_val_score(decisiontree_clf_pipe, X_train, y_train).mean(), 3)
    model_comparison_dict['Decision Tree'] = {'Training Score': train_score,
                                              'Mean Cross Validation Score': mean_cv_score}

    # c) SVM 
    svc_clf = SVC(random_state=123)
    svc_clf_pipe = make_pipeline(
        StandardScaler(), svc_clf
    )
    svc_clf_pipe.fit(X_train, y_train)
    train_score = round(svc_clf_pipe.score(X_train, y_train), 3)
    mean_cv_score = round(cross_val_score(svc_clf_pipe, X_train, y_train).mean(), 3)
    model_comparison_dict['SVM'] = {'Training Score': train_score,
                                    'Mean Cross Validation Score': mean_cv_score}
    
    # d) Logistic Regression
    lg_clf = LogisticRegression(random_state=123)
    lg_clf_pipe = make_pipeline(
        StandardScaler(), lg_clf
    )
    lg_clf_pipe.fit(X_train, y_train)
    train_score = round(lg_clf_pipe.score(X_train, y_train), 3)
    mean_cv_score = round(cross_val_score(lg_clf_pipe, X_train, y_train).mean(), 3)
    model_comparison_dict['Logistic Regression'] = {'Training Score': train_score,
                                                    'Mean Cross Validation Score': mean_cv_score}
    
    # e) K-Nearest Neighbors 
    knn_clf = KNeighborsClassifier()
    knn_clf_pipe = make_pipeline(
        StandardScaler(), knn_clf
    )
    knn_clf_pipe.fit(X_train, y_train)
    train_score = round(knn_clf_pipe.score(X_train, y_train), 3)
    mean_cv_score = round(cross_val_score(knn_clf_pipe, X_train, y_train).mean(), 3)
    model_comparison_dict['K-Nearest Neighbors'] = {'Training Score': train_score,
                                                    'Mean Cross Validation Score': mean_cv_score}
    
    # Save as csv 
    model_comparison_df = pd.DataFrame(model_comparison_dict)
    output_file = output_dir_path + 'model_comparison_table.csv'
    model_comparison_df.to_csv(output_file)
    return 

    
if __name__ == "__main__":
    main(opt['--train_df_path'], opt['--test_df_path'], opt['--output_dir_path'])