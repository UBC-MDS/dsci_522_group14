# author: Shirley Zhang 
# date: 2022-11-25

'''Script which takes as input the preprocessed training and test .csv files as well as a path to the output directory, reports scores of baseline classification models, and performs hyperparameter tuning on a Decision Tree model. It will output a table called 'model_comparison_table.csv' with scores of different models, and a figure called 'hyperparam_plot.png' with details about hyperparameter optimization. 


Usage: src/fit_maternal_risk_predict_model.py --train_df_path=<train_df_path> --test_df_path=<test_df_path> --output_dir_path=<output_dir_path>

Options:  
--train_df_path=<train_df_path>     Path to the train_df.csv file 
--test_df_path=<test_df_path>       Path to the test_df.csv file 
--output_dir_path=<output_dir_path> Path to directory where outputs will be stored
'''

##### Import statements
import pandas as pd
import os
from docopt import docopt
import altair as alt
from altair_saver import save
alt.renderers.enable('mimetype')
import vl_convert as vlc
import pickle
import dataframe_image as dfi
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import randint
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


opt = docopt(__doc__)
    
    
##### Helper functions
def load_split_train_test_df(train_df_path, test_df_path):
    '''
    Takes in the path to the training and test data files, reads them as dataframes, 
    and splits them into X and y. Returns the dataframes X_train, y_train, X_test, and y_test. 
    
    Parameters
    ----------
    train_df_path: str
        path that stores the training set
        
    test_df_path: str
        path that stores the testing set
    
    Returns
    ----------
    X_train: dataframe
        values of all features in the training set 
    
    y_train: dataframe
        values of target in the training set
        
    X_test: dataframe
        values of all features in the testing set 
    
    y_test: dataframe
        values of target in the testing set
    '''
    # Read data into dataframes 
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)
    
    # Split into X and y
    X_train = train_df.drop(columns=['RiskLevel'])
    y_train = train_df['RiskLevel']
    X_test = test_df.drop(columns=['RiskLevel'])
    y_test = test_df['RiskLevel']
    
    return X_train, y_train, X_test, y_test

def compare_models(X_train, y_train, output_dir_path):

    '''
    Takes in the training data and the path to the directory to save the results. 
    Runs 5 different classification models and saves the training and cross-val 
    scores in a table to be exported as a .csv file. 
    
    (models: Dummy Classifier, Decision Tree, SVM, Logistic Regression, KNN)
    
    Parameters
    ----------
    X_train: dataframe
        values of all features in the training set 
    
    y_train: dataframe
        values of target in the training set
    
    output_dir_path: str
        Storing the path of the output directory
    '''
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
    
    try:
        model_comparison_df.to_csv(output_file)
    except:
        os.makedirs(os.path.dirname(output_file))
        model_comparison_df.to_csv(output_file)
        
    #output model comparison as image
    dfi.export(model_comparison_df, output_dir_path + 'model_comparison_table.png')
    
    return 

def decisiontree_hyperparamopt(X_train, y_train):
    '''
    Takes in the training data. Performs hyperparameter optimization for the parameter 
    'max_depth' using randomized search and returns the randomized search object.     
    
    Parameters
    ----------
    X_train: dataframe
        values of all features in the training set 
    
    y_train: dataframe
        values of target in the training set
    
    Returns
    ----------
    random_search: randomsearch
        Randomsearch object from randomizedsearchcv
    '''
    # Pipeline
    dt_pipe = make_pipeline(
        StandardScaler(), DecisionTreeClassifier()
    )
    # Hyperparameters
    param_dist = {
        'decisiontreeclassifier__max_depth': randint(1, 50)
    }
    # Random search
    random_search = RandomizedSearchCV(
        dt_pipe,
        param_dist,
        n_iter=30,
        verbose=1,
        n_jobs=-1,
        random_state=123,
        return_train_score=True, 
        cv=50
    )
    # Fit the model 
    random_search.fit(X_train, y_train)
    
    return random_search
    
def hyperparam_plot(random_search, output_dir_path):

    '''
    
    Takes in the randomized search object created in decisiontree_hyperparamopt() and the 
    path to the directory where outputs will be saved. Creates a plot of cross-validation 
    scores for different hyperparameter values, and saves it as a .png file. 
    
    Parameters
    ----------
    random_search: randomsearch
        Randomsearch object from randomizedsearchcv    
        
    output_dir_path: str
        Storing the path of the output directory    
    '''
    # Create dataframes for plotting
    randomizedsearchcv_results = pd.DataFrame(random_search.cv_results_)[['param_decisiontreeclassifier__max_depth', 'mean_test_score', 'mean_train_score']]
    randomizedsearchcv_results_explode = pd.melt(randomizedsearchcv_results, id_vars=['param_decisiontreeclassifier__max_depth'], value_vars=['mean_test_score', 'mean_train_score'])
    
    # Create plot 
    point_plot = alt.Chart(randomizedsearchcv_results_explode).mark_circle(opacity=0.5).encode(
    alt.X('param_decisiontreeclassifier__max_depth', title='Max Depth', scale=alt.Scale(zero=False)),
    alt.Y('value', title='Score', scale=alt.Scale(zero=False)), 
    color=alt.Color('variable', title=None)
    )
    line_plot = point_plot.mark_line(opacity=0.5)
    combined = point_plot + line_plot
    
    # Save plot as png
    plot_path = output_dir_path + 'hyperparam_plot.png'
    try:
        save_chart(combined, plot_path, 2)
    except:
        os.makedirs(os.path.dirname(plot_path))
        save_chart(combined, plot_path, 2)
    return

def save_chart(chart, filename, scale_factor=1):
    '''
    (Function attributed to Joel Ostblom, UBC)
    
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")

def save_bestmodel_pickle(random_search, output_dir_path):

    '''
    Takes in the randomized search object created in decisiontree_hyperparamopt() and the 
    path to the directory where outputs will be saved. Saves the object as a pickle file. 

    Parameters
    ----------
    random_search: randomsearch
        Randomsearch object from randomizedsearchcv    
        
    output_dir_path: str
        Storing the path of the output directory    
    '''
    path_filename = output_dir_path + 'bestmodel.pkl'
    pickle.dump(random_search, open(path_filename, 'wb'))
    
    return 
        
# Main function 
def main(train_df_path, test_df_path, output_dir_path):
    '''
    Takes in the randomized search object created in decisiontree_hyperparamopt() and the 
    path to the directory where outputs will be saved.    
    
    Parameters
    ----------
    train_df_path: str
        path that stores the training set    
    
    test_df_path: str
        path that stores the training set
    
    output_dir_path: str
        Storing the path of the output directory    
    '''
    
    # 1) Load and split train and test into X and y
    X_train, y_train, X_test, y_test = load_split_train_test_df(train_df_path, test_df_path)
    
    # 2) Run basic comparison of multiple classification models, save table as .csv 
    compare_models(X_train, y_train, output_dir_path)
    
    # 3) Decision Tree hyperparameter optimization 
    random_search = decisiontree_hyperparamopt(X_train, y_train)
    
    # 4) Plot hyperparameters 
    hyperparam_plot(random_search, output_dir_path)
    
    # 5) Save the best model as a pickle 
    save_bestmodel_pickle(random_search, output_dir_path)
    
    #test that the expected output are in the correct directory
    assert os.path.isfile(output_dir_path + 'model_comparison_table.csv'), 'Model Comparison table is not in the directory'
    assert os.path.isfile(output_dir_path + 'hyperparam_plot.png'), 'Hyperparameter plot file is not in the directory'
    assert os.path.isfile(output_dir_path + 'bestmodel.pkl'), 'Pickle file is not in the directory'

    
    
if __name__ == "__main__":
    main(opt['--train_df_path'], opt['--test_df_path'], opt['--output_dir_path'])