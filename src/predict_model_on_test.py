# author: Shirley Zhang 
# date: 2022-12-01

'''Script which takes as input the best model (as a .pkl file) from hyperparameter optimization, as well as the test data. It computes the accuracy score of the model on the test data, as well as a confusion matrix. 

Usage: src/predict_model_on_test.py --bestmodel_path=<bestmodel_path> --test_df_path=<test_df_path> --output_dir_path=<output_dir_path>

Options:  
--bestmodel_path=<bestmodel_path>   Path to the bestmodel.pkl file 
--test_df_path=<test_df_path>       Path to the test_df.csv file 
--output_dir_path=<output_dir_path> Path to directory where outputs will be stored
'''

# Import statements
import pandas as pd
import pickle
from docopt import docopt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

opt = docopt(__doc__)

# Main function 
def main(bestmodel_path, test_df_path, output_dir_path):
    
    # Split test dataframe into X and y
    X_test, y_test = split_test_df(test_df_path)
    
    # Score the model on the test data
    test_score_df, random_bestmodel = get_test_score(X_test, y_test, bestmodel_path)
    
    # Save to a .csv file 
    save_testscore_csv(test_score_df, output_dir_path)
    
    # Create a confusion matrix
    create_confusionmatrix(X_test, y_test, random_bestmodel, output_dir_path)
    
def split_test_df(test_df_path):
    
    # Read data into dataframes 
    test_df = pd.read_csv(test_df_path)
    
    # Split into X and y
    X_test = test_df.drop(columns=['RiskLevel'])
    y_test = test_df['RiskLevel']
    
    return X_test, y_test
    
def get_test_score(X_test, y_test, bestmodel_path):
    
    # Load pickle file 
    random_bestmodel = pickle.load(open(bestmodel_path, 'rb'))
    
    # Score on test data 
    result = random_bestmodel.score(X_test, y_test)
    
    # Save score as a dataframe 
    test_score_dict = {}
    test_score_dict['test_score'] = [result]
    test_score_df = pd.DataFrame(test_score_dict) 

    return test_score_df, random_bestmodel
     
def save_testscore_csv(test_score_df, output_dir_path):

    output_file = output_dir_path + 'test_score.csv'
    test_score_df.to_csv(output_file)
    
    return 

def create_confusionmatrix(X_test, y_test, random_bestmodel, output_dir_path):
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, random_bestmodel.predict(X_test))
    cm_df = pd.DataFrame(data = cm, 
                 index = ['True High Risk', 'True Low Risk', 'True Mid Risk'],
                 columns = ['Predicted High Risk', 'Predicted Low Risk', 'Predicted Mid Risk'])
    
    # Save as csv 
    output_file = output_dir_path + 'testdata_confusion_matrix.csv'
    cm_df.to_csv(output_file)
    
    return 
    

if __name__ == "__main__":
    main(opt['--bestmodel_path'], opt['--test_df_path'], opt['--output_dir_path']) 