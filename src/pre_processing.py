# author: Lennon Au-Yeung
# date: 2022-11-24

"""Splits data in train and test format, one set of train set has three classes(high, mid, low) and the other has binary classes(high, mid and low).
Usage: src/pre_processing.py --data_location=<data_location> --output_location=<output_location>
Options:
--data_location=<data_location>    Location of the data to be preprocessed
output_location=<output_location>  Location to output the train and test file
"""
  
from docopt import docopt
import requests
import os
import pandas as pd
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(data_location, output_location):

    maternal_risk_df = pd.read_csv(data_location)
    
    train_df, test_df = train_test_split(maternal_risk_df, test_size=0.2, random_state=123)  
    
    try:
        train_df.to_csv(output_location+'train_df.csv', index = False)
    except:
        os.makedirs(os.path.dirname(output_location+'train_df.csv'))
        train_df.to_csv(output_location+'train_df.csv', index = False)
    
    try:
        test_df.to_csv(output_location+'test_df.csv', index = False)
    except:
        os.makedirs(os.path.dirname(output_location+'test_df.csv'))
        test_df.to_csv(output_location+'test_df.csv', index = False)
        
    assert os.path.isfile(output_location+'train_df.csv'), "Train data is not in the data/processed directory." 
    assert os.path.isfile(output_location+'test_df.csv'), "Test data is not in the data/processed directory." 
    
    maternal_risk_df.loc[maternal_risk_df['RiskLevel'] == 'mid risk', 'RiskLevel'] = 'low and mid risk'
    maternal_risk_df.loc[maternal_risk_df['RiskLevel'] == 'low risk', 'RiskLevel'] = 'low and mid risk'
    
    train_df, test_df = train_test_split(maternal_risk_df, test_size=0.2, random_state=123)
    
    try:
        train_df.to_csv(output_location+'train_df_binary.csv', index = False)
    except:
        os.makedirs(os.path.dirname(output_location+'train_df_binary.csv'))
        train_df.to_csv(output_location+'train_df_binary.csv', index = False)
        
    try:
        test_df.to_csv(output_location+'test_df_binary.csv', index = False)
    except:
        os.makedirs(os.path.dirname(output_location+'test_df_binary.csv'))
        test_df.to_csv(output_location+'test_df_binary.csv', index = False)    
    

if __name__ == "__main__":
  main(opt["--data_location"], opt["--output_location"])

#python src/pre_processing.py --data_location='data/raw/maternal_risk.csv' --output_location='data/processed/'