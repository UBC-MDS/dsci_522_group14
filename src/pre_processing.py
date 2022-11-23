from docopt import docopt
import requests
import os
import pandas as pd
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(data_location):
    
    maternal_risk_df = pd.read_csv(data_location)
    
    train_df, test_df = train_test_split(maternal_risk_df, test_size=test_size, random_state=123)  

if __name__ == "__main__":
  main(opt["--data_location"])