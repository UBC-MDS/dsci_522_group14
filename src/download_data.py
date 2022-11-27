# author: Lennon Au-Yeung
# date: 2022-11-20


"""Downloads data csv data from the web to the raw data file as csv file format.
Usage: src/down_data.py --out_type=<out_type> --url=<url> --out_file=<out_file>
Options:
--out_type=<out_type>    Type of file to write locally (script supports either feather or csv)
--url=<url>              URL from where to download the data (must be in standard csv format)
--out_file=<out_file>    Path (including filename) of where to locally write the file
"""
  
from docopt import docopt
import requests
import os
import pandas as pd

opt = docopt(__doc__)

def main(out_type, url, out_file):
    try: 
        request = requests.get(url)
        request.status_code == 200
    except Exception as req:
        print("Website at the provided url does not exist.")
        print(req)

    data = pd.read_csv(url)
    #path= '../data/raw/'
    if out_type == "csv":
        try:
          data.to_csv(out_file, index = False)
        except:
          os.makedirs(os.path.dirname(out_file))
          data.to_csv(out_file, index = False)
    
    assert os.path.isfile(out_file), "Dataset is not in the data/raw directory." 


if __name__ == "__main__":
  main(opt["--out_type"], opt["--url"], opt["--out_file"])


#adopted from https://github.com/ttimbers/breast_cancer_predictor/blob/master/src/download_data.py