# Maternal Health Risk Predictor

## Authors 

- Lennon Au-Yeung
- Chenyang Wang
- Shirley Zhang

(Team 14) 

This data analysis project was created in fulfillment of the team project requirements for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia. 

## About

In this project, we propose a Decision Tree classification model to predict whether an individual may be at low, mid, or high maternal health risk given some information about their age and health. Our final chosen model had a max depth of 29, and performed relatively well on unseen data with 203 observations. The test score was 0.823, with 53 out of 60 high risk targets predicted correctly. However, further steps can be taken to improve the model, such as tuning or other hyperparameters or grouping the target classes into high risk and 'other'. 

The full data set was sourced from the UCI Machine Learning Repository (Dua and Graff 2017), and can be found [here](https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set). A .csv format of the data can be directly downloaded using [this link](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv). The data can be attributed to Marzia Ahmed (Daffodil International University, Dhaka, Bangladesh) and Mohammod Kashem (Dhaka University of Science and Technology, Gazipur, Bangladesh) (Ahmed and Kashem, 2020).  

## Usage 

To replicate the analysis done in this project, you follow the steps below:

1. Install the dependencies listed under "Dependencies"

- [Dependencies](https://github.com/UBC-MDS/maternal_health_risk_predictor#dependencies)

2. Clone the repository (the following shows cloning through ssh keys):

```
git clone git@github.com:UBC-MDS/maternal_health_risk_predictor.git
```

3. Move to the cloned directory:

```
cd maternal_health_risk_predictor
```

4. Run the following commands at the command line/terminal from the root of the directory of this project.
```
#Download the full data set from [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/) and save locally it as 'maternal_risk.csv' under the `data/raw/` directory: 
python src/download_data.py --out_type='csv' --url='https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv' --out_file='data/raw/maternal_risk.csv'


#Render the exploratory data analysis file:
jupyter nbconvert --execute --to notebook --inplace src/maternal_risk_eda.ipynb


#Preprocess the downloaded raw data set (from step 4) and save it as 'train_df.csv' and 'test_df.csv' under the `data/processed/` directory:
python src/pre_processing.py --data_location='data/raw/maternal_risk.csv' --output_location='data/processed/'


#Read the full dataset, perform exploratory data analysis, and save it as 'EDA.png' under the `src/maternal_risk_eda_figures` directory:
python src/eda_script.py --data_location='data/raw/maternal_risk.csv' --output_location='src/maternal_risk_eda_figures/'


#Run the script to fit and model a Decision Tree classifier on the training and test data (from step 6), and save results under the `src/maternal_risk_model_figures/` directory:
python src/fit_maternal_risk_predict_model.py --train_df_path='data/processed/train_df.csv' --test_df_path='data/processed/test_df.csv' --output_dir_path='src/maternal_risk_model_figures/'


#Render the final report under the `/doc` directory:
Rscript -e "rmarkdown::render('doc/final_report.Rmd')"
```

## Dependencies 

Python 3.10 and Python packages:
- docopt==0.6.2
- pandas==1.5.1
- numpy==1.23.5
- re==2022.10.31
- altair==4.2.0
- altair_saver==0.5.0
- requests==2.22.0
- vl_convert
- graphviz==0.20.1
- nbconvert

Model building packages: 
- sklearn==1.1.3
- scipy==1.3.2

R version 4.2.1 and R packages: 
- knitr==1.26
- tidyverse==1.2.1
- kableExtra==1.3.4


## EDA and Final Report 

Link to initial exploratory data analysis in Markdown: [maternal_risk_eda.md](https://github.com/UBC-MDS/maternal_health_risk_predictor/blob/main/src/maternal_risk_eda.md)

Link to final report in Markdown: [final_report.md](https://github.com/UBC-MDS/maternal_health_risk_predictor/blob/main/doc/final_report.md)


## License

The Maternal Health Risk Predictor materials are licensed under the Creative Commons Attribution 2.5 Canada License (CC BY 2.5 CA). If re-using/re-mixing please provide attribution and link to this webpage.


Further license information can be viewed in the `LICENSE` file in the root folder of this repository.


## Attributions 

The data set is attributed to Marzia Ahmed and Mohammod Kashem (Ahmed and Kashem, 2020) as well as the UCI Machine Learning Repository (Dua and Graff 2017). 

## References 

Ahmed, M., Kashem, M.A., Rahman, M., Khatun, S. (2020). Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT). In: , *et al.* InECCE2019. Lecture Notes in Electrical Engineering, vol 632. Springer, Singapore. https://doi.org/10.1007/978-981-15-2317-5_30

Ahmed, M. and Kashem, M. A. (2020). IoT Based Risk Level Prediction Model For Maternal Health Care In The Context Of Bangladesh. In: 2nd International Conference on Sustainable Technologies for Industry 4.0 (STI), 2020, pp. 1-6. https://doi.org/10.1109/STI50764.2020.9350320

Dua, Dheeru, and Casey Graff. (2017). “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.

WHO. (2019). "Maternal mortality". World Health Organization. https://www.who.int/news-room/fact-sheets/detail/maternal-mortality